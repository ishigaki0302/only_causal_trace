import argparse
import json
import os
import re
import unicodedata
from collections import defaultdict

import numpy
import torch
from datasets import load_dataset
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from knowns import KnownsDataset
from tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)
import nethook
from globals import DATA_DIR
from runningstats import Covariance, tally

import matplotlib.pyplot as plt
import japanize_matplotlib
# plt.rcParams['font.family'] = 'Hiragino Maru Gothic Pro'
# print("/rome/experiments/causal_trace.py 28")

class ModelAndTokenizer:
    """
    An object to hold on to (or automatically download and hold)
    a GPT-style language model and tokenizer.  Counts the number
    of layers.
    """
    def __init__(
        self,
        model_name=None,
        model=None,
        tokenizer=None,
        low_cpu_mem_usage=False,
        torch_dtype=None,
    ):
        if tokenizer is None:
            assert model_name is not None
            print("experiments/causal_trace 467")
            # tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        if model is None:
            assert model_name is not None
            print("experiments/causal_trace 472")
            # model = AutoModelForCausalLM.from_pretrained(
            #     model_name, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype
            # )
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
            nethook.set_requires_grad(False, model)
            model.eval().cuda()
        self.tokenizer = tokenizer
        self.model = model
        self.layer_names = [
            n
            for n, m in model.named_modules()
            if (re.match(r"^(transformer|gpt_neox)\.(h|layers)\.\d+$", n))
        ]
        self.num_layers = len(self.layer_names)
    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )

def layername(model, num, kind=None):
    if hasattr(model, "transformer"):
        if kind == "embed":
            return "transformer.wte"
        return f'transformer.h.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "gpt_neox"):
        if kind == "embed":
            return "gpt_neox.embed_in"
        if kind == "attn":
            kind = "attention"
        return f'gpt_neox.layers.{num}{"" if kind is None else "." + kind}'
    assert False, "unknown transformer structure"

def guess_subject(prompt):
    return re.search(r"(?!Wh(o|at|ere|en|ich|y) )([A-Z]\S*)(\s[A-Z][a-z']*)*", prompt)[
        0
    ].strip()

def plot_trace_heatmap(result, savepdf=None, title=None, xlabel=None, modelname=None):
    differences = result["scores"]
    differences = differences.view(differences.size()[0],differences.size()[1])
    low_score = result["low_score"]
    answer = result["answer"]
    kind = (
        None
        if (not result["kind"] or result["kind"] == "None")
        else str(result["kind"])
    )
    window = result.get("window", 10)
    labels = list(result["input_tokens"])
    for i in range(*result["subject_range"]):
        labels[i] = labels[i] + "*"
    # with plt.rc_context(rc={"font.family": "Times New Roman"}):
    with plt.rc_context(rc={"font.family": "IPAexGothic"}):
        fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
        h = ax.pcolor(
            differences,
            # cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "attn": "Reds"}[
            #     kind
            # ],
            cmap="Reds",
            vmin=low_score, # これは、vminより少ない値をすべて下限値としてプロットするもの
        )
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(differences))])
        ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
        ax.set_yticklabels(labels)
        if not modelname:
            modelname = "GPT"
        if not kind:
            # ax.set_title("Impact of restoring state after corrupted input")
            # ax.set_xlabel(f"single restored layer within {modelname}")
            ax.set_title("hidden neuron")
            ax.set_xlabel(f"layer number")
        else:
            kindname = "MLP" if kind == "mlp" else "Attn"
            # ax.set_title(f"Impact of restoring {kindname} after corrupted input")
            # ax.set_xlabel(f"center of interval of {window} restored {kindname} layers")
            ax.set_xlabel(f"layer number")
            ax.set_title(f"{kindname} module")
        cb = plt.colorbar(h)
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        elif answer is not None:
            # The following should be cb.ax.set_xlabel, but this is broken in matplotlib 3.5.1.
            cb.ax.set_title(f"p({str(answer).strip()})", y=-0.16, fontsize=10)
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

# Utilities for dealing with tokens
def make_inputs(tokenizer, prompts, device="cuda"):
    print("/workspace/romeworkspace/rome/experiments/causal_trace.py:610")
    token_lists = [tokenizer.encode(p) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    # position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    # print(prompts)
    # print("old")
    # print(torch.tensor(input_ids))
    # print(torch.tensor(attention_mask))
    # token_ids = tokenizer.encode_plus(prompts[0], add_special_tokens=False, return_attention_mask = True, return_tensors="pt")
    # print("new")
    # print(token_ids["input_ids"])
    # print(token_ids["attention_mask"])
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        #    position_ids=torch.tensor(position_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )
    # return dict(
    #     input_ids=token_ids["input_ids"].to(device),
    #     attention_mask=token_ids["attention_mask"].to(device)
    # )

def decode_tokens(tokenizer, token_array):
    # print(token_array)
    # for t in token_array:
    #     print(t)
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode(t) for t in token_array]

def find_token_range(tokenizer, token_array, substring):
    print("/workspace/romeworkspace/rome/experiments/causal_trace.py:648")
    # 入力文中の主題を探す関数
    toks = decode_tokens(tokenizer, token_array)
    whole_string = "".join(toks)
    print(whole_string)
    print(substring)
    try:
        char_loc = whole_string.index(substring) # もとのコード
    except:
        char_loc = None
    try:
        if char_loc is None:
            char_loc = whole_string.index(substring.replace(" ","")) # 日本語LLMを使うとき用
    except:
        char_loc = None
    try:
        if char_loc is None:
            """""
            ジャン=ピエール・ヴァン・ロッセムはどの国の市民権を持っていますか?</s>                                                                       d
            ジャン＝ピエール・ヴァン・ロッセム
            """""
            char_loc = whole_string.index(substring.replace("＝", "="))
    except:
        char_loc = None
    if char_loc is None:
        """""
        ムアーウィヤ1世はどの宗教と関連していますか?</s>
        ムアーウィヤ１世
        """""
        char_loc = whole_string.index(unicodedata.normalize('NFKC', substring)) # もとのコード
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    return (tok_start, tok_end)

def predict_token(mt, prompts, return_p=False, o="Seattle"):
    inp = make_inputs(mt.tokenizer, prompts)
    preds, p = predict_from_input(mt.model, mt.tokenizer, inp, o)
    result = [mt.tokenizer.decode(c) for c in preds]
    if return_p:
        result = (result, p)
    return result

def predict_from_input(model, tokenizer, inp, o="Seattle"):
    print("/workspace/romeworkspace/rome/experiments/causal_trace.py:650")
    # o_index = tokenizer.encode(o) # もとのコード
    o_index = tokenizer.encode(o)[0] # 謎だが、りんなgptは配列の要素が2個あったので、とりあえず、1個目を使う。
    # 謎ではない！1単語が複数トークンに分かれているだけ！
    # use_fastを使うと，[UNK]トークンとかがいっぱい出てきてしまう．
    # o_indexs = tokenizer.encode(o)
    # o_indexs = [i for i in o_indexs if i not in [263, 3]]
    # o_index = o_indexs
    print(F"o:{o}")
    print(f"o_index:{o_index}")
    # print(f"o_index:{o_indexs}")
    out = model(**inp)["logits"]
    probs = torch.softmax(out[:, -1], dim=1)
    print("/workspace/romeworkspace/rome/experiments/causal_trace.py:681")
    # p, preds =  torch.max(probs, dim=1) # もとのコード
    # p, preds = probs[0, o_index], torch.Tensor(o_index).int() # 目的のオブジェクト(O)のロジットを確認するため
    p, preds = probs[0, o_index], torch.Tensor([o_index]).int() # 日本語用：目的のオブジェクト(O)のロジットを確認するため
    p = p.unsqueeze(0) # りんなGPTのときだけON
    # import pdb;pdb.set_trace()
    print("preds:" + str(preds))
    print("p:" + str(p))
    return preds, p

def collect_embedding_std(mt, subjects):
    alldata = []
    for s in subjects:
        inp = make_inputs(mt.tokenizer, [s])
        with nethook.Trace(mt.model, layername(mt.model, 0, "embed")) as t:
            mt.model(**inp)
            alldata.append(t.output[0])
    alldata = torch.cat(alldata)
    noise_level = alldata.std().item()
    return noise_level