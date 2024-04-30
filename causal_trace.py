import os
import re
import unicodedata
import torch
import matplotlib.pyplot as plt
import japanize_matplotlib
import nethook
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
set_seed(42) # シード固定

class ModelAndTokenizer:
    """
    GPTスタイルの言語モデルとトークナイザを保持する(または自動的にダウンロードして保持する)オブジェクト
    レイヤーの数を数える
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
            print("tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        if model is None:
            assert model_name is not None
            print("model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)")
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
            # 埋め込み層の名前を返す
            return "transformer.wte"
        # Transformerの各層の名前を返す
        return f'transformer.h.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "gpt_neox"):
        # GPT-NeoXの埋め込み層の名前を返す
        if kind == "embed":
            return "gpt_neox.embed_in"
        # GPT-NeoXのAttntion層の名前を返す
        if kind == "attn":
            kind = "attention"
        # GPT-NeoXの各層の名前を返す
        return f'gpt_neox.layers.{num}{"" if kind is None else "." + kind}'
    # モデルの構造が不明な場合はエラーを発生させる
    assert False, "unknown transformer structure"

# この関数 plot_trace_heatmap は、モデルの内部状態による出力の変化をヒートマップとして可視化する関数
def plot_trace_heatmap(result, savepdf=None, title=None, xlabel=None, modelname=None):
    # resultから必要なデータを取得
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
    # ラベルの一部に "*" を追加して強調表示
    for i in range(*result["subject_range"]):
        labels[i] = labels[i] + "*"
    # Matplotlibのフォントを設定(IPAexGothicだと日本語も表示できた)
    # with plt.rc_context(rc={"font.family": "Times New Roman"}):
    with plt.rc_context(rc={"font.family": "IPAexGothic"}):
        # 図とアクシスを作成
        fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
        # ヒートマップを描画
        h = ax.pcolor(
            differences,
            # cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "attn": "Reds"}[
            #     kind
            # ],
            cmap="Reds",
            vmin=low_score, # これは、vminより少ない値をすべて下限値としてプロットするもの
        )
        # 縦軸を反転
        ax.invert_yaxis()
        # 縦軸と横軸のラベルを設定
        ax.set_yticks([0.5 + i for i in range(len(differences))])
        ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
        ax.set_yticklabels(labels)
        # モデル名が指定されていない場合は "GPT" を使用
        if not modelname:
            modelname = "GPT"
        # kindが指定されていない場合
        if not kind:
            # ax.set_title("Impact of restoring state after corrupted input")
            # ax.set_xlabel(f"single restored layer within {modelname}")
            ax.set_title("hidden neuron")
            ax.set_xlabel(f"layer number")
        # kindが指定されている場合
        else:
            kindname = "MLP" if kind == "mlp" else "Attn"
            # ax.set_title(f"Impact of restoring {kindname} after corrupted input")
            # ax.set_xlabel(f"center of interval of {window} restored {kindname} layers")
            ax.set_xlabel(f"layer number")
            ax.set_title(f"{kindname} module")
        # カラーバーを追加
        cb = plt.colorbar(h)
        # タイトルが指定されている場合は設定
        if title is not None:
            ax.set_title(title)
        # 横軸ラベルが指定されている場合は設定
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        # answerが指定されている場合は、カラーバーのタイトルに確率を表示
        elif answer is not None:
            # The following should be cb.ax.set_xlabel, but this is broken in matplotlib 3.5.1.
            cb.ax.set_title(f"p({str(answer).strip()})", y=-0.16, fontsize=10)
        # savedfが指定されている場合
        if savepdf:
            # ディレクトリが存在しない場合は作成
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            # 図を保存
            plt.savefig(savepdf, bbox_inches="tight")
            # 図を閉じる
            plt.close()
         # savedfが指定されていない場合
        else:
            # 図を表示
            plt.show()

"""
この関数 make_inputs は、与えられたプロンプト(prompts)をトークナイズし、モデルに入力するための形式に変換する
1. 各プロンプトをトークナイザーでトークン化し、トークンIDのリストに変換します。
2. トークンIDリストの最大長を求めます。
3. パディングトークンのIDを取得します。トークナイザーに "[PAD]" トークンが定義されている場合はその ID を使用し、そうでない場合は 0 を使用します。
4. 各トークンIDリストの長さを最大長に合わせるために、パディングトークンを追加します。
5. アテンションマスクを作成します。パディングトークンは 0、実際のトークンは 1 となるようにマスクを設定します。
6. 入力データを dict 形式で返します。input_ids はパディング済みのトークンIDのテンソル、attention_mask はアテンションマスクのテンソルです。
これらのテンソルは指定されたデバイス(device)に移動されます。
"""
def make_inputs(tokenizer, prompts, device="cuda"):
    # 各promptをトークン化してトークンIDのリストに変換
    token_lists = [tokenizer.encode(p) for p in prompts]
    # トークンIDリストの最大長を求める
    maxlen = max(len(t) for t in token_lists)
    # パディングトークンのIDを取得
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    # 各トークンIDリストにパディングを追加して、最大長に合わせる
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    # アテンションマスクを作成 (パディングは0、トークンは1)
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    # 入力データをdictで返す
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )

# この関数 decode_tokens は、トークンIDの配列をトークナイザーを使用してデコードし、対応するテキストのリストを返す
def decode_tokens(tokenizer, token_array):
    # token_arrayが多次元の場合
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        # 各行に対してdecode_tokensを再帰的に適用
        return [decode_tokens(tokenizer, row) for row in token_array]
    # token_arrayが1次元の場合
    # 各トークンをデコードしてリストに格納
    return [tokenizer.decode(t) for t in token_array]

def find_token_range(tokenizer, token_array, substring):
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

def predict_from_input(model, tokenizer, inp, o="Seattle"):
    o_index = tokenizer.encode(o) # もとのコード
    # o_index = tokenizer.encode(o)[0] # 謎だが、りんなgptは配列の要素が2個あったので、とりあえず、1個目を使う。
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
    # p, preds =  torch.max(probs, dim=1) # もとのコード
    p, preds = probs[0, o_index], torch.Tensor(o_index).int() # 目的のオブジェクト(O)のロジットを確認するため
    # p, preds = probs[0, o_index], torch.Tensor([o_index]).int() # 日本語用：目的のオブジェクト(O)のロジットを確認するため
    # p = p.unsqueeze(0) # りんなGPTのときだけON
    # import pdb;pdb.set_trace()
    print("preds:" + str(preds))
    print("p:" + str(p))
    return preds, p

"""
1. 各subjectをトークナイザーで処理し、モデルの入力形式に変換します。
2. PyTorchモデル(mt.model)の最初の埋め込み層の出力をトレースします。
3. トレースした埋め込み層の出力を、alldata というリストに追加します。
4. 全subjectの埋め込みベクトルを結合します。
5. 結合された埋め込みベクトルの標準偏差を計算します。
6. 計算された標準偏差(noise_level)を返します。

この関数は、与えられたsubjectsの埋め込みベクトルのばらつきを定量化するために使用されます。
標準偏差が大きいほど、埋め込みベクトルのばらつきが大きいことを示しています。
これは、モデルがsubjects間の差異をどの程度捉えているかを評価するための指標になります。
"""
def collect_embedding_std(mt, subjects):
    alldata = []
    for s in subjects:
        # 各subjectをトークナイズしてモデルの入力形式に変換
        inp = make_inputs(mt.tokenizer, [s])
        # モデルの最初の埋め込み層の出力をトレース
        with nethook.Trace(mt.model, layername(mt.model, 0, "embed")) as t:
            mt.model(**inp)
            # トレースした埋め込み層の出力をalldataに追加
            alldata.append(t.output[0])
    # 全subjectの埋め込みベクトルを結合
    alldata = torch.cat(alldata)
    # 埋め込みベクトルの標準偏差を計算
    noise_level = alldata.std().item()
    return noise_level