# roopディレクトリへ移動
def sys_path_to_root():
    import os, sys
    ROOT_PATH = 'rome'
    # スクリプトのディレクトリパスを取得
    script_directory = os.path.dirname(os.path.abspath(__file__))
    # rootディレクトリのパスを計算
    root_directory = os.path.abspath(os.path.join(script_directory, ROOT_PATH))
    sys.path.append(root_directory)
sys_path_to_root()

import os
print(os.getcwd())

import numpy, os
import pandas as pd
from matplotlib import pyplot as plt
import math
import datetime
import torch
from collections import defaultdict
import nethook
from knowns import KnownsDataset
from globals import DATA_DIR
from causal_trace import (
    ModelAndTokenizer,
    layername,
    plot_trace_heatmap,
    make_inputs,
    decode_tokens,
    find_token_range,
    predict_from_input,
    collect_embedding_std,
)

# all_flow_dataの配列が省略されることがあるので、それ対策
torch.set_printoptions(threshold=10_000)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

# Uncomment the architecture to plot.
# arch = "gpt2-xl"
# archname = "GPT-2-XL"

# arch = 'EleutherAI_gpt-j-6B'
# arch = 'EleutherAI_gpt-j-6B_original'
# archname = 'GPT-J-6B'

# arch = "naclbit_gpt-j-japanese-6.8b"
# archname = 'GPT-J-japanese-6B'

# arch = 'rinna_japanese-gpt-neox-3.6b-instruction-sft'
# arch = "rinna_japanese-gpt-neox-3.6b_add_prompt"
# arch = "rinna_japanese-gpt-neox-3.6b"
# archname = 'GPT-NEOX-3.6B'

# arch = "matsuo-lab_weblab-10b"
# arch = "cyberagent_open-calm-7b"
# archname = 'GPT-NEOX'

# arch = "rinna_japanese-gpt2-medium"
# archname = "GPT-2-MEDIUM"

# arch = 'EleutherAI_gpt-neox-20b'
# archname = 'GPT-NeoX-20B'

arch = "Ryoma0302/gpt_0.76B_global_step20000_japanese"
archname = "transformer"

dt_now = datetime.datetime.now()
# data_len = 1000

torch.set_grad_enabled(False)
# model_name = "gpt2-xl"
# model_name = "EleutherAI/gpt-j-6B"
# model_name = "rinna/japanese-gpt-neox-3.6b-instruction-sft"
# model_name = "rinna/japanese-gpt-neox-3.6b"
# model_name = "rinna/japanese-gpt2-medium"
# model_name = "naclbit/gpt-j-japanese-6.8b"
# model_name = "matsuo-lab/weblab-10b"
# model_name = "cyberagent/open-calm-7b"
model_name = "Ryoma0302/gpt_0.76B_global_step20000_japanese"
'''''
使うときは,
experiments.causal_traceのpredict_from_input
char_loc = whole_string.index(substring)
p, preds = probs[0, o_index], torch.Tensor(o_index).int()
を書き換える。
'''''
mt = ModelAndTokenizer(
    model_name,
    # low_cpu_mem_usage=IS_COLAB,
    torch_dtype=(torch.float16 if "20b" in model_name else None),
)

# CSVファイルのパス
# csv_file_path = 'data/text_data_converted_to_csv.csv'
csv_file_path = "data/en2jp_data.csv"
df = pd.read_csv(csv_file_path)
# df = df.dropna()
data_len = len(df)

knowns = KnownsDataset(DATA_DIR)  # Dataset of known facts
noise_level = 3 * collect_embedding_std(mt, [k["subject"] for k in knowns])
print(f"Using noise level {noise_level}")

# change_prompt_client = ChangePrompt()

class Avg:
    def __init__(self):
        self.d = []

    def add(self, v):
        self.d.append(v[None])

    def add_all(self, vv):
        self.d.append(vv)

    def avg(self):
        if self.d != []:
            return numpy.concatenate(self.d).mean(axis=0)
        else:
            return numpy.array([0])

    def std(self):
        if self.d != []:
            return numpy.concatenate(self.d).std(axis=0)
        else:
            return numpy.array([0])

    def size(self):
        return sum(datum.shape[0] for datum in self.d)
def trace_with_patch(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    trace_layers=None,  # List of traced outputs to return
):
    prng = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                x[1:, b:e] += noise * torch.from_numpy(
                    prng.randn(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device)
            return x
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers
    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layername] + list(patch_spec.keys()) + additional_layers,
        edit_output=patch_rep,
    ) as td:
        outputs_exp = model(**inp)

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return probs, all_traced

    return probs

def calculate_hidden_flow(
    mt, prompt, subject, o="Seattle", samples=10, noise=0.1, window=10, kind=None
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1))
    with torch.no_grad():
        answer_t, base_score = [d[0] for d in predict_from_input(mt.model, mt.tokenizer, inp, o)]
    [answer] = decode_tokens(mt.tokenizer, [answer_t])
    print(subject)
    e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
    low_score = trace_with_patch(
        mt.model, inp, [], answer_t, e_range, noise=noise
    ).item()
    if not kind:
        differences = trace_important_states(
            mt.model, mt.num_layers, inp, e_range, answer_t, noise=noise
        )
    else:
        differences = trace_important_window(
            mt.model,
            mt.num_layers,
            inp,
            e_range,
            answer_t,
            noise=noise,
            window=window,
            kind=kind,
        )
    differences = differences.detach().cpu()
    return dict(
        scores=differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0],
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        subject_range=e_range,
        answer=answer,
        window=window,
        kind=kind or "",
    )


def trace_important_states(model, num_layers, inp, e_range, answer_t, noise=0.1):
    ntoks = inp["input_ids"].shape[1]
    table = []
    for tnum in range(ntoks):
        row = []
        for layer in range(0, num_layers):
            r = trace_with_patch(
                model,
                inp,
                [(tnum, layername(model, layer))],
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def trace_important_window(
    model, num_layers, inp, e_range, answer_t, kind, window=10, noise=0.1
):
    ntoks = inp["input_ids"].shape[1]
    table = []
    for tnum in range(ntoks):
        row = []
        for layer in range(0, num_layers):
            layerlist = [
                (tnum, layername(model, L, kind))
                for L in range(
                    max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
                )
            ]
            r = trace_with_patch(
                model, inp, layerlist, answer_t, tokens_to_mix=e_range, noise=noise
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)

def plot_hidden_flow(
    mt,
    prompt,
    subject=None,
    o="Seattle",
    samples=10,
    noise=0.1,
    window=10,
    kind=None,
    modelname=None,
    savepdf=None,
):
    result = calculate_hidden_flow(
        mt, prompt, subject, o, samples=samples, noise=noise, window=window, kind=kind
    )
    plot_trace_heatmap(result, savepdf, modelname=modelname)
    return result


def plot_all_flow(mt, prompt, subject=None, o="Seattle", noise=0.1, modelname=None, savepdf=None, kind=None):
    if kind is None:
        savepdf=f"hidden_{savepdf}"
    else:
        savepdf=f"{kind}_{savepdf}"
    result = plot_hidden_flow(
        mt, prompt, subject, o, modelname=modelname, noise=noise, kind=kind, savepdf=f'result_pdf/{dt_now}/{savepdf}'
    )
    return result

def read_knowlege(count=150, kind=None, arch="gpt2-xl"):
    (
        avg_fe,
        avg_ee,
        avg_le,
        avg_fa,
        avg_ea,
        avg_la,
        avg_hs,
        avg_ls,
        avg_fs,
        avg_fle,
        avg_fla,
    ) = [Avg() for _ in range(11)]
    all_flow_data = []
    prompt_templates = [
"""###タスク###
質問に対して一問一答で回答してください．
###例###
入力：日本で一番高い山は何ですか？
出力：富士山です．
入力：日本の首都はどこですか？
出力：東京
入力：""",
"出力："]
    # for i, knowledge in enumerate(knowns[:data_len]):
    for i, knowledge in df[:data_len].iterrows():
        # prompt = knowledge["prompt"] # 穴埋め形式の英語
        new_prompt = knowledge["prompt"] # 質問形式の日本語
        # new_prompt = knowledge["new_prompt"] # 質問形式の英語
        subject = knowledge["subject"]
        attribute = knowledge["attribute"]
        # new_prompt = change_prompt_client.send(prompt, subject, attribute)
        # print(f'prompt: {prompt}')
        print(f'subject: {subject}')
        print(f'attribute: {attribute}')
        print(f'new_prompt: {new_prompt}')
        # data = plot_all_flow(mt, prompt=prompt, subject=knowledge["subject"], o=knowledge["attribute"], noise=noise_level, savepdf=f'result_pdf/{i}', kind=kind)
        # data = plot_all_flow(mt, prompt=prompt_templates[0]+new_prompt+prompt_templates[1], subject=knowledge["subject"], o=knowledge["attribute"], noise=noise_level, savepdf=f'result_pdf/{i}', kind=kind)
        # data = plot_all_flow(mt, prompt=new_prompt, subject=knowledge["subject"], o=knowledge["attribute"], noise=noise_level, savepdf=f'result_pdf/{i}', kind=kind)
        try:
            data = plot_all_flow(mt, prompt=new_prompt, subject=knowledge["subject"], o=knowledge["attribute"], noise=noise_level, savepdf=f'result_pdf/{i}', kind=kind)
        except:
            continue
        all_flow_data.append(data)
        # # (n,36,1)→(n,36)に変更する必要があった
        # scores = data["scores"].squeeze().to('cpu')
        scores = data["scores"].to('cpu')
        first_e, first_a = data["subject_range"]
        last_e = first_a - 1
        last_a = len(scores) - 1
        # original prediction
        avg_hs.add(data["high_score"].to('cpu'))
        # prediction after subject is corrupted
        avg_ls.add(torch.tensor(data["low_score"]))
        avg_fs.add(scores.max())
        # some maximum computations
        avg_fle.add(scores[last_e].max())
        avg_fla.add(scores[last_a].max())
        # First subject middle, last subjet.
        avg_fe.add(scores[first_e])
        avg_ee.add_all(scores[first_e + 1 : last_e])
        avg_le.add(scores[last_e])
        # First after, middle after, last after
        try:
            avg_fa.add(scores[first_a])
            avg_ea.add_all(scores[first_a + 1 : last_a])
            avg_la.add(scores[last_a])
        except:
            avg_fa.add(scores[-1])
            avg_ea.add_all(scores[-1 : ])
            avg_la.add(scores[-1])

    result = numpy.stack(
        [
            avg_fe.avg(),
            avg_ee.avg(),
            avg_le.avg(),
            avg_fa.avg(),
            avg_ea.avg(),
            avg_la.avg(),
        ]
    )
    result_std = numpy.stack(
        [
            avg_fe.std(),
            avg_ee.std(),
            avg_le.std(),
            avg_fa.std(),
            avg_ea.std(),
            avg_la.std(),
        ]
    )
    print("Average Total Effect", avg_hs.avg() - avg_ls.avg())
    print(
        "Best average indirect effect on last subject",
        avg_le.avg().max() - avg_ls.avg(),
    )
    print(
        "Best average indirect effect on last token", avg_la.avg().max() - avg_ls.avg()
    )
    print("Average best-fixed score", avg_fs.avg())
    print("Average best-fixed on last subject token score", avg_fle.avg())
    print("Average best-fixed on last word score", avg_fla.avg())
    print("Argmax at last subject token", numpy.argmax(avg_le.avg()))
    print("Max at last subject token", numpy.max(avg_le.avg()))
    print("Argmax at last prompt token", numpy.argmax(avg_la.avg()))
    print("Max at last prompt token", numpy.max(avg_la.avg()))
    return dict(
        low_score=avg_ls.avg(), result=result, result_std=result_std, size=avg_fe.size(), all_flow_data=all_flow_data
    )

def plot_array(
    differences,
    kind=None,
    savepdf=None,
    title=None,
    low_score=None,
    high_score=None,
    archname="GPT2-XL",
):
    if low_score is None:
        low_score = differences.min()
    if high_score is None:
        high_score = differences.max()
    answer = "AIE"
    labels = [
        "First subject token",
        "Middle subject tokens",
        "Last subject token",
        "First subsequent token",
        "Further tokens",
        "Last token",
    ]

    fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
    h = ax.pcolor(
        differences,
        cmap={None: "Purples", "mlp": "Greens", "attn": "Reds"}[kind],
        # vmin=low_score,
        # vmax=high_score,
    )
    if title:
        ax.set_title(title)
    ax.invert_yaxis()
    ax.set_yticks([0.5 + i for i in range(len(differences))])
    ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
    ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
    ax.set_yticklabels(labels)
    if kind is None:
        ax.set_xlabel(f"single patched layer within {archname}")
    else:
        ax.set_xlabel(f"center of interval of 10 patched {kind} layers")
    cb = plt.colorbar(h)
    # The following should be cb.ax.set_xlabel(answer), but this is broken in matplotlib 3.5.1.
    if answer:
        cb.ax.set_title(str(answer).strip(), y=-0.16, fontsize=10)

    if savepdf:
        os.makedirs(os.path.dirname(savepdf), exist_ok=True)
        plt.savefig(savepdf, bbox_inches="tight")
    plt.show()


the_count = data_len
count = data_len
high_score = None  # Scale all plots according to the y axis of the first plot

for kind in [None, "mlp", "attn"]:
# for kind in ["mlp", "attn"]:
    d = read_knowlege(the_count, kind, arch)
    count = d["size"]
    what = {
        None: "Indirect Effect of $h_i^{(l)}$",
        "mlp": "Indirect Effect of MLP",
        "attn": "Indirect Effect of Attn",
    }[kind]
    title = f"Avg {what} over {count} prompts"
    result = numpy.clip(d["result"] - d["low_score"], 0, None)
    kindcode = "" if kind is None else f"_{kind}"
    if kind not in ["mlp", "attn"]:
        high_score = result.max()
    plot_array(
        result,
        kind=kind,
        title=title,
        low_score=0.0,
        high_score=high_score,
        archname=archname,
        savepdf=f"results/{arch}/causal_trace/summary_pdfs/rollup{kindcode}.pdf",
    )
    df_all_flow_data = pd.DataFrame(d["all_flow_data"])
    df_all_flow_data.to_csv(f"data/all_flow_data/{arch}{kindcode}.csv", index=False)

labels = [
    "First subject token",
    "Middle subject tokens",
    "Last subject token",
    "First subsequent token",
    "Further tokens",
    "Last token",
]
color_order = [0, 1, 2, 4, 5, 3]
x = None

cmap = plt.get_cmap("tab10")
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=True, dpi=200)
for j, (kind, title) in enumerate(
    [
        (None, "single hidden vector"),
        ("mlp", "run of 10 MLP lookups"),
        ("attn", "run of 10 Attn modules"),
    ]
):
    print(f"Reading {kind}")
    # d = read_knowlege(225, kind, arch)
    d = read_knowlege(data_len, kind, arch)
    for i, label in list(enumerate(labels)):
        y = d["result"][i] - d["low_score"]
        if x is None:
            x = list(range(len(y)))
        std = d["result_std"][i]
        error = std * 1.96 / math.sqrt(count)
        axes[j].fill_between(
            x, y - error, y + error, alpha=0.3, color=cmap.colors[color_order[i]]
        )
        axes[j].plot(x, y, label=label, color=cmap.colors[color_order[i]])

    axes[j].set_title(f"Average indirect effect of a {title}")
    axes[j].set_ylabel("Average indirect effect on p(o)")
    axes[j].set_xlabel(f"Layer number in {archname}")
    # axes[j].set_ylim(0.1, 0.3)
axes[1].legend(frameon=False)
plt.tight_layout()
savepdf = f"results/{arch}/causal_trace/summary_pdfs/lineplot-causaltrace.pdf"
# plt.savefig(f"results/{arch}/causal_trace/summary_pdfs/lineplot-causaltrace.pdf")
os.makedirs(os.path.dirname(savepdf), exist_ok=True)
plt.savefig(savepdf)