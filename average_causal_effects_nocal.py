import os, re, json
print(os.getcwd())
import numpy, os
import pandas as pd
from matplotlib import pyplot as plt
import math
import datetime
import torch
from read_all_flow_data import read_all_flow_data

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

# Uncomment the architecture to plot.
# arch = "gpt2-xl"
# archname = "GPT-2-XL"

# arch = 'EleutherAI_gpt-j-6B_original'
# arch = 'EleutherAI_gpt-j-6B'
# archname = 'GPT-J-6B'

# arch = 'rinna_japanese-gpt-neox-3.6b-instruction-sft'
# arch = "rinna_japanese-gpt-neox-3.6b"
# archname = 'GPT-NEOX-3.6B'

# arch = "cyberagent_open-calm-7b"
# archname = 'GPT-NEOX'

# arch = 'EleutherAI_gpt-neox-20b'
# archname = 'GPT-NeoX-20B'

arch = "Ryoma0302_gpt_0.76B_global_step20000_japanese"
archname = "transformer"

dt_now = datetime.datetime.now()
data_len = 500
'''''
使うときは,
experiments.causal_traceのpredict_from_input
char_loc = whole_string.index(substring)
p, preds = probs[0, o_index], torch.Tensor(o_index).int()
を書き換える。
'''''

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

def read_knowlege(count=150, kind=None, arch="gpt2-xl", all_flow_data=[]):
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
    for i, data in enumerate(all_flow_data[:data_len]):
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
        low_score=avg_ls.avg(), result=result, result_std=result_std, size=avg_fe.size()
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
        # cmap={None: "Purples", "mlp": "Greens", "attn": "Reds"}[kind],
        cmap="Reds",
        vmin=low_score,
        vmax=high_score,
    )
    if title:
        ax.set_title(title)
    ax.invert_yaxis()
    ax.set_yticks([0.5 + i for i in range(len(differences))])
    ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
    ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
    ax.set_yticklabels(labels)
    if kind is None:
        # ax.set_xlabel(f"single patched layer within {archname}")
        ax.set_xlabel(f"layer number")
    else:
        # ax.set_xlabel(f"center of interval of 10 patched {kind} layers")
        ax.set_xlabel(f"layer number")
    cb = plt.colorbar(h)
    # カラーバーを作成し、ScalarFormatter を設定
    # formatter = ScalarFormatter(useOffset=False)  # オフセットを使用しない
    # formatter.set_powerlimits((0, 0))  # すべての数値を指数形式で表示
    # cb = plt.colorbar(h, format=formatter)
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
    if kind is None:
        data_path = f"data/all_flow_data/{arch}.csv"
    else:
        data_path = f"data/all_flow_data/{arch}_{kind}.csv"
    all_flow_data = read_all_flow_data(data_path)
    d = read_knowlege(the_count, kind, arch, all_flow_data)
    count = d["size"]
    # what = {
    #     None: "Indirect Effect of $h_i^{(l)}$",
    #     "mlp": "Indirect Effect of MLP",
    #     "attn": "Indirect Effect of Attn",
    # }[kind]
    what = {
        None: "hidden neuron",
        "mlp": "MLP module",
        "attn": "Attn module",
    }[kind]
    # title = f"Avg {what} over {count} prompts"
    title = f"{what}"
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
        savepdf=f"results/{arch}/causal_trace/summary_pdfs/rollup{kindcode}.png",
    )

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
    # [
    #     (None, "single hidden vector"),
    #     ("mlp", "run of 10 MLP lookups"),
    #     ("attn", "run of 10 Attn modules"),
    # ]
    [
        (None, "layer number"),
        ("mlp", "layer number"),
        ("attn", "layer number"),
    ]
):
    print(f"Reading {kind}")
    if kind is None:
        data_path = f"data/all_flow_data/{arch}.csv"
    else:
        data_path = f"data/all_flow_data/{arch}_{kind}.csv"
    all_flow_data = read_all_flow_data(data_path)
    d = read_knowlege(the_count, kind, arch, all_flow_data)
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
os.makedirs(os.path.dirname(savepdf), exist_ok=True)
plt.savefig(savepdf)