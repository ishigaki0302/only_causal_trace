import torch, numpy
from collections import defaultdict
import datetime
import torch
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
dt_now = datetime.datetime.now()
torch.set_grad_enabled(False)

# model_name = "gpt2-xl"
# model_name = "EleutherAI/gpt-j-6B"
model_name = "Ryoma0302/gpt_0.76B_global_step13000"
# model_name = "EleutherAI/gpt-neox-20b"
# model_name = "rinna/japanese-gpt-neox-3.6b-instruction-sft"
# model_name = "rinna/japanese-gpt-neox-3.6b"
# model_name = "cyberagent/open-calm-7b"
# model_name = "meta-llama/Meta-Llama-3-8B"

mt = ModelAndTokenizer(
    model_name,
    torch_dtype=(torch.float16 if "20b" in model_name else None),
)

# 既知の事実のデータセットを読み込む
knowns = KnownsDataset(DATA_DIR)
# データセット内の各事実の主語を表示
print([k["subject"] for k in knowns])
# 言語モデルのエンベディングの標準偏差を計算し、ノイズレベルを設定
noise_level = 3 * collect_embedding_std(mt, [k["subject"] for k in knowns])
print(f"Using noise level {noise_level}")

"""
パッチ
    モデルの特定の部分を一時的に変更する操作
    1. トークンの特定の範囲にノイズを追加
    2. トレース中に特定のレイヤーの隠れ状態の出力を修正
"""
def trace_with_patch(
    model,  # モデル
    inp,  # 入力セット
    states_to_patch,  # 復元するトークンインデックスとレイヤー名の3重タプルのリスト
    answers_t,  # 収集する回答確率
    tokens_to_mix,  # 破損させるトークンの範囲（開始、終了）
    noise=0.1,  # 追加するノイズのレベル
    trace_layers=None,  # 返すトレースされた出力のリスト
):
    # 再現性のために、疑似ランダムノイズを使用
    prng = numpy.random.RandomState(1)
    # パッチを適用するレイヤーとトークンのマッピングを作成
    patch_spec = defaultdict(list)  # デフォルト値がリストのディクショナリを作成
    for t, l in states_to_patch:
        patch_spec[l].append(t)  # レイヤー名をキー、トークンインデックスをリストの要素として追加
    # 埋め込み層の名前を取得
    embed_layername = layername(model, 0, "embed")  # モデルの最初の埋め込み層の名前を取得
    def untuple(x):
        return x[0] if isinstance(x, tuple) else x  # タプルの場合は最初の要素を返し、そうでない場合はそのまま返す
    # モデルパッチングルールを定義する
    def patch_rep(x, layer):
        if layer == embed_layername:
            # 指定した範囲(subject)のトークン埋め込みを破損させる処理
            if tokens_to_mix is not None:
                b, e = tokens_to_mix  # 破損させるトークンの範囲を取得
                x[1:, b:e] += noise * torch.from_numpy(
                    prng.randn(x.shape[0] - 1, e - b, x.shape[2])  # 正規分布から乱数を生成し、テンソルに変換
                ).to(x.device)
            return x
        if layer not in patch_spec:
            return x  # パッチを適用しない場合はそのまま返す
        # 引数のレイヤーがpatch_specに含まれている場合、選択されたトークン位置の隠れ状態をクリーンな隠れ状態に復元する
        h = untuple(x)  # タプルの場合は最初の要素を取り出す
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]  # 指定されたトークン位置の隠れ状態を、バッチの最初の要素の隠れ状態で置き換える
        return x
    # パッチングルールが定義されたら、パッチされたモデルを推論中に実行する
    additional_layers = [] if trace_layers is None else trace_layers  # トレースするレイヤーがない場合は空のリストを使用
    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layername] + list(patch_spec.keys()) + additional_layers,  # パッチを適用するレイヤーとトレースするレイヤーを指定
        edit_output=patch_rep,  # パッチングルールを適用する関数を指定
    ) as td:
        outputs_exp = model(**inp)  # パッチを適用したモデルで推論を実行
    # 指定されたanswers_tトークンの予測確率を計算し、報告する
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]
    # すべてのレイヤーをトレースする場合、すべての活性化を収集して返す
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2  # 各レイヤーの出力を収集し、新しい次元に沿ってスタックする
        )
        return probs, all_traced  # answers_tトークンの予測確率とトレースされた活性化を返す
    return probs  # answers_tトークンの予測確率を返す

def calculate_hidden_flow(
    mt, prompt, subject, o="Seattle", samples=10, noise=0.1, window=10, kind=None
):
    """
    ネットワーク全体の各トークン/レイヤーの組み合わせに対して因果推論を実行し、
    結果を数値で要約した辞書を返します。
    """
    # 入力のセットを作成
    inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1))
    # モデルからの予測結果を取得
    with torch.no_grad():
        answer_t, base_score = [d[0] for d in predict_from_input(mt.model, mt.tokenizer, inp, o)]
    # 予測されたトークンから回答をデコード
    [answer] = decode_tokens(mt.tokenizer, [answer_t])
    print(subject)
    # 主語のトークン範囲を見つける
    e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
    # パッチングされたトレースを実行し、最低スコアを計算
    low_score = trace_with_patch(
        mt.model, inp, [], answer_t, e_range, noise=noise
    ).item()
    # kindに基づいて重要なstateをトレース
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

# 隠れ状態用
def trace_important_states(model, num_layers, inp, e_range, answer_t, noise=0.1):
    # 入力のトークン数を取得
    ntoks = inp["input_ids"].shape[1]
    table = []
    for tnum in range(ntoks):
        row = []
        for layer in range(0, num_layers):
            # 指定されたトークンとレイヤーについてパッチを適用してトレースを実行
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

# MLPとAttn用
def trace_important_window(
    model, num_layers, inp, e_range, answer_t, kind, window=10, noise=0.1
):
    # 入力のトークン数を取得
    ntoks = inp["input_ids"].shape[1]
    table = []
    for tnum in range(ntoks):
        row = []
        for layer in range(0, num_layers):
            # 指定されたウィンドウ内のレイヤーについてパッチを適用するレイヤーのリストを作成
            layerlist = [
                (tnum, layername(model, L, kind))
                for L in range(
                    max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
                )
            ]
            # 指定されたトークンとウィンドウ内のレイヤーについてパッチを適用してトレースを実行
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
    # 隠れ状態のフローを計算
    result = calculate_hidden_flow(
        mt, prompt, subject, o, samples=samples, noise=noise, window=window, kind=kind
    )
    # 結果をヒートマップとしてプロットし、PDFに保存
    plot_trace_heatmap(result, savepdf, modelname=modelname)
    return result

def plot_all_flow(mt, prompt, subject=None, o="Seattle", noise=0.1, modelname=None, savepdf=None):
    three_result = []
    for kind in [None, "mlp", "attn"]:
        if kind is None:
            savepdf=f"hidden_{savepdf}"
        else:
            savepdf=f"{kind}_{savepdf}"
        # 指定されたkindについて隠れ状態のフローをプロットし、結果を保存
        result = plot_hidden_flow(
            mt, prompt, subject, o, modelname=modelname, noise=noise, kind=kind, savepdf=f'result_pdf/{dt_now}/{savepdf}'
        )
        three_result.append(result)
    return three_result

prompt = "Windows Media Player is developed by"
# new_prompt = "Who developed Windows Media Player?"
subject = "Windows Media Player"
attribute = "Microsoft"
# prompt = "大谷翔平が所属するチームはどこですか？"
# prompt = "Shohei Ohtani is a member of the"
# prompt = "Where does Shohei Ohtani belong?"
# new_prompt = "Who developed Windows Media Player?"
# subject = "大谷翔平"
# subject = "Shohei Ohtani"
# attribute = "エンゼルス"
# attribute = "Angels"
plot_all_flow(mt, prompt=prompt, subject=subject, o=attribute, noise=noise_level, savepdf=f'result_pdf/0')
# plot_all_flow(mt, prompt=new_prompt, subject=subject, o=attribute, noise=noise_level, savepdf=f'result_pdf/1')