import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# モデルをロードする
tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_3b")
model = AutoModelForCausalLM.from_pretrained("openlm-research/open_llama_3b")
model.to("cuda")

class CausalMetricsModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.device = next(self.model.parameters()).device
    def forward(self, input_ids, noise=False, intervention=None):
        with torch.no_grad():
            # モデルの出力と隠れ状態を取得
            outputs = self.model(input_ids, output_hidden_states=True)
            hidden_states = list(outputs.hidden_states)
        if noise:
            # Step 2: 入力の埋め込みベクトルにノイズを加える
            input_noise = torch.randn(hidden_states[0].size(), device=self.device)
            hidden_states[0] += input_noise
        if intervention is not None:
            # Step 4: 指定された層とトークンの隠れ状態をクリーンな状態に戻す
            layer, token = intervention
            modified_hidden_states = [hidden_state.clone() for hidden_state in hidden_states]
            modified_hidden_states[layer][:, token] = outputs.hidden_states[layer][:, token]
            hidden_states = modified_hidden_states
        # Step 5: 修正された隠れ状態をモデルに入力し、新しい予測を得る
        past_key_values = tuple(hidden_state.unsqueeze(0) for hidden_state in hidden_states[1:])
        outputs = self.model(inputs_embeds=hidden_states[0], past_key_values=past_key_values)
        return outputs.logits

def causal_metrics(model, inputs, layer_range=None):
    if layer_range is None:
        # 全ての層を対象とする
        layer_range = model.model.config.num_hidden_layers
    causal_metrics = []
    # ノイズを加えた状態での予測をベースラインとする
    baseline_logits = model(inputs, noise=True)
    for layer in layer_range:
        layer_metrics = []
        for token in range(inputs.size(1)):
            # 各層とトークンについて介入を行う
            intervention_logits = model(inputs, intervention=(layer, token))
            # ベースラインとの予測の差を計算
            causal_influence = (intervention_logits - baseline_logits).norm(dim=-1).mean().item()
            layer_metrics.append(causal_influence)
        causal_metrics.append(layer_metrics)
    return causal_metrics

# # ファイル名
# output_file = "model_parameters.txt"
# # 層の確認
# num_dict = {}
# with open(output_file, 'w') as f:
#     for name, param in model.named_parameters():
#         num_dict[name] = param.numel()
#         f.write(name + '\n')
# print("出力が {} に書き込まれました。".format(output_file))

import pdb;pdb.set_trace()
# # 実行例
# input_text = "The quick brown fox"
# inputs = tokenizer.encode_plus(input_text, return_tensors="pt").to("cuda")
# causal_model = CausalMetricsModel(model)
# # 全ての層について因果的な影響を計算
# causal_influences = causal_metrics(causal_model, inputs["input_ids"])
# # 結果を表示
# for layer, influences in enumerate(causal_influences):
#     print(f"Layer {layer}:")
#     for token, influence in enumerate(influences):
#         print(f"Token {token}: {influence:.4f}")
#     print()