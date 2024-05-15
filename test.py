import wandb
wandb.init(project="knowledge-editing", name="Ryoma0302/gpt_0.76B_global_step20000_japanese", entity="dsml-kernel24")
from PIL import Image
image = Image.open("result_pdf/2024-05-15 22:13:44.384687/hidden_result_pdf/0.png")
wandb.log({"graph": wandb.Image(image)})