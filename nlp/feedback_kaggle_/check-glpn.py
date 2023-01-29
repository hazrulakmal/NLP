from transformers import AutoModel
from optimum.bettertransformer import BetterTransformer
import torch 

model_id = "hf-internal-testing/tiny-random-DPTModel"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_id).to(device)
model = BetterTransformer.transform(model)
print(model)