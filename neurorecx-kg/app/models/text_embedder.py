from transformers import AutoTokenizer,AutoModel

import torch

class TextEmbedder:
        def __init__(self):
                self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                self.model = AutoModel.from_pretrained("distilbert-base-uncased")

        def embed(self,text):
                inputs = self.tokenizer(text,return_tensors="pt",truncation=True,padding=True)
                with torch.no_grad():
                        outputs = self.model(**inputs)
                return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


















