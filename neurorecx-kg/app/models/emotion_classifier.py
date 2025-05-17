from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class EmotionClassifier:
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
        self.tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
        self.labels = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", 
                       "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
                       "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
                       "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]

    def classify(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=1).squeeze()
        top = torch.topk(probs, 3)
        return [(self.labels[i], float(probs[i])) for i in top.indices]
