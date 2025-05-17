import torchaudio
import torch
import soundfile as sf

class AudioEmbedder:
    def __init__(self):
        self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        self.model.eval()

    def embed(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:  # stereo → mono
            waveform = waveform.mean(dim=0).unsqueeze(0)
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        with torch.no_grad():
            embedding = self.model(waveform)
        return embedding.mean(dim=0).numpy()

#all three embedders return numpy vectors