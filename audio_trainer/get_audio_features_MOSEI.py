import os
import numpy as np
from tqdm import tqdm
import whisper
from sklearn.decomposition import PCA
from audio import MyModel
import torch
from whisper.audio import (
    FRAMES_PER_SECOND,
    HOP_LENGTH,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    log_mel_spectrogram,
    pad_or_trim,
)
import pickle
from easydict import EasyDict as edict

class getFeatures():
    def __init__(self,args, input_dir, output_dir):
        self.data_dir = input_dir
        self.output_dir = output_dir
        self.padding_mode = 'zeros'
        self.padding_location = 'back'
        model_id = r"AudioModel.pth"
        self.model = MyModel(args)
        self.model.load_state_dict(torch.load(model_id))
        self.model.to("cuda")


    def results(self):
        features_A = []
        with open(self.data_dir, 'rb') as file:
            data = pickle.load(file)
            audios = data['audio']
            lens = len(audios)
            for i, audio in enumerate(audios):
                print(f"{i}/{lens}")
                mel = log_mel_spectrogram(audio, 128, padding=N_SAMPLES)
                mel_segment = pad_or_trim(mel, N_FRAMES).to("cuda").to(torch.float32).unsqueeze(0)
                # 进模型
                embedding_A = self.model(mel_segment).squeeze().cpu().detach().numpy()
                # embedding_A = self.model.transcribe(audio_path, only_features=True).squeeze().cpu().numpy()

                features_A.append(embedding_A)
        # 保存
        save_path = os.path.join(self.output_dir, 'audioFeature.npz')
        np.savez(save_path, audio=audios, feature_A=features_A)

        print('Features are saved in %s!' % save_path)





if __name__ == "__main__":
    input_dir= "D:\Search\FT\\MOSEI.pkl"
    output_dir= 'D:\Search\MSA\MOSEI\AudioFeature'
    args={
        "n_mels": 128,
        "n_audio_ctx": 1500,
        "n_audio_state": 1280,
        "n_audio_head": 20,
        "n_audio_layer": 32,
    }
    args = edict(args)
    os.makedirs(output_dir, exist_ok=True)
    gf = getFeatures(args, input_dir,output_dir)
    gf.results()