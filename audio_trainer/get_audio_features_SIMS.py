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
from easydict import EasyDict as edict

class getFeatures():
    def __init__(self,args, input_dir, output_dir):
        self.data_dir = input_dir
        self.output_dir = output_dir
        self.padding_mode = 'zeros'
        self.padding_location = 'back'
        model_id = r"middleModel3.pth"
        self.model = MyModel(args)
        self.model.load_state_dict(torch.load(model_id))
        self.model.to("cuda")

    def time_down(self,sequence, window_size=50):
        num_windows = sequence.shape[0] // window_size
        averaged_time_series = np.zeros((num_windows, sequence.shape[1]))
        for i in range(num_windows):
            start_idx = i * window_size
            end_idx = (i + 1) * window_size
            window_data = sequence[start_idx:end_idx, :]
            averaged_time_series[i, :] = np.mean(window_data, axis=0)

        return averaged_time_series

    def dim_reduce (self, dataMat, n_components=999999):
        # 实例化PCA对象
        pca = PCA(n_components)
        # 拟合数据
        pca.fit(dataMat)
        # 进行降维
        skreduced = pca.transform(dataMat)
        return skreduced

    def results(self):
        audio_id = []
        audio_clip_id = []
        audio_id_clip_id = []
        features_A = []
        file_list = os.listdir(self.data_dir)
        num_fold = len(file_list)
        for i, video_dir in enumerate(file_list):
            print(f"{i}/{num_fold}")
            # 枚举单个音频
            single_audio_files = os.listdir(os.path.join(self.data_dir, video_dir))
            for single_audio in tqdm(single_audio_files):
                audio_id.append(video_dir)
                audio_clip_id.append(single_audio.split('.')[0])
                audio_id_clip_id.append(str(video_dir) + "_" + str(single_audio.split('.')[0]))
                # print(single_audio)  ### 0001.wav
                audio_path = os.path.join(self.data_dir, video_dir, single_audio)
                # 预处理
                mel = log_mel_spectrogram(audio_path, 128, padding=N_SAMPLES)
                mel_segment = pad_or_trim(mel, N_FRAMES).to("cuda").to(torch.float32).unsqueeze(0)
                # 进模型
                embedding_A = self.model(mel_segment).squeeze().cpu().detach().numpy()
                # embedding_A = self.model.transcribe(audio_path, only_features=True).squeeze().cpu().numpy()

                # embedding_A = self.dim_reduce(embedding_A, n_components=128)
                # embedding_A = self.time_down(embedding_A)
                features_A.append(embedding_A)
        # 保存
        save_path = os.path.join(self.output_dir, 'audioFeature.npz')
        np.savez(save_path, audio_id=audio_id, audio_clip_id=audio_clip_id, audio_id_clip_id=audio_id_clip_id,
                 feature_A=features_A)

        print('Features are saved in %s!' % save_path)





if __name__ == "__main__":
    input_dir= "D:\Search\MSA\SIMS\AudioFeature\\audioRaw"
    output_dir= 'D:\Search\MSA\SIMS\AudioFeature'
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