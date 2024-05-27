import os
import numpy as np
from tqdm import tqdm
from video_XCLIP import MyModel
import torch
from transformers import AutoProcessor
import av

from easydict import EasyDict as edict

class getFeatures():
    def __init__(self,processor, model, input_dir, output_dir):
        self.processor = processor
        self.model = model
        self.data_dir = input_dir
        self.output_dir = output_dir


    def read_video_pyav(self, container, indices):
        '''
        Decode the video with PyAV decoder.
        Args:
            container (`av.container.input.InputContainer`): PyAV container.
            indices (`List[int]`): List of frame indices to decode.
        Returns:
            result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        '''
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])

    def sample_frame_indices(self, clip_len, frame_sample_rate, seg_len):
        '''
        Sample a given number of frame indices from the video.
        Args:
            clip_len (`int`): Total number of frames to sample.
            frame_sample_rate (`int`): Sample every n-th frame.
            seg_len (`int`): Maximum allowed index of sample's last frame.
        Returns:
            indices (`List[int]`): List of sampled frame indices
        '''
        converted_len = int(clip_len * frame_sample_rate)
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        return indices

    def results(self):
        video_id = []
        video_clip_id = []
        features_V = []
        file_list = os.listdir(self.data_dir)
        num_fold = len(file_list)
        for i, video_dir in enumerate(file_list):
            print(f"{i}/{num_fold}")
            # 枚举单个视频
            single_video_files = os.listdir(os.path.join(self.data_dir, video_dir))
            for single_video in tqdm(single_video_files):
                video_id.append(video_dir)
                video_clip_id.append(single_video.split('.')[0])
                # print(single_video)  ### 0001.wav
                video_path = os.path.join(self.data_dir, video_dir, single_video)
                # 预处理
                container = av.open(video_path)
                # sample 8 frames
                indices = self.sample_frame_indices(clip_len=8, frame_sample_rate=1,
                                               seg_len=container.streams.video[0].frames)
                # 每隔一秒钟采样一次，一共采样8秒钟
                video = self.read_video_pyav(container, indices)
                inputs = self.processor(videos=list(video), return_tensors="pt")

                # 进模型
                embedding_V = self.model(inputs["pixel_values"].to("cuda")).squeeze().cpu().detach().numpy()
                features_V.append(embedding_V)
        # 保存
        save_path = os.path.join(self.output_dir, 'videoFeature.npz')
        np.savez(save_path, video_id=video_id, video_clip_id=video_clip_id,
                 feature_V=features_V)

        print('Features are saved in %s!' % save_path)





if __name__ == "__main__":
    input_dir= "D:\Search\MSA\SIMS\SIMS_raw\Raw"
    output_dir= 'D:\Search\MSA\SIMS\VideoFeature'
    args={
        'learning_rate': 1e-5,
        "device": 'cuda',
        "num_epochs": 20,
        "batch_size": 64,
        "file_path": 'D:\Search\FT\\SIMS_unsplit.pkl',
        "save_path": "Model",
        "seq_weight": "D:\Search\FT\\video_trainer\Model2.pth",
    }
    processor = AutoProcessor.from_pretrained("D:\Search\LLM\\xclip-base-patch32")
    model = MyModel(args).to(args["device"])
    args = edict(args)
    os.makedirs(output_dir, exist_ok=True)
    gf = getFeatures(processor, model, input_dir,output_dir)
    gf.results()