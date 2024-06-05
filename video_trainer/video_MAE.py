import av
import torch
from transformers import AutoImageProcessor, VideoMAEModel, AutoProcessor
import os
import numpy as np
from glob import glob
from tqdm import tqdm
import torch.nn as nn
import pickle
from torch.utils.data import Dataset
from torch import optim
from torch import nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from easydict import EasyDict as edict
import logging
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import pandas as pd
from torch.nn.parameter import Parameter
logger = logging.getLogger('video')

def _set_logger(log_dir, verbose_level=1):

    # base logger
    log_file_path = Path(log_dir) /"video.log"
    # log_file_path = log_dir
    logger = logging.getLogger('video')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    ch = logging.StreamHandler()
    ch.setLevel(stream_level[verbose_level])
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger

class MyModel(nn.Module):
    def __init__(self,args):
        super(MyModel, self).__init__()
        self.seq_weight = args["seq_weight"]
        self.args = args
        self.model_id = "D:\Search\LLM\\videomae-large"
        # self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = VideoMAEModel.from_pretrained(self.model_id)
        for name, param in self.model.named_parameters():
            # print(name, param.size())
            param.requires_grad = False
        self.LSTM = nn.LSTM(input_size=1024, hidden_size=1024, num_layers=1, batch_first=True)
        self.seq1 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1024),
            nn.Sigmoid(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1024, out_features=1024)
        )
        self.seq2 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),

            nn.Dropout(p=0.1),
            nn.Linear(in_features=512, out_features=128)
        )

        self.seq3 = nn.Sequential(
            nn.Linear(in_features=128, out_features=16),
            nn.Sigmoid(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=16, out_features=1)
        )

        if os.path.exists(self.seq_weight):
            state_dict = torch.load(self.seq_weight)
            self.LSTM.load_state_dict(state_dict['LSTM'])
            self.seq1.load_state_dict(state_dict['seq1'])
            self.seq2.load_state_dict(state_dict['seq2'])
            self.seq3.load_state_dict(state_dict['seq3'])
            logger.info("Loaded seq weights")
        else:
            logger.info("No seq weights")

        self.sig = nn.Sigmoid()
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)
        # self.Th = nn.functional.tanh()


    def forward(self, inputs):
        video_features = self.model(inputs).last_hidden_state
        _, final_states = self.LSTM(video_features)
        h = self.seq1(final_states[0].squeeze(0))
        output = self.seq2(h)
        h = self.seq3(output)
        output = torch.sigmoid(h)
        output = output * self.output_range + self.output_shift

        return output

def my_collate(batch):
    videos = []
    labels = []
    processor = AutoProcessor.from_pretrained("D:\Search\LLM\\xclip-base-patch32")
    # processor = AutoImageProcessor.from_pretrained("D:\Search\LLM\\videomae-large")
    def read_video_pyav(container, indices):

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

    def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
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

    for video_path, label in batch:
        container = av.open(video_path)
        # sample 8 frames
        indices = sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=container.streams.video[0].frames)

        video = read_video_pyav(container, indices)


        # video_array = np.array(video)
        # video_min = video_array.min()
        # video_max = video_array.max()
        # video_normalized = (video_array - video_min) / (video_max - video_min)
        # inputs = processor(list(video_normalized), return_tensors="pt")

        inputs = processor(videos=list(video), return_tensors="pt")

        videos.append(inputs["pixel_values"].squeeze(0))
        labels.append(label)


    return torch.stack(videos), torch.tensor(labels)

class MyDataset(Dataset):
    def __init__(self, args, mode):
        super(MyDataset, self).__init__()
        self.args = args
        with open(args['file_path'], 'rb') as f:
            data = pickle.load(f)
            self.data = data[mode]
            # print(f"Mode: {mode}, Number of samples: {len(self.data['audio'])}")

    def __len__(self):
        return len(self.data['audio'])

    def __getitem__(self, item):
        video = self.data['video'][item]
        label = self.data['label'][item]
        return video, label

class trainer():
    def __init__(self, args, model):
        self.args = args
        self.model = model.to(args.device)
        self.criterion = nn.L1Loss()
    def __multiclass_acc(self, y_pred, y_true):
        return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))
    def metrics(self, y_pred, y_true):
        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()
        test_preds = np.clip(test_preds, a_min=-1., a_max=1.)
        test_truth = np.clip(test_truth, a_min=-1., a_max=1.)

        # two classes{[-1.0, 0.0], (0.0, 1.0]}
        ms_2 = [-1.01, 0.0, 1.01]
        test_preds_a2 = test_preds.copy()
        test_truth_a2 = test_truth.copy()
        for i in range(2):
            test_preds_a2[np.logical_and(test_preds > ms_2[i], test_preds <= ms_2[i+1])] = i
        for i in range(2):
            test_truth_a2[np.logical_and(test_truth > ms_2[i], test_truth <= ms_2[i+1])] = i

        # three classes{[-1.0, -0.1], (-0.1, 0.1], (0.1, 1.0]}
        ms_3 = [-1.01, -0.1, 0.1, 1.01]
        test_preds_a3 = test_preds.copy()
        test_truth_a3 = test_truth.copy()
        for i in range(3):
            test_preds_a3[np.logical_and(test_preds > ms_3[i], test_preds <= ms_3[i+1])] = i
        for i in range(3):
            test_truth_a3[np.logical_and(test_truth > ms_3[i], test_truth <= ms_3[i+1])] = i

        # five classes{[-1.0, -0.7], (-0.7, -0.1], (-0.1, 0.1], (0.1, 0.7], (0.7, 1.0]}
        ms_5 = [-1.01, -0.7, -0.1, 0.1, 0.7, 1.01]
        test_preds_a5 = test_preds.copy()
        test_truth_a5 = test_truth.copy()
        for i in range(5):
            test_preds_a5[np.logical_and(test_preds > ms_5[i], test_preds <= ms_5[i+1])] = i
        for i in range(5):
            test_truth_a5[np.logical_and(test_truth > ms_5[i], test_truth <= ms_5[i+1])] = i

        mae = np.mean(np.absolute(test_preds - test_truth)).astype(np.float64)   # Average L1 distance between preds and truths
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a2 = self.__multiclass_acc(test_preds_a2, test_truth_a2)
        mult_a3 = self.__multiclass_acc(test_preds_a3, test_truth_a3)
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        f_score = f1_score(test_truth_a2, test_preds_a2, average='weighted')

        eval_results = {
            "Mult_acc_2": round(mult_a2, 4),
            "Mult_acc_3": round(mult_a3, 4),
            "Mult_acc_5": round(mult_a5, 4),
            "F1_score": round(f_score, 4),
            "MAE": round(mae, 4),
            "Corr": round(corr, 4), # Correlation Coefficient
        }
        return eval_results

    def do_train(self, train_dataloader, val_dataloader):
        parameters_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(parameters_to_optimize, lr=self.args["learning_rate"])
        train_losses = []
        train_Mult_acc_2 = []
        train_MAE = []
        eval_losses = []
        eval_Mult_acc_2 = []
        eval_MAE = []
        save_time = 3

        for epoch in range(self.args.num_epochs):
            y_pred, y_true = [], []
            train_loss = 1
            self.model.train()

            for batch in tqdm(train_dataloader):
                optimizer.zero_grad()
                video, label = batch
                video = video.to(self.args.device)
                # audio = np.array(list(audio))
                label = label.to(self.args.device)
                # inputs = self.spec_augment(audio)
                outputs = self.model(video).squeeze()
                loss = self.criterion(outputs, label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                y_pred.append(outputs.cpu())
                y_true.append(label.cpu())

                # 可能这个数据集的音频数据好
                # 能不能自适应地去选择模态
                # 针对不同的数据集选择不同的fusion方式
                # 试试其他数据集
                # 合适的微调机制——更好的特征

            if epoch % 10 == 9:
                save_time += 1

            # torch.save(self.model.state_dict(), self.args["save_path"])
            # logging.info(f"save model in {self.args['save_path']}")
            path = "MOSI_mae_" + str(save_time) + ".pth"

            state_dict = {
                'LSTM': self.model.LSTM.state_dict(),
                'seq1': self.model.seq1.state_dict(),
                'seq2': self.model.seq2.state_dict(),
                'seq3': self.model.seq3.state_dict()
            }
            torch.save(state_dict, path)
            logger.info(f"save model in {path}")

            train_loss /= len(train_dataloader)
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)

            logger.info(
                f"Epoch [{epoch + 1}/{self.args.num_epochs}], Loss: {train_loss:.4f}, trian_acc: {train_results['Mult_acc_2']:.4f}, train_mae: {train_results['MAE']:.4f}")

            test_results = self.do_test(val_dataloader)


            train_losses.append(train_loss)
            train_Mult_acc_2.append(train_results["Mult_acc_2"])
            train_MAE.append(train_results["MAE"])
            eval_losses.append(test_results["test_loss"])
            eval_Mult_acc_2.append(test_results["test_result"]["Mult_acc_2"])
            eval_MAE.append(test_results["test_result"]["MAE"])

        results = {
            "train_losses": train_losses,
            "eval_losses": eval_losses,
            "train_Mult_acc_2": train_Mult_acc_2,
            "eval_Mult_acc_2": eval_Mult_acc_2,
            "train_MAE": train_MAE,
            "eval_MAE": eval_MAE

        }
        return results


    def do_test(self, dataloader):
        test_loss = 0
        y_pred, y_true = [], []
        self.model.eval()
        for batch in tqdm(dataloader):
            audio, label = batch
            audio = audio.to(self.args.device)
            # audio = np.array(list(audio))
            label = label.to(self.args.device)
            # inputs = self.spec_augment(audio)
            outputs = self.model(audio).squeeze()
            loss = self.criterion(outputs, label)

            test_loss += loss.item()
            y_pred.append(outputs.cpu())
            y_true.append(label.cpu())


        test_loss /= len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        test_result = self.metrics(pred, true)
        logger.info(f"Test_Loss: {test_loss:.4f}, test_acc: {test_result['Mult_acc_2']:.4f}, test_mae: {test_result['MAE']:.4f}")
        results = {
            "test_loss": test_loss,
            "test_result": test_result
        }
        return results



if __name__ == "__main__":
    args = {
        'learning_rate': 1e-5,
        "device": 'cuda',
        "num_epochs": 10,
        "batch_size": 16,
        "file_path": 'D:\Search\FT\\MOSI.pkl',

        "save_path": "Model",
        "seq_weight": "MOSI_mae_3.pth",

    }
    _set_logger(log_dir='D:\Search\FT\\video_trainer')
    args = edict(args)
    # file_path = 'D:\Search\LLM\FT\\SIMS.pkl'
    dataset_train = MyDataset(args, 'train')
    dataset_val = MyDataset(args, 'valid')
    dataset_test = MyDataset(args, 'test')

    train_dataloader = DataLoader(dataset_train, batch_size=args["batch_size"], shuffle=True, collate_fn=my_collate)
    valid_dataloader = DataLoader(dataset_val, batch_size=args["batch_size"], shuffle=True, collate_fn=my_collate)
    test_dataloader = DataLoader(dataset_test, batch_size=args["batch_size"], shuffle=True, collate_fn=my_collate)

    model = MyModel(args).to(args["device"])
    # model.load_state_dict(torch.load(args["load_path"]))
    trainer = trainer(args, model)
    logger.info('Start training...')

    # train
    epoch_results = trainer.do_train(train_dataloader, valid_dataloader)

    logger.info('End training...')
    # test
    test_results = trainer.do_test(test_dataloader)
    criterions = list(test_results["test_result"].keys())
    # save result to csv
    csv_file = Path(f"D:\Search\FT\\video.csv")
    if csv_file.is_file():
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame(columns=criterions)
    res = []
    # save results
    for c in criterions:
        # values = [r[c] for r in test_results["test_result"]]
        values = test_results["test_result"][c]
        # mean = round(np.mean(values) * 100, 2)
        # std = round(np.std(values) * 100, 2)
        # res.append((mean, std))
        res.append(values)
    df.loc[len(df)] = res
    df.to_csv(csv_file, index=None)
    logger.info(f"Results saved to {csv_file}.")

    # show_pic
    plt.plot(epoch_results['train_losses'], label="train:")
    plt.plot(epoch_results['eval_losses'], label="eval:")
    plt.title(f'{args["save_path"]}_losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(epoch_results['train_Mult_acc_2'], label="train:")
    plt.plot(epoch_results['eval_Mult_acc_2'], label="eval:")
    plt.title(f'{args["save_path"]}_acc')
    plt.xlabel('Epoch')
    plt.ylabel('acc')
    plt.legend()
    plt.show()

    plt.plot(epoch_results['train_MAE'], label="train:")
    plt.plot(epoch_results['eval_MAE'], label="eval:")
    plt.title(f'{args["save_path"]}_MAE')
    plt.xlabel('Epoch')
    plt.ylabel('mae')
    plt.legend()
    plt.show()
    del model
    torch.cuda.empty_cache()

