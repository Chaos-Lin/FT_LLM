import pickle
from torch.utils.data import Dataset
from torch import optim
from torch import nn as nn
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
import logging
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import pandas as pd
from torch.nn.parameter import Parameter

from whisper.audio import (
    N_FRAMES,
    N_SAMPLES,
    log_mel_spectrogram,
    pad_or_trim,
)
from whisper.model import AudioEncoder

logger = logging.getLogger('LLM')

def _set_logger(log_dir, verbose_level=1):

    # base logger
    log_file_path = Path(log_dir) /"audio.log"
    # log_file_path = log_dir
    logger = logging.getLogger('LLM')
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


# with open(r'D:\Search\LLM\SIMS.pkl', 'rb') as f:
#     data = pickle.load(f)
#     print(data['train'].keys())
#     print(data['train']['audio'].shape)

class MyModel(nn.Module):
    def __init__(self,dims):
        super(MyModel, self).__init__()
        self.dims = dims
        # model_id = r"D:\Search\LLM\whisper-main\\large-v3.pt"
        # self.model = whisper.load_model(model_id)
        # torch.save(self.model.encoder.state_dict(), 'encoder_weights.pth')
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        # self.encoder.load_state_dict(torch.load("encoder_weights.pth"))
        for name, param in self.encoder.named_parameters():
            # print(name, param.size())
            param.requires_grad = False
        self.lstm = nn.LSTM(input_size=1280, hidden_size=1280, num_layers=1, batch_first=True)
        self.seq1 = nn.Sequential(
            nn.Linear(in_features=1280, out_features=1280),
            nn.Sigmoid(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1280, out_features=1280)
        )

        self.seq2 = nn.Sequential(
            nn.Linear(in_features=1280, out_features=512),

            nn.Dropout(p=0.1),
            nn.Linear(in_features=512, out_features=128)
        )

        self.seq3 = nn.Sequential(
            nn.Linear(in_features=128, out_features=16),
            nn.Sigmoid(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=16, out_features=1)
        )

        self.sig = nn.Sigmoid()
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)
        # self.Th = nn.functional.tanh()


    def forward(self, x):
        h = self.encoder(x)
        _, final_states = self.lstm(h)
        h = self.seq1(final_states[0].squeeze(0))
        output = self.seq2(h)
        h = self.seq3(output)
        output = torch.sigmoid(h)
        output = output * self.output_range + self.output_shift

        return output


def my_collate(batch):
    mel_segments = []
    labels = []
    for audio, label in batch:
        mel = log_mel_spectrogram(audio, 128, padding=N_SAMPLES)
        mel_segment = pad_or_trim(mel, N_FRAMES).to("cuda").to(torch.float32)
        mel_segments.append(mel_segment)
        labels.append(label)

    return torch.stack(mel_segments), torch.tensor(labels)

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
        audio = self.data['audio'][item]
        label = self.data['label'][item]
        return audio, label


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
        save_time = 0

        for epoch in range(self.args.num_epochs):
            y_pred, y_true = [], []
            train_loss = 0
            self.model.train()

            for batch in tqdm(train_dataloader):
                optimizer.zero_grad()
                audio, label = batch
                audio = audio.to(self.args.device)
                # audio = np.array(list(audio))
                label = label.to(self.args.device)
                # inputs = self.spec_augment(audio)
                outputs = self.model(audio).squeeze()
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

            if epoch % 1 == 0:
                save_time += 1

            # torch.save(self.model.state_dict(), self.args["save_path"])
            # logging.info(f"save model in {self.args['save_path']}")
            path = "MOSEIModel" + str(save_time) + ".pth"
            torch.save(self.model.state_dict(), path)
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
        "num_epochs": 5,
        "batch_size": 16,
        "file_path": 'D:\Search\FT\\MOSEI.pkl',
        "n_mels": 128,
        "n_audio_ctx": 1500,
        "n_audio_state": 1280,
        "n_audio_head": 20,
        "n_audio_layer": 32,

        "save_path": "MOSEIModel",
        "load_path": "SIMS_whisper_0.pth",

    }
    _set_logger(log_dir='D:\Search\FT\\audio_trainer')
    args = edict(args)
    # file_path = '/SIMS.pkl'
    dataset_train = MyDataset(args, 'train')
    dataset_val = MyDataset(args, 'valid')
    dataset_test = MyDataset(args, 'test')

    train_dataloader = DataLoader(dataset_train, batch_size=args["batch_size"], shuffle=True, collate_fn=my_collate)
    valid_dataloader = DataLoader(dataset_val, batch_size=args["batch_size"], shuffle=True, collate_fn=my_collate)
    test_dataloader = DataLoader(dataset_test, batch_size=args["batch_size"], shuffle=True, collate_fn=my_collate)

    model = MyModel(args).to(args["device"])
    model.load_state_dict(torch.load(args["load_path"]))
    logger.info("succeed loading weights")
    trainer = trainer(args, model)
    logger.info('Start training...')

    # train
    epoch_results = trainer.do_train(train_dataloader, valid_dataloader)

    logger.info('End training...')
    # test
    test_results = trainer.do_test(test_dataloader)
    criterions = list(test_results["test_result"].keys())
    # save result to csv
    csv_file = Path(f"D:\Search\FT\\audio.csv")
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





