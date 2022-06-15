import argparse, os, sys
import numpy as np
import torch
from models.architecture import autoencoder, seed_everything
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
from matplotlib import pyplot as plt
import pandas as pd

def main(args):
    seed_everything(42)

    if args.mode == 'train':
        label_data_train_Y = pd.read_csv(args.save_path + "/" + args.train_data)
        label_data_train_Y = label_data_train_Y.drop(['Unnamed: 0','cohe','perp'], axis = 1)

        label_data_train_Y = label_data_train_Y.to_numpy()
        label_data_train_Y = torch.tensor(label_data_train_Y, dtype=torch.float32)
        ds = TensorDataset(label_data_train_Y)

        loader = DataLoader(ds, batch_size=10, shuffle=True)
        model = autoencoder().cpu()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)


        losses = []

        for epoch in range(100): # epoch
            batch_loss=0
            for data in loader:
                optimizer.zero_grad()
                y_pred = model(data[0])
                loss = criterion(y_pred.view_as(data[0]), data[0])
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
            losses.append(batch_loss)
            print('epoch [{}], loss:{:.4f}'.format(epoch,batch_loss))

        plt.plot(losses,label="Training Loss")
        plt.savefig(args.save_path + '/Train_Val_loss.png')

        label_data_train_pred = model(label_data_train_Y)
        label_data_train_pred = label_data_train_pred.detach().numpy()
        label_data_train_Y = label_data_train_Y.detach().numpy()

        label_data_train_loss = np.mean(np.square(label_data_train_pred - label_data_train_Y), axis=1)
        threshold = np.mean(label_data_train_loss) + args.sigma * np.std(label_data_train_loss)

        mean1 = round(np.mean(label_data_train_loss), 5)
        std1 = round(np.std(label_data_train_loss), 5)

        name1 = "m" + str(mean1) +"_" +"s" +str(std1)

        save_path = args.ckpt_path + \
                    f'/{epoch}_{batch_loss}_{name1}.pt'
        torch.save(model.state_dict(), save_path)

        print("Weight 저장 ",save_path)
        print("복원 오류 임계치 : ", threshold)

    elif args.mode == 'test':
        model = autoencoder().cpu()
        model.load_state_dict(torch.load(args.ckpt_path +'/'+ args.weight_name))
        model.eval()

        label_data_test_Y = pd.read_csv(args.test_path)

        date_label = pd.DataFrame()
        date_label['date'] = label_data_test_Y['Unnamed: 0']

        label_data_test_Y = label_data_test_Y.drop(['Unnamed: 0', 'cohe', 'perp'], axis=1)

        label_data_test_Y = label_data_test_Y.to_numpy()
        label_data_test_Y = torch.tensor(label_data_test_Y, dtype=torch.float32)
        label_data_predict_Y = model(label_data_test_Y)

        label_data_predict_Y = label_data_predict_Y.detach().numpy()
        label_data_test_Y = label_data_test_Y.detach().numpy()

        label_data_test_Y_mse = np.mean(np.square(label_data_predict_Y - label_data_test_Y), axis=1)

        print(args.weight_name)

        threshold_mean = (args.weight_name).find('m')
        threshold_mean = float((args.weight_name)[threshold_mean+1:threshold_mean+7])

        threshold_std = (args.weight_name).find('s')
        threshold_std = float((args.weight_name)[threshold_std + 1:threshold_std +7])

        threshold = threshold_mean + args.sigma * threshold_std
        label_data_test_Y_anomalies = label_data_test_Y_mse > threshold

        # range = pd.date_range(args.end_date, args.start_date, freq='1D')

        label_data_test_Y_mse = pd.DataFrame(label_data_test_Y_mse)
        date_label['predict'] =label_data_test_Y_mse[0]

        print(date_label)
        print(date_label[label_data_test_Y_anomalies])
        print("임계값", threshold)
        print("불량 시점 개수", np.sum(label_data_test_Y_anomalies))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default=os.path.abspath('./Data/10_22_Data.csv'), help="dataset path")
    parser.add_argument(
        "--save_path", type=str, default=os.path.abspath('./save'), help="model save dir path")
    parser.add_argument(
        "--ckpt_path", type=str, default=os.path.abspath('./ckpt/'), help="model save dir path")
    parser.add_argument(
        "--weight_name", type=str, default='99_0.002167178269701253_m0.00014_s0.00013.pt', help="model save dir path")
    parser.add_argument(
        "--test_path", type=str, default=os.path.abspath('./save/New_Minmax_test.csv'), help="model save dir path")
    parser.add_argument(
        "--mode", type=str, default='test', help="model save dir path")
    parser.add_argument(
        "--train_data", type=str, default='Minmax_Train_data.csv', help="model save dir path")
    parser.add_argument(
        "--epochs", type=int, default=50, help="number of epochs to train (default: 100)")
    parser.add_argument(
        "--batch_size", type=int, default=16, help="input batch size for training (default: 16)")
    parser.add_argument(
        "--sigma", type=int, default=6, help="input batch size for training (default: 16)")


    # wandb
    parser.add_argument(
        "--wandb", action="store_true", default="True", help="wandb implement or not")
    parser.add_argument(
        "--entity", type=str, default="troy2331", help="wandb entity name (default: jaehwan)", )
    parser.add_argument(
        "--project", type=str, default="torch", help="wandb project name (default: torch)")

    args = parser.parse_args()
    main(args)