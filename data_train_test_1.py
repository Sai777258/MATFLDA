import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# Data Loader

def get_data(Ai, ij):
    data = []
    for item in ij:
        feature = []
        for dim in range(Ai.shape[0]):
            mean = (Ai[dim][item[0]] + Ai[dim][item[1]]) / 2
            feature.append(mean)
        data.append(np.array(feature))
    return np.array(data)

# def get_data(Ai, ij):
#     data = []
#     for item in ij:
#         feature = np.array([Ai[0][item[0]], Ai[0][item[1]]])
#         for dim in range(1, Ai.shape[0]):
#             feature = np.concatenate((feature, np.array([Ai[dim][item[0]], Ai[dim][item[1]]])))
#         data.append(feature)
#     return np.array(data)

class myDataset(Dataset):
    def __init__(self, dataset, fold, dimension, mode='train') -> None:
        super().__init__()

        self.mode = mode
        Ai = []
        A = np.load('data/' + dataset + '/A_' + str(fold) + '.npy')
        Ai.append(A)
        for i in range(dimension - 1):
            tmp = np.dot(Ai[i], A)
            np.fill_diagonal(tmp, 0)
            tmp = tmp / np.max(tmp)
            Ai.append(copy.copy(tmp))
        Ai = np.array(Ai)
        positive_ij = np.load('data/' + dataset + '/positive_ij.npy')
        negative_ij = np.load('data/' + dataset + '/negative_ij.npy')
        positive5foldsidx = np.load('data/' + dataset + '/positive5foldsidx.npy', allow_pickle=True)
        negative5foldsidx = np.load('data/' + dataset + '/negative5foldsidx.npy', allow_pickle=True)

        if mode == 'test':
            positive_test_ij = positive_ij[positive5foldsidx[fold]['test']]
            negative_test_ij = negative_ij[negative5foldsidx[fold]['test']]

            # Balance positive and negative samples in test mode
            num_positive = len(positive_test_ij)
            if len(negative_test_ij) > num_positive:
                selected_negative_indices = np.random.choice(len(negative_test_ij), num_positive, replace=False)
                negative_test_ij = negative_test_ij[selected_negative_indices]

            positive_test_data = torch.Tensor(get_data(Ai, positive_test_ij))
            negative_test_data = torch.Tensor(get_data(Ai, negative_test_ij))
            self.data = torch.cat((positive_test_data, negative_test_data)).transpose(1, 2)
            self.target = torch.Tensor([1] * len(positive_test_ij) + [0] * len(negative_test_ij))

        elif mode == 'train':
            positive_train_ij = positive_ij[positive5foldsidx[fold]['train']]
            negative_train_ij = negative_ij[negative5foldsidx[fold]['train']]

            # Balance positive and negative samples in train mode
            num_positive = len(positive_train_ij)
            if len(negative_train_ij) > num_positive:
                selected_negative_indices = np.random.choice(len(negative_train_ij), num_positive, replace=False)
                negative_train_ij = negative_train_ij[selected_negative_indices]

            positive_train_data = torch.Tensor(get_data(Ai, positive_train_ij))
            negative_train_data = torch.Tensor(get_data(Ai, negative_train_ij))
            self.data = torch.cat((positive_train_data, negative_train_data)).transpose(1, 2)
            self.target = torch.Tensor([1] * len(positive_train_ij) + [0] * len(negative_train_ij))

        print('Finished reading the {} set of Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.data.shape[1:]))

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return len(self.data)




def prep_dataloader(dataset, fold, mode, dimension, batch_size, n_jobs=0):
    ''' Generates a dataset, then is put into a dataloader. '''
    dataset = myDataset(dataset, fold, dimension, mode=mode)
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)
    return dataloader


# Train
def focal_loss_with_label_smoothing(preds, targets, gamma=2, epsilon=0.1, lambda_entropy=0.01):

    preds = preds.view(-1)
    targets = targets.view(-1)

    # 标签平滑
    targets_smoothed = targets.float() * (1 - epsilon) + 0.5 * epsilon

    # 计算 p_t
    p_t = preds * targets_smoothed + (1 - preds) * (1 - targets_smoothed)

    # 计算焦点损失
    focal_factor = (1 - p_t) ** gamma
    ce_loss = - (targets_smoothed * torch.log(preds + 1e-8) + (1 - targets_smoothed) * torch.log(1 - preds + 1e-8))
    focal_loss = focal_factor * ce_loss

    # 熵正则项
    entropy = - (preds * torch.log(preds + 1e-8) + (1 - preds) * torch.log(1 - preds + 1e-8))

    loss = focal_loss.mean() + lambda_entropy * entropy.mean()

    return loss

def train(tr_set, model, config, gpu):
    # 损失函数
    n_epochs = config['n_epochs']
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    loss_record = []
    epoch = 0
    while epoch < n_epochs:
        model.train()
        for x, y in tr_set:
            optimizer.zero_grad()
            x, y = x.to(gpu), y.to(gpu)
            pred = model(x)
            # 确保pred经过sigmoid层输出概率
            if pred.dim() == 2 and pred.size(1) == 1:
                pred = pred.squeeze(1)
            elif pred.dim() > 1:
                pred = pred.view(-1)

            loss = focal_loss_with_label_smoothing(pred, y, gamma=2, epsilon=0.1, lambda_entropy=0.01)
            loss.backward()
            optimizer.step()
            loss_record.append(loss.detach().cpu().item())

        epoch += 1

        # 判断
        if loss.item() > 1:
            print(f"Early stopping at epoch {epoch} due to large loss {loss.item():.4f}")
            break

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    print('Finished training after {} epochs'.format(epoch))
    return loss_record



# Test

def test(tt_set, model, device):
    model.eval()
    preds = []
    labels = []
    for x, y in tt_set:
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
        labels.append(y.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return labels, preds