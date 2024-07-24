import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataset import TrainSet
from model import LSTM
from tqdm import tqdm


df = pd.read_csv('dataset/dataset_train_clean.csv', index_col=0, encoding="gbk")


LR = 0.0001
EPOCH = 1000
TRAIN_END = -300
DAYS_BEFORE = 30
DAYS_PRED = 7

# 创建 dataloader
train_set = TrainSet(df)
# 划分数据
split_ratio = 0.8
split_idx = int(len(train_set) * split_ratio)
train_data = train_set[:split_idx]
val_data = train_set[split_idx:]
train_loader = DataLoader(train_data, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_data, batch_size=1024, shuffle=True)

model = LSTM()

if torch.cuda.is_available():
    model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.MSELoss()

best_loss = 1000

if not os.path.exists('weights'):
    os.mkdir('weights')

# 假设best_loss初始化为一个很大的数
best_loss = float('inf')

# 检查并加载模型
model_path = 'weights/model_best.pth'
if torch.path.exists(model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_loss = checkpoint['best_loss']
    start_epoch = checkpoint['epoch']
else:
    start_epoch = 0

EPOCH = 10  # 假设总训练轮次

for step in tqdm(range(start_epoch, EPOCH), desc='Epoch'):
    train_loss_sum = 0.0
    batch_count = 0

    # 训练循环
    for tx, ty in train_loader:
        if torch.cuda.is_available():
            tx, ty = tx.cuda(), ty.cuda()

        output = model(tx)
        loss = loss_func(torch.squeeze(output), ty)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()
        batch_count += 1

    train_avg_loss = train_loss_sum / batch_count
    tqdm.write(f'Epoch {step + 1}: Train Avg Loss = {train_avg_loss:.4f}')

    # 验证循环
    with torch.no_grad():
        val_loss_sum = 0.0
        val_batch_count = 0
        for tx, ty in val_loader:
            if torch.cuda.is_available():
                tx, ty = tx.cuda(), ty.cuda()

            output = model(tx)
            loss = loss_func(torch.squeeze(output), ty)

            val_loss_sum += loss.item()
            val_batch_count += 1

        val_avg_loss = val_loss_sum / val_batch_count
        tqdm.write(f'Epoch {step + 1}: Val Avg Loss = {val_avg_loss:.4f}')

        # 更新最佳损失和保存模型
        if val_avg_loss < best_loss:
            best_loss = val_avg_loss
            torch.save({
                'epoch': step + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss
            }, model_path)
            print(f'New model saved at epoch {step + 1} with val_loss {best_loss}')

# rnn = LSTM()
# rnn = torch.load('weights/rnn.pkl')
#
# generate_data_train = []
# generate_data_test = []
#
# # 测试数据开始的索引
# test_start = len(all_series_test1) + TRAIN_END
#
# # 对所有的数据进行相同的归一化
# all_series_test1 = (all_series_test1 - train_mean) / train_std
# all_series_test1 = torch.Tensor(all_series_test1)
#
# # len(all_series_test1)  # 3448
#
# for i in range(DAYS_BEFORE, len(all_series_test1) - DAYS_PRED, DAYS_PRED):
#     x = all_series_test1[i - DAYS_BEFORE:i]
#     # 将 x 填充到 (bs, ts, is) 中的 timesteps
#     x = torch.unsqueeze(torch.unsqueeze(x, dim=0), dim=2)
#
#     if torch.cuda.is_available():
#         x = x.cuda()
#
#     y = torch.squeeze(rnn(x))
#
#     if i < test_start:
#         generate_data_train.append(torch.squeeze(y.cpu()).detach().numpy() * train_std + train_mean)
#     else:
#         generate_data_test.append(torch.squeeze(y.cpu()).detach().numpy() * train_std + train_mean)
#
# generate_data_train = np.concatenate(generate_data_train, axis=0)
# generate_data_test = np.concatenate(generate_data_test, axis=0)
#
# # print(len(generate_data_train))   # 3122
# # print(len(generate_data_test))    # 294
#
# plt.figure(figsize=(12, 8))
# plt.plot(df_index[DAYS_BEFORE: len(generate_data_train) + DAYS_BEFORE], generate_data_train, 'b',
#          label='generate_train')
# plt.plot(df_index[TRAIN_END:len(generate_data_test) + TRAIN_END], generate_data_test, 'k', label='generate_test')
# plt.plot(df_index, all_series_test1.clone().numpy() * train_std + train_mean, 'r', label='real_data')
# plt.legend()
# plt.show()
#
# plt.figure(figsize=(10, 16))
#
# plt.subplot(2, 1, 1)
# plt.plot(df_index[100 + DAYS_BEFORE: 130 + DAYS_BEFORE], generate_data_train[100: 130], 'b', label='generate_train')
# plt.plot(df_index[100 + DAYS_BEFORE: 130 + DAYS_BEFORE],
#          (all_series_test1.clone().numpy() * train_std + train_mean)[100 + DAYS_BEFORE: 130 + DAYS_BEFORE], 'r',
#          label='real_data')
# plt.legend()
#
# plt.subplot(2, 1, 2)
# plt.plot(df_index[TRAIN_END + 50: TRAIN_END + 80], generate_data_test[50:80], 'k', label='generate_test')
# plt.plot(df_index[TRAIN_END + 50: TRAIN_END + 80],
#          (all_series_test1.clone().numpy() * train_std + train_mean)[TRAIN_END + 50: TRAIN_END + 80], 'r',
#          label='real_data')
# plt.legend()
#
# plt.show()
#
# generate_data_train = []
# generate_data_test = []
#
# all_series_test2 = np.array(all_series.copy().tolist())
#
# # 对所有的数据进行相同的归一化
# all_series_test2 = (all_series_test2 - train_mean) / train_std
# all_series_test2 = torch.Tensor(all_series_test2)
#
# iter_series = all_series_test2[:DAYS_BEFORE]
#
# index = DAYS_BEFORE
#
# while index < len(all_series_test2) - DAYS_PRED:
#     x = torch.unsqueeze(torch.unsqueeze(iter_series[-DAYS_BEFORE:], dim=0), dim=2)
#
#     if torch.cuda.is_available():
#         x = x.cuda()
#
#     y = torch.squeeze(rnn(x))

#     iter_series = torch.cat((iter_series.cpu(), y.cpu()))
#
#     index += DAYS_PRED
#
# iter_series = iter_series.detach().cpu().clone().numpy() * train_std + train_mean
#
# print(len(all_series_test2))
# print(len(df_index))
# print(len(iter_series))
#
# plt.figure(figsize=(12, 8))
# plt.plot(df_index[: len(iter_series)], iter_series, 'b', label='generate_train')
# plt.plot(df_index, all_series_test2.clone().numpy() * train_std + train_mean, 'r', label='real_data')
# plt.legend()
# plt.show()
#
# plt.figure(figsize=(10, 16))
#
# plt.subplot(2, 1, 1)
# plt.plot(df_index[3000: 3049], iter_series[3000:3049], 'b', label='generate_train')
# plt.plot(df_index[3000: 3049], all_series_test2.clone().numpy()[3000: 3049] * train_std + train_mean, 'r',
#          label='real_data')
# plt.legend()
#
# plt.show()
