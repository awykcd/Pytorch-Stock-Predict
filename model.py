import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=6,  # 输入尺寸为 6，表示每个时间步有 6 个特征
            hidden_size=128,  # 隐藏层大小
            num_layers=1,  # 层数
            batch_first=True)  # 批处理维度为第一个维度

        # 修改输出层以预测未来 5 个时间步
        self.out = nn.Sequential(
            nn.Linear(128, 6)  # 将隐藏层的输出映射到 6 个特征
        )

    def forward(self, x):
        # x 的尺寸为 [256, 30, 6]
        r_out, (h_n, h_c) = self.lstm(x)  # 输入 x 到 LSTM
        # 我们需要预测未来 5 个时间步，因此我们需要修改 r_out 的尺寸
        # 将最后一个时间步的输出作为序列的初始状态
        out = r_out[:, -1:, :]  # 取最后一个时间步的输出

        # 将输出通过全连接层
        out = self.out(out)

        # 我们需要预测未来 5 个时间步，因此我们需要重复使用最后一个时间步的输出
        # 将输出的尺寸从 [256, 1, 6] 调整为 [256, 5, 6]
        out = out.expand(-1, 5, -1)

        return out