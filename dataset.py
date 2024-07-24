from torch.utils.data import Dataset
import torch

class TrainSet(Dataset):
    def __init__(self, df):
        # 读取CSV文件
        self.df = df

        # 确保数据是按照索引排序的，假设索引已经正确设置
        # 如果索引未正确设置，取消下一行的注释
        # self.df = self.df.sort_values(by=['股票代码', '日期代码']).set_index('索引列')

        # 按照股票代码分组，并筛选出至少有35天数据的组
        self.groups = self.df.groupby(level=0)

        # 存储转换后的数据对(x, y)
        self.x_y_pairs = []
        for name, group in self.groups:
            for i in range(30, len(group) - 5 + 1, 35):
                x = group.iloc[i - 30:i, -6:].values
                y = group.iloc[i:i + 5, -3].values
                x = torch.tensor(x, dtype=torch.float32)
                y = torch.tensor(y, dtype=torch.float32)
                self.x_y_pairs.append((x, y))

    def __getitem__(self, index):
        data = self.x_y_pairs[index]
        return data

    def __len__(self):
        return len(self.x_y_pairs)