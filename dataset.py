from torch.utils.data import Dataset
import torch

class TrainSet(Dataset):
    def __init__(self, df):
        # ��ȡCSV�ļ�
        self.df = df

        # ȷ�������ǰ�����������ģ����������Ѿ���ȷ����
        # �������δ��ȷ���ã�ȡ����һ�е�ע��
        # self.df = self.df.sort_values(by=['��Ʊ����', '���ڴ���']).set_index('������')

        # ���չ�Ʊ������飬��ɸѡ��������35�����ݵ���
        self.groups = self.df.groupby(level=0)

        # �洢ת��������ݶ�(x, y)
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