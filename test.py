import torch  
from torch.utils.data import Dataset, random_split, DataLoader  
  
# 假设你已经有了一个包含58个样本，每个样本有5个特征的数据集  
# data 是一个形状为 (58, 5) 的numpy数组  
# labels 是一个长度为58的标签列表或数组  
# 这里只是示例，你需要替换成你实际的数据  
data = torch.randn(58, 5)  # 示例数据  
labels = torch.randint(0, 2, (58,))  # 示例标签，假设是二分类问题  
  
# 创建一个自定义的Dataset类  
class MyDataset(Dataset):  
    def __init__(self, data, labels):  
        self.data = data  
        self.labels = labels  
          
    def __len__(self):  
        return len(self.data)  
      
    def __getitem__(self, idx):  
        return self.data[idx], self.labels[idx]  
  
# 实例化Dataset对象  
dataset = MyDataset(data, labels)  
  
# 设定训练集和测试集的比例，例如80%用于训练，20%用于测试  
train_size = int(0.8 * len(dataset))  
test_size = len(dataset) - train_size  
  
# 使用random_split切分数据集  
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])  
print(len(train_dataset),len(test_dataset))
  
# 创建DataLoader对象，用于加载数据批次  
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  
  
# 现在你可以使用train_loader和test_loader来训练和测试你的模型了