import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import datautil

def get_data_loader(file_path, file_name, batch_size):
    
    # 示例数据
    train_dataset, test_dataset = datautil.get_data_fromfile(file_path,file_name)
    

    # 创建数据集和数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    # print('sssaaaa',train_dataset[1])
    return train_loader, test_loader



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file_path = r'data/2024年1月5日数据'
    filename_filter = 'out.txt'
    dataloader = get_data_loader(file_path,filename_filter,10)
    # for batch_x, batch_y in dataloader:
    #     batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    #     print('batch_x',batch_x)
    #     print('batch_y',batch_y)
    #     break
    # print('test',dataloader.dataset)
    pass