import torch
import torch.nn as nn
import torch.optim as optim

from model import FcNN
from train import train_model
from data_loader import get_data_loader



if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 模型参数
    input_size = 5
    hidden_size = 128
    output_size = 5
    dropout_rate = 0.1
    batch_size = 96
    num_epochs = 500
    learning_rate = 0.001
    file_path = r'data/2024年1月5日数据'
    filename = 'out.txt'

    # 创建模型实例并将其迁移到设备上
    model = FcNN(input_size, hidden_size, output_size, dropout_rate).to(device)

    # 定义损失函数和优化器
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 获取数据加载器
    train_loader, test_loader = get_data_loader(file_path, filename, batch_size)

    # # 训练模型
    train_model(model, train_loader, test_loader, loss_fn, optimizer, device, num_epochs)
   