import torch



def train_model(model, train_loader, test_loader, loss_fn, optimizer, device, num_epochs):
    min_test_loss = 100.0
    for epoch in range(num_epochs):
        train_loss = 0.0
        # 设置模型为训练模式
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # print('batch_x',batch_x.shape)
            # print('batch_y',batch_y.shape)
            # print("--------")
            # 前向传播
            output = model(batch_x)
            
            # 计算损失
            loss = loss_fn(output, batch_y)
            # print("trainloss",loss)
            train_loss += loss.item()
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        test_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                # 前向传播
                output = model(batch_x)
                
                # 计算损失
                loss = loss_fn(output, batch_y)
                # print("batchloss:",loss)
                test_loss += loss.item()
                
        # 打印当前轮次的损失
        print(f"Epoch {epoch + 1}/{num_epochs}, train_loss: {train_loss/len(train_loader)}, test_loss: {test_loss/len(test_loader)}")
        min_test_loss = min(min_test_loss, test_loss/len(test_loader))
    print("测试集最小loss：",min_test_loss)
