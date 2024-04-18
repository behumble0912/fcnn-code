import re
import numpy as np
import copy
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import random

# 切分数据集
def get_split(n, random_state, train_size, test_size):

    idx_list = list(range(n))

    random.seed(random_state)
    train_idx = random.sample(idx_list, train_size)
    idx_list = list(set(idx_list) - set(train_idx))

    random.seed(random_state)
    test_idx = random.sample(idx_list, test_size)
    idx_list = list(set(idx_list) - set(test_idx))

    val_idx = idx_list

    return train_idx, test_idx, val_idx

# 两点间距离
def calc_dist(a, b):
    # print("坐标",a[0:2],b[0:2])
    return np.linalg.norm(a-b)

# 一个坐标到四个基站的距离
def calc_dists2bases(a, bases):
    dists = []
    for b in bases:
        dists.append(calc_dist(a, b))
    dists = np.asfarray(dists)
    return dists

# 从路径中读取基站坐标文件
def get_base_coords_frompath(data_path):
    base_num = 5
    base_coords = np.zeros((base_num, 3))
    with open(os.path.join(data_path, 'bases.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            lhs = line.split(';')[0].split('=')[0]
            base_i = int(lhs.split('[')[1].split(']')[0])
            # print(base_num)
            sub_strs = line.split(';')
            val = np.asfarray([eval(sub_str.split('=')[1]) for sub_str in sub_strs if "=" in sub_str])
            base_coords[base_i] = val
    return base_coords

# 从文件中读取测量记录
def get_data_fromfile(data_path, file_name, sep='begin'):
    
    base_coords = get_base_coords_frompath(data_path)
    f_dist_mf = False
    f_gt_dist = False
    dist_mf = torch.empty(1,5)
    gt_dist = torch.empty(1,5)
    # grand truth距离
    gt_dists = []
    dists_mf = []
    print("读取记录")
    with open(os.path.join(data_path, file_name), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        

        for line in lines:
            first_word = line.split(':')[0]
            # print(first_word)
            
            try:
                first_val_strs = line.split('(')[1].split(')')[0].split(',')
                if first_word == 'end':
                    # print(f_dist_mf,f_gt_dist)
                    if f_dist_mf == True and f_gt_dist == True:
                        dists_mf.append(dist_mf)
                        gt_dists.append(gt_dist)
                        # print('gt_dists',gt_dists)
                        # print('dists_mf',dists_mf)
                    f_dist_mf = False
                    f_gt_dist = False
                first_val_np = np.asfarray([float(str_i) for str_i in first_val_strs])
            except Exception as e:
                continue
            # print(first_word)
            # 遇到一次mf 就标记一次， 遇到gt 标记一次， 每次遇到end判断是否要加入

            if first_word == 'dists mf' and f_dist_mf == False:
                    f_dist_mf = True
                    dist_mf = torch.from_numpy(first_val_np).float()
                    # print('dists mf:',dist_mf)
            
            if first_word == 'coord' and f_gt_dist == False:
                f_gt_dist = True
                gt_coord = first_val_np
                # gt_dist = calc_dists2bases(gt_coord, base_coords)
                # gt_dist = gt_dist.reshape(-1,1)
                gt_dist = torch.from_numpy(calc_dists2bases(gt_coord, base_coords)).float()
                # print(gt_dist)
                # gt_dists.append(gt_dist)
                # print(gt_dists)
                # # gt_dists = torch.cat((gt_dists,gt_dist),dim=0)

                # # print(gt_coord)
                # print('coord:',gt_dist)
            # print(f_dist_mf,f_gt_dist)


    
    x_train = torch.stack(dists_mf, dim=0)
    y_train = torch.stack(gt_dists, dim=0)
    print('gt_dists',y_train.shape)
    print('dists_mf',x_train.shape)
    # print(len(x_train))
    dataset = TensorDataset(x_train, y_train)
    # print(dataset)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    # 设置随机数种子
    torch.manual_seed(11)
    train_dataset, test_dataset = random_split(dataset,[train_size,test_size])
    print("训练集数量",len(train_dataset),'测试集数量',len(test_dataset))
    # for x in train_dataset:
    #     print(x)
    # print('train_dataset',train_dataset[0])
    # print('test_dataset',test_dataset[0])
    return train_dataset, test_dataset
            
    
    

# 从数据集路径读取测量记录
def get_data_frompath(data_path, sep='begin', cut=None, filename_filter=None):
    file_list = os.listdir(os.path.join(data_path))
    # print(file_list)
    for file_name in file_list:
        
        if filename_filter != None:
            # print(file_name)
            pattern = re.compile(filename_filter)
            if pattern.search(file_name) == None:
                continue
        print(file_name)
        get_data_fromfile(data_path, file_name, sep)

        

if __name__ == '__main__':
    # file_path = r'data/2022年7月1日车外静态仅外圈'
    file_path = r'data/2024年1月5日数据'
    a = []
    b = []
    # data = get_data_fromfile(file_path,'test.txt')
    a, b = get_data_fromfile(file_path,'out.txt')


    # print('TensorDatasetaa',a[3])
    # print(out2)
    # file_path = r'data/2022年7月1日车外静态仅外圈'
    # get_measures_frompath(file_path,filename_filter='1-1')

    pass