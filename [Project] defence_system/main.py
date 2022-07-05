from operator import index
import torch
import torch.nn as nn
import net
import pandas as pd
import torch.optim as optim
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class Dataset(torch.utils.data.Dataset):  
  def __init__(self, x, y):
        self.x = x
        self.y = y
  def __len__(self):
        return len(self.x)
  def __getitem__(self, index):
        X = self.x[index]
        Y = self.y[index]
        return X, Y

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, x):
        self.x = x
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        return X

if __name__ == "__main__":
    
    # 如果有GPU的話，使用GPU訓練，否則使用CPU訓練
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 讀取訓練資料的csv檔
    train_data = pd.read_csv("train_DefenseSystem.csv")
    y_train_data = train_data['event_rule_category']
    y_train_data = LabelEncoder().fit_transform(y_train_data).astype(float)
    
    # 取得所有可以做為特徵的類別
    categorical_cols = [col for col in train_data.columns]
    categorical_cols.remove("event_rule_category")
    x_train_data = train_data.drop(columns=['event_rule_category'])
    len_of_x_train_data = len(x_train_data)
    
    # 讀取測試資料的csv檔
    x_test_data = pd.read_csv("test_DefenseSystem.csv")
    x_all_data = pd.concat(objs=[x_train_data, x_test_data], axis=0)
    
    # 轉換成one hot encoding
    df = pd.get_dummies(x_all_data, columns = categorical_cols)
    df = df.values.astype(float)
    x_train_data_processed = df[:len_of_x_train_data]
    x_test_data_processed = df[len_of_x_train_data:]
    
    # 切割訓練以及驗證資料
    x_train, x_val, y_train, y_val = train_test_split(x_train_data_processed, y_train_data, test_size=0.2, random_state=1)
    
    # 記錄三個模型的error rate
    all_err = []

    # 訓練模型時的超參數設定
    epoch_num = 30
    L = 0.0001
    print_freq = 5
    valid_freq = 10
    
    # 使用 cross entropy 作為我們的 loss function
    criterion = nn.BCEWithLogitsLoss()
    
    params = {'batch_size': 32,
              'shuffle': False,
              'num_workers': 1} # train
    params2 = {'batch_size': 1,   
              'shuffle': False,
              'num_workers': 1}  # valid
    params3 = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 1}  # test

    # 實例化資料集
    train_set = Dataset(np.array(x_train), np.array(y_train))
    valid_set = Dataset(np.array(x_val), np.array(y_val))
    test_set = TestDataset(np.array(x_test_data_processed))

    # 實例化dataloader
    training_dataloader = torch.utils.data.DataLoader(train_set, **params)
    valid_dataloader = torch.utils.data.DataLoader(valid_set, **params2)
    test_dataloader  = torch.utils.data.DataLoader(test_set, **params3)

    # 實例化第一個模型 (該模型搭建於net.py中)
    net1 = net.Net1()

    # 將模型放置在GPU上 (如果有的話)
    net1.to(device)

    # 實例化優化器，選擇adam 作為我們的優化器
    optimizer = optim.Adam(net1.parameters(), lr=L, eps=1e-08, weight_decay=0, amsgrad=False)
    
    # 開始訓練模型
    net1.train()
    for epoch in range(epoch_num):
        running_loss = 0.0
        for local_batch, local_labels in training_dataloader:
            local_batch = local_batch.float()
            local_labels = local_labels.float()
            input, labels = local_batch.to(device),local_labels.to(device)
            optimizer.zero_grad()
            outputs = net1(input)
            loss = criterion(outputs.float(), labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 印出我們的模型1, 在訓練途中的loss值    
        if epoch % print_freq == 0:
            print('net1 : [%d] loss: %.5f' % (epoch + 1, running_loss/len(training_dataloader)))

        # 開始使用validation set 驗證我們的模型1    
        if epoch % valid_freq == 0:
            valid_running_loss = 0.0
            error = 0
            for local_batch_valid, local_labels_valid in valid_dataloader:
                local_batch_valid = local_batch_valid.float()
                local_labels_valid = local_labels_valid.float()
                input, labels = local_batch_valid.to(device), local_labels_valid.to(device)
                outputs = net1(input)
                result = []
                if outputs[0][0]<0.5:
                    result.append(0)
                else:
                    result.append(1)
                if result[0] != labels[0].tolist():
                    error+=1
                loss = criterion(outputs.float(), labels.unsqueeze(1)) 
                valid_running_loss += loss.item()
            
            # 印出驗證時的錯誤率以及loss
            print("error rate: %.2f" %(error/1000*100),"%")
            print('net1_valid : [%d] loss: %.5f' % (epoch + 1, valid_running_loss/1000))
    
    all_err.append(error)
    
    # 實例化第二個模型 (該模型搭建於net.py中)
    net2 = net.Net2()
    
    # 將模型2放置在GPU上 (如果有的話)
    net2.to(device)

    # 實例化優化器，選擇adam 作為我們的優化器
    optimizer = optim.Adam(net2.parameters(), lr=L, eps=1e-08, weight_decay=0, amsgrad=False)
    
    # 開始訓練第二個模型
    net2.train()
    for epoch in range(epoch_num):
        running_loss = 0.0
        for local_batch, local_labels in training_dataloader:
            local_batch = local_batch.float()
            local_labels = local_labels.float()
            input, labels = local_batch.to(device),local_labels.to(device)
            optimizer.zero_grad()
            outputs = net2(input)
            # outputs = torch.round(outputs.squeeze(dim=-1))
            loss = criterion(outputs.float(), labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # 印出我們的模型2, 在訓練途中的loss值
        if epoch % print_freq == 0:
            print('net2 : [%d] loss: %.5f' % (epoch + 1, running_loss/4000))
            running_loss = 0.0
        
        # 開始使用validation set 驗證我們的模型2
        if epoch % valid_freq == 0:
            valid_running_loss = 0.0
            error = 0
            for local_batch_valid, local_labels_valid in valid_dataloader:
                local_batch_valid = local_batch_valid.float()
                local_labels_valid = local_labels_valid.float()
                input, labels = local_batch_valid.to(device), local_labels_valid.to(device)
                outputs = net2(input)
                result = []
                if outputs[0][0] <0.5:
                    result.append(0)
                else:
                    result.append(1)
                
                if result[0] != labels[0].tolist():
                    error+=1
                loss = criterion(outputs.float(), labels.unsqueeze(1)) 
                valid_running_loss += loss.item()
            
            # 印出驗證時的錯誤率以及loss
            print("error rate: %.2f" %(error/1000*100),"%")
            print('net2_valid : [%d] loss: %.5f' % (epoch + 1, valid_running_loss/1000))
    
    all_err.append(error)

    # 實例化第三個模型 (該模型搭建於net.py中)
    net3 = net.Net3()
    
    # 將模型3放置在GPU上 (如果有的話)
    net3.to(device)
    
    # 實例化優化器，選擇adam 作為我們的優化器
    optimizer = optim.Adam(net3.parameters(), lr=L, eps=1e-08, weight_decay=0, amsgrad=False)
    
    # 開始訓練第三個模型
    net3.train()
    for epoch in range(epoch_num):
        running_loss = 0.0
        for local_batch, local_labels in training_dataloader:
            local_batch = local_batch.float()
            local_labels = local_labels.float()
            input, labels = local_batch.to(device),local_labels.to(device)
            optimizer.zero_grad()
            outputs = net3(input)
            # outputs = torch.round(outputs.squeeze(dim=-1))
            loss = criterion(outputs.float(), labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        if epoch % print_freq == 0:
            print('net3 : [%d] loss: %.5f' % (epoch + 1, running_loss/4000))
            running_loss = 0.0

        if epoch % valid_freq == 0:
            valid_running_loss = 0.0
            error = 0
            for local_batch_valid, local_labels_valid in valid_dataloader:
                local_batch_valid = local_batch_valid.float()
                local_labels_valid = local_labels_valid.float()
                input, labels = local_batch_valid.to(device), local_labels_valid.to(device)
                outputs = net3(input)
                result = []
                if outputs[0][0]<0.5:
                    result.append(0)
                else:
                    result.append(1)
                if result[0] != labels[0].tolist():
                    error+=1
                loss = criterion(outputs.float(), labels.unsqueeze(1)) 
                valid_running_loss += loss.item()

            print("error rate: %.2f" % (error/1000 *100),"%")
            print('net3_valid : [%d] loss: %.5f' % (epoch + 1, valid_running_loss/1000))

    all_err.append(error)
    best_model_index = all_err.index(max(all_err))
    
    # 使用表現最好的模型來預測test的結果，並將結果輸出 
    result =[]
    for local_batch in test_dataloader:
        local_batch = local_batch.float()
        input = local_batch.to(device)
        
        if best_model_index==0:
            output = net1(input)
        elif best_model_index==1:
            output = net2(input)
        else:
            output = net3(input)

        if output[0]<0.5:
            result.append("Access Control")
        else:
            result.append("Web Attack")
    
    result = {"result":result}
    print(result)
    df = pd.DataFrame(data=result)
    df.to_csv("./result.csv",index=False)