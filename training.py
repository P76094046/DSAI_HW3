# -*- coding: utf-8 -*-
"""
Created on Sat May  8 22:47:51 2021

@author: algo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import csv 
import time 
#%% Data Preprocessing

data_path = 'C://Users/P76094046/Desktop/DSAI/training_data'
# training = pd.read_csv(data_path+'training_data/target0.csv', parse_dates = [0])
data = pd.read_csv(data_path+'/target0.csv', index_col = 0, parse_dates = True)
label = data.values
n = len(data)


#%%
all_data = pd.DataFrame(columns=["generation", "consumption"])
for i in range(50):
    filename = "/target%d.csv" %i
    data = pd.read_csv(data_path + filename, index_col = 0, parse_dates = True)
    all_data = pd.concat([all_data, data], axis = 0)

mean_gen = (all_data.generation).mean()
mean_con = (all_data.consumption).mean()

std_gen = (all_data.generation).std()
std_con = (all_data.consumption).std()


#%% DataLoader2 
# scaler = MinMaxScaler(feature_range=(0, 1))
num_4_pred = 168
num_pred = 1

class dataset2(torch.utils.data.Dataset):     # for generation
    def __init__(self, data):
        self.data = data
        self.label = self.data.values
        # self.data_ = scaler.fit_transform(self.data.values)
        self.gen_train = (self.data.generation - mean_gen) / std_gen
        self.gen_train = np.expand_dims(self.gen_train, axis = 1)
        # print(self.gen_train.shape)
        self.gen_train_label = self.label[:, 0]
        # self.gen_val = self.data_[5000:, 0]
   
    def __len__(self):  
        return len(self.gen_train) - num_4_pred - num_pred + 1
    
    def __getitem__(self, index):
        X = self.gen_train[index : index + num_4_pred]
        y = self.gen_train_label[index + num_4_pred : index + num_4_pred + num_pred]
        X = torch.from_numpy(X).type(torch.Tensor)
        y = torch.from_numpy(y).type(torch.Tensor)
        # print(X)
        return X, y
    
batch_size = 16
num_epochs = 50 #n_iters / (len(train_X) / batch_size)

data = pd.read_csv(data_path+'/target0.csv',index_col = 0, parse_dates = True)
gen_dataset2 = dataset2(data)
gen_loader2 = torch.utils.data.DataLoader(dataset = gen_dataset2,#dataset 
                                           batch_size = batch_size, 
                                           shuffle = False)

#%% 

class dataset_con(torch.utils.data.Dataset):     # for consumption
    def __init__(self, data):
        self.data = data
        self.label = self.data.values
        # self.data_ = scaler.fit_transform(self.data.values)
        self.con_train = (self.data.consumption - mean_con) / std_con
        self.con_train = np.expand_dims(self.con_train, axis = 1)
        self.con_train_label = self.label[:, 0]
    
    def __len__(self):  
        return len(self.con_train) - num_4_pred - num_pred + 1
    
    def __getitem__(self, index):
        X = self.con_train[index : index + num_4_pred]
        y = self.con_train_label[index + num_4_pred : index + num_4_pred + num_pred]
        X = torch.from_numpy(X).type(torch.Tensor)
        y = torch.from_numpy(y).type(torch.Tensor)
        return X, y
    
#%% LSTM model
input_dim = 1
hidden_dim = 16
num_layers = 2 
output_dim = 1

# Here we define our model as a class
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        #batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        
        out, (hn, cn) = self.lstm(x, None)
        out = self.fc(hn[1]) 
        # print(hn.shape)
        # print(out)
        return out
    
model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
model2 = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
model.cuda()
model2.cuda()
loss_fn = torch.nn.MSELoss()

optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
print(model)
print(len(list(model.parameters())))
for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())
    

#%% training for 50 targets

target_id = np.arange(50)
for i in target_id:
    print(i)
    filename = "/target%d.csv" %i
    data = pd.read_csv(data_path + filename, index_col = 0, parse_dates = True)
    gen_dataset2 = dataset2(data)
    gen_loader2 = torch.utils.data.DataLoader(dataset = gen_dataset2,#dataset 
                                              batch_size = batch_size, 
                                              shuffle = False)
    con_dataset = dataset_con(data)
    con_loader = torch.utils.data.DataLoader(dataset = con_dataset,#dataset 
                                           batch_size = batch_size, 
                                           shuffle = False)

    
    for t in range(num_epochs):
        for num , data in enumerate(gen_loader2, 0):
            input_data, label = data
            input_data, label = input_data.float().cuda(), label.float().cuda() 
            pred = model(input_data)
            # print(pred)
            loss = loss_fn(pred, label)
            print('gen_epcoh %d'%t, loss.item())
            # hist[t] = loss.item()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            
    for t in range(num_epochs):       
        for num , data in enumerate(con_loader, 0):
            input_data, label = data
            input_data, label = input_data.float().cuda(), label.float().cuda() 
            pred = model2(input_data)
            # print(pred)
            loss = loss_fn(pred, label)
            print('con_epcoh %d'%t, loss.item())
            # hist[t] = loss.item()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
           
            
torch.save(model, "C://Users/P76094046/Desktop/DSAI/HW3/model3.hdf5")
torch.save(model2, "C://Users/P76094046/Desktop/DSAI/HW3/model4.hdf5")
#%% 

filename = "/target0.csv" 
data = pd.read_csv(data_path + filename, index_col = 0, parse_dates = True)
gen_dataset2 = dataset2(data)
gen_loader2 = torch.utils.data.DataLoader(dataset = gen_dataset2,#dataset 
                                              batch_size = batch_size, 
                                              shuffle = False)
con_dataset = dataset_con(data)
con_loader = torch.utils.data.DataLoader(dataset = con_dataset,#dataset 
                                           batch_size = batch_size, 
                                           shuffle = False)

    
for t in range(num_epochs):
    for num , data in enumerate(gen_loader2, 0):
        input_data, label = data
        input_data, label = input_data.float().cuda(), label.float().cuda() 
        pred = model(input_data)
        # print(pred)
        loss = loss_fn(pred, label)
        print('gen_epcoh %d'%t, loss.item())
        # hist[t] = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
            
for t in range(num_epochs):       
    for num , data in enumerate(con_loader, 0):
        input_data, label = data
        input_data, label = input_data.float().cuda(), label.float().cuda() 
        pred = model2(input_data)
        # print(pred)
        loss = loss_fn(pred, label)
        print('con_epcoh %d'%t, loss.item())
        # hist[t] = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
           
            
torch.save(model, "C://Users/P76094046/Desktop/DSAI/HW3/model.hdf5")
torch.save(model2, "C://Users/P76094046/Desktop/DSAI/HW3/model2.hdf5")

#%% forecasting

'''
generation = pd.read_csv(data_path+'/target0.csv',index_col = 0, parse_dates = True)
time0 = generation.index[-1]
gen_normalize = (generation.values[:, 0] - mean_gen) / std_gen 
gen = np.expand_dims(gen_normalize, axis = 1)
gen = gen[np.newaxis,:]
gen = gen[:167]
gen = torch.from_numpy(gen).type(torch.Tensor)
    
pred_list = []

gen = gen.float().cuda()
pred = model(gen)
print(pred)
pred_list.append(pred)   

def data_processor(data, mean, std):
    # data_normalize = (data.values - mean) / std 
    data = np.expand_dims(data, axis = 1)
    data = data[np.newaxis, :]
    data = data[: 167]
    data = torch.from_numpy(data).type(torch.Tensor)
    data = data.float().cuda()
    return data

for i in range(24):
    gen_normalize = np.append(gen_normalize[1:168], np.array((pred.cpu().detach().numpy() - mean_gen) / std_gen))
    gen = data_processor(gen_normalize, mean_gen, std_gen)
    pred_gen = model(gen)
    pred_list.append(pred_gen)

'''
