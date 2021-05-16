# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 23:31:14 2021

@author: P76094046
"""
import pandas as pd
import numpy as np
import datetime
import time
import torch
import torch.nn as nn
import csv 

# You should not modify this part.
def config():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--consumption", default="./sample_data/consumption.csv", help="input the consumption data path")
    parser.add_argument("--generation", default="./sample_data/generation.csv", help="input the generation data path")
    parser.add_argument("--bidresult", default="./sample_data/bidresult.csv", help="input the bids result path")
    parser.add_argument("--output", default="output.csv", help="output the bids path")

    return parser.parse_args()


def output(path, a_list, p_list, v_list):
    
    df = pd.DataFrame(data, columns=["time", "action", "target_price", "target_volume"])
    df.to_csv(path, index=False)

    return

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
    
def data_processor(data, mean, std):
    # data_normalize = (data.values - mean) / std 
    # data = np.expand_dims(data, axis = 1)
    data = data[np.newaxis, :]
    data = data[: 168]
    data = torch.from_numpy(data).type(torch.Tensor)
    # data = data.float().cuda()
    return data
def data_processor2(data, mean, std):
    # data_normalize = (data.values - mean) / std 
    data = np.expand_dims(data, axis = 1)
    data = data[np.newaxis, :]
    data = data[: 168]
    data = torch.from_numpy(data).type(torch.Tensor)
    # data = data.float().cuda()
    return data

if __name__ == "__main__":
    args = config()
    
    generation = pd.read_csv(args.generation, index_col = 0, parse_dates = True)
    consumption = pd.read_csv(args.consumption, index_col = 0, parse_dates = True)
    time = generation.index[-1]
    
    model_gen = torch.load('model.hdf5', map_location=torch.device('cpu'))
    model_con = torch.load('model2.hdf5', map_location=torch.device('cpu'))
    # model_gen.cuda()
    # model_con.cuda()
    model_gen.eval()
    model_con.eval()
    
    mean_gen = 0.7808354380859563
    mean_con = 1.4448073121851448
    std_gen = 1.1890449107112804
    std_con = 1.2769669115909832
    
    gen_norm = (generation.values - mean_gen) / std_gen
    con_norm = (consumption.values - mean_con) / std_con
    
    gen = data_processor(gen_norm, mean_gen, std_gen)
    con = data_processor(con_norm, mean_con, std_con)
    # print(gen.shape)
    # pred = model(data) 
    pred_gen = model_gen(gen)
    pred_con = model_con(con)
    
    pred_gen_list = []
    pred_gen_list.append(pred_gen)
    pred_con_list = []
    pred_con_list.append(pred_con)
    
    for i in range(24):
        gen_norm = np.append(gen_norm[1:168], np.array((pred_gen.cpu().detach().numpy() - mean_gen) / std_gen))
        gen = data_processor2(gen_norm, mean_gen, std_gen)
        pred_gen = model_gen(gen)
        pred_gen_list.append(pred_gen)
    
        con_norm = np.append(con_norm[1:168], np.array((pred_con.cpu().detach().numpy() - mean_con) / std_con))
        con = data_processor2(con_norm, mean_con, std_con)
        pred_con = model_con(con)
        pred_con_list.append(pred_con)
    
    
    action_list = []
    target_price_list = []
    target_volume_list = []
    for i in range(len(pred_gen_list)):
        print(pred_gen_list[i], pred_con_list[i])
        if (pred_gen_list[i] >= pred_con_list[i]):           # 如果產電大於耗電 則賣電
            action_list.append("sell")
            target_price_list.append(3)
            target_volume_list.append(abs(abs(pred_gen_list[i]) - abs(pred_con_list[i])))
        elif (pred_gen_list[i] < pred_con_list[i]):         # 如果產電小於耗電 則買電
            action_list.append("buy")
            target_price_list.append(2.5)
            target_volume_list.append(abs(abs(pred_con_list[i]) - abs(pred_gen_list[i])))
    
    # Output
    with open(args.output, 'w', newline = '') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["time", "action", "target_price", "target_volume"])
        for i in range(len(action_list) -1):
            time = time + datetime.timedelta(hours=1)
            # time_ = str(time.strftime("%Y-%m-%d %H:%M:%S"))
            writer.writerow([time, action_list[i], target_price_list[i], '%.2f'%target_volume_list[i].cpu().detach().numpy()[0][0]])
    