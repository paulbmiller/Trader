# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from statsmodels.tools.eval_measures import rmse
import datetime


def preprocess(history_filename, cols_w_dollarsign, cols2drop=[],
               add_mean=True):
    df = pd.read_csv(history_filename, sep=', ', parse_dates=True,
                     index_col='Date', engine='python')
    
    for col in cols_w_dollarsign:
        df = remove_dollarsign(df, col)
    
    df['Mean'] = (df['High'] + df['Low']) / 2
    
    df = df.drop(cols2drop, axis=1)
    
    df = df.sort_index()

    return df


def remove_dollarsign(df, col_name):
    df[col_name] = df[col_name].apply(lambda x: float(x[1:]))
    
    return df


class LSTM(nn.Module):
    """Class for the LSTM model"""
    def __init__(self, input_size=2, hidden_layer=200, n_input=20):
        super().__init__()
        self.input_size = input_size
        self.hidden_layer = hidden_layer
        self.n_input = n_input
        
        self.lstm = nn.LSTM(input_size, hidden_layer)
        
        self.linear = nn.Linear(hidden_layer * self.n_input, input_size)
        
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer),
                            torch.zeros(1, 1, self.hidden_layer))
    
    def forward(self, x):
        lstm_out, self.hidden_cell = self.lstm(x, self.hidden_cell)
        predictions = self.linear(lstm_out.reshape(1, -1))
        return predictions


if __name__ == '__main__':
    cols2drop = ['Open', 'Close/Last', 'Volume', 'High', 'Low']
    amazon = preprocess('AmazonHistory.csv', ['High', 'Low'], cols2drop)
    apple = preprocess('AppleHistory.csv', ['High', 'Low'], cols2drop)
    microsoft = preprocess('MicrosoftHistory.csv', ['High', 'Low'], cols2drop)
    amazon.columns = ['Amazon']
    df = amazon
    df['Apple'] = apple
    df['Microsoft'] = microsoft
    
    diffed = False
    n_input = 20
    epochs = 25
    input_size = 3
    n_preds = 1
    test_size = 20
    
    model = LSTM(n_input=n_input, input_size=input_size)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train = df.iloc[:-test_size].copy()
    test = df.iloc[-test_size:].copy()
    
    # Data differencing
    if diffed:
        train = train.diff()[1:]
        last_val = df.iloc[len(train)-1]
    
    scaler = MinMaxScaler()
    
    # Here we choose a smoothing window so that values are scaled within their
    # own time period to avoid that early low stock data has little impact
    smoothing_window = 500
    for i in range(0, 2000, smoothing_window):
        scaler.fit(train.iloc[i:i+smoothing_window,:])
        train.iloc[i:i+smoothing_window,:] = scaler.transform(
            train.iloc[i:i+smoothing_window,:])

    scaler.fit(train.iloc[i+smoothing_window:,:])
    train.iloc[i+smoothing_window:,:] = scaler.transform(
        train.iloc[i+smoothing_window:,:])
    
    test_sc = scaler.transform(test)
    
    inputs = []

    for i in range(n_input, len(train)):
        inputs.append((train.iloc[i-n_input:i].values, train.iloc[i].values))
    
    model.train()
    losses = []

    for i in range(epochs):
        for seq, label in inputs:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer),
                                 torch.zeros(1, 1, model.hidden_layer))
            seq = torch.FloatTensor(seq)
            label = torch.FloatTensor(label).reshape(-1, input_size)
            seq = seq.reshape(-1, 1, input_size)
            y_pred = model(seq)
            loss = loss_fn(y_pred, label)
            loss.backward()
            optimizer.step()
    
        L = loss.item()
        losses.append(L)
        print('epoch: {} loss: {:.7f}'.format(i, L))
    
    current_batch = train.iloc[-test_size:].values
    
    model.eval()
    preds = None
    
    for i in range(test_size-n_preds+1):
        current_batch = torch.FloatTensor(current_batch).reshape(-1, 1, input_size)

        for j in range(n_preds):
            with torch.no_grad():
                model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer),
                                     torch.zeros(1, 1, model.hidden_layer))
            
            pred = model(current_batch)
            if preds is None:
                preds = pred.detach().numpy()
            else:
                preds = np.append(preds, pred.detach().numpy(), axis=0)
        
            current_batch = torch.cat((current_batch[1:],
                                       pred.detach().reshape(-1, 1, input_size)))
    
        if i < test_size-n_preds-1:
            current_batch = np.append(train.iloc[-test_size+i+1:].values,
                                      test_sc[:i+1], axis=0)
    
    preds_unsc = scaler.inverse_transform(preds.reshape(-1, input_size))
    
    stocks = ['Amazon', 'Apple', 'Microsoft']
    colors = ['blue', 'orange', 'green']
    
    for i in range(input_size):
        plt.figure()
        title = '{} {}-day stock predictions'.format(stocks[i], n_preds)
        label = 'Predictions'
        df[stocks[i]].iloc[-2*test_size:].plot(figsize=(12, 6), color=colors[i])
        for j in range(len(test.index)-n_preds+1):
            if j == 0:
                day_before = df.index[len(train)-1]
            else:
                day_before = test.index[j-1]

            x = test.index[j:j+n_preds].insert(0, day_before)

            y = np.append(df.loc[day_before].values[i],
                          preds_unsc[n_preds*j:n_preds*j+n_preds, i])
            
            plt.title(title)
            plt.plot(x, y, color=colors[i], linestyle='--', label=label)
        
        plt.show()

    df4diff = df.iloc[len(train)-1:len(train)].copy()
    dfdiff = df4diff.append(test)
    dfdiff = (dfdiff - dfdiff.shift(1)).dropna()
    dfdiff = (dfdiff>0).astype(int)

    preds = pd.DataFrame(preds_unsc, index=test.index,
                         columns=['Amazon', 'Apple', 'Microsoft'])
    preds_diff = ((preds - df.iloc[len(train)-1:-1])>0).astype(int)
    
    dfdiff = dfdiff - preds_diff
    for stock in stocks:
        print('{} : {} wrong predictions out of {}'.format(
            stock, int(dfdiff[stock].sum()), test_size))

