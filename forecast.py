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
from pytrends.request import TrendReq
from pytrends import dailydata


def over_time(kw_list, draw=True, **kwargs):
    """
    Fetches the DataFrame of the Google Trend for a list of keywords. By
    default, it will draw the plot.

    Parameters
    ----------
    kw_list : list
        List of keywords to search.
    draw : bool, optional
        Draw the bar plot for the keywords or not. The default is True.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the keywords as columns and the country as index.

    """
    pytrend = TrendReq()
    
    pytrend.build_payload(kw_list=kw_list, **kwargs)
    
    df = pytrend.interest_over_time()
    
    df = df.drop('isPartial', axis=1)
    
    if draw:
        df.plot()
    
    return df


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
    def __init__(self, input_size=3, hidden_layer=200, output_size=3,
                 n_input=20):
        super().__init__()
        self.input_size = input_size
        self.hidden_layer = hidden_layer
        self.output_size = output_size
        self.n_input = n_input
        
        self.lstm = nn.LSTM(input_size, hidden_layer)
        
        self.linear = nn.Linear(hidden_layer * self.n_input, output_size)
        
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer),
                            torch.zeros(1, 1, self.hidden_layer))
    
    def forward(self, x):
        lstm_out, self.hidden_cell = self.lstm(x, self.hidden_cell)
        predictions = self.linear(lstm_out.reshape(1, -1))
        return predictions


def evaluate(model, train_w_trends, test_w_trends, test_size, scaler,
             diffed=False, df=None):
    model.eval()
            
    current_batch = train_w_trends.iloc[-test_size:].values

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
        
            if n_preds > 1:
                current_batch = torch.cat((current_batch[1:],
                                           pred.detach().reshape(
                                               -1, 1, input_size)))
    
        if i < test_size-n_preds:
            """
            print('{}-th test batch will use dates from {} to {}'.format(
                i + 1, train.index[-test_size+i+1].strftime('%Y-%m-%d'),
                test.index[i].strftime('%Y-%m-%d')))
            """
            current_batch = np.append(
                train_w_trends.iloc[-test_size+i+1:].values,
                test_w_trends[:i+1], axis=0)
    
    stock_values = preds[:, :len(stocks)]
    preds_unsc = scaler.inverse_transform(stock_values)
    
    if diffed:
        preds_undiff = np.append(
            df.iloc[-test_size - 1].values.reshape(-1, 3), preds_unsc, axis=0)
        preds_undiff = preds_undiff.cumsum(axis=0)[1:]
        
        return preds_undiff
    
    return preds_unsc


def get_binary_errors(preds_unsc, test, stocks, dfdiff, diffed):
    preds = pd.DataFrame(preds_unsc, index=test.index, columns=stocks)
    if not diffed:
        preds_diff = ((preds - df.iloc[-len(test)-1:-1].values)>0).astype(int)
    preds_diff = preds > 0
    
    dfdiff = dfdiff - preds_diff
    for stock in stocks:
        print('{} : {} wrong predictions out of {}'.format(
            stock, int(dfdiff[stock].abs().sum()), len(test)))


if __name__ == '__main__':
    cols2drop = ['Open', 'Close/Last', 'Volume', 'High', 'Low']
    dollarsign_cols = ['High', 'Low']
    stocks = ['Amazon', 'Apple', 'Microsoft']
    for i in range(len(stocks)):
        if i == 0:
            df = preprocess(stocks[i] + 'History.csv',
                            dollarsign_cols,
                            cols2drop)
        else:
            df = pd.concat([df, preprocess(stocks[i] + 'History.csv',
                                           dollarsign_cols,
                                           cols2drop)], axis=1)
    
    df.columns=stocks
    
    kw_search = []
    
    """
    To create the Google Trend CSV files for each stock
    for i in range(len(stocks)):
        kw = stocks[i] + ' stock'
        daily = dailydata.get_daily_data(kw, 2010, 7, 2020, 7).iloc[5:-5]
        daily[[kw + '_unscaled']].to_csv(stocks[i] + 'Trend.csv')
    """
    
    """
    Notes: this is in testing, multiple combinations won't work. For example,
    the diffed attribute doesn't work properly
    """
    
    use_trends = False
    diffed = False
    n_input = 20
    epochs = 25
    eval_every = 5
    input_size = len(stocks)
    if use_trends:
        input_size += len(stocks)
    output_size = input_size
    n_preds = 1
    test_size = 20
    hidden_layer = 200
    lr = 1e-3
    kw = 'scandal'
    
    if use_trends:
        for i in range(len(stocks)):
            if i == 0:
                trends = pd.read_csv(stocks[i] + kw.capitalize() + '.csv',
                                     index_col='date', parse_dates=True)
            else:
                trends = pd.concat([trends, pd.read_csv(
                    stocks[i] + kw.capitalize() +'.csv', index_col='date',
                    parse_dates=True)], axis=1)
    
    # Data differencing
    if diffed:
        train = df.iloc[:-test_size].diff()[1:]
    else:
        train = df.iloc[:-test_size].copy()
        
    test = df.iloc[-test_size:].copy()
        
    if use_trends:
        trends_train = trends.loc[:test.index.min()].copy()
        trends_test = trends.loc[train.index.max():].copy()
        
        scaler_trends = MinMaxScaler()
        trends_train.iloc[:, :] = scaler_trends.fit_transform(
            trends_train).round(4)
        trends_test.iloc[:, :] = scaler_trends.transform(trends_test).round(4)
    
    model = LSTM(n_input=n_input, input_size=input_size,
                 hidden_layer=hidden_layer, output_size=output_size)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    scaler = MinMaxScaler()
    
    # Here we choose a smoothing window so that values are scaled within their
    # own time period to avoid that early low stock data has little impact
    if not diffed:
        smoothing_window = 500
        for i in range(0, 2000, smoothing_window):
            scaler.fit(train.iloc[i:i+smoothing_window,:])
            train.iloc[i:i+smoothing_window,:] = scaler.transform(
                train.iloc[i:i+smoothing_window,:]).round(4)
    
        scaler.fit(train.iloc[i+smoothing_window:,:])
        train.iloc[i+smoothing_window:,:] = scaler.transform(
            train.iloc[i+smoothing_window:,:]).round(4)
        
        if use_trends:
            train_w_trends = pd.concat([train, trends_train], axis=1, join='inner')
        else:
            train_w_trends = train
        
        scaled_values = scaler.transform(test)
        test_sc = test.copy()
        for i in range(len(stocks)):
            test_sc.iloc[:, i] = scaled_values[:, 0]
    
    else:
        train.iloc[:, :] = scaler.fit_transform(train)
        test_sc = test.copy()
        test_sc.iloc[:, :] = scaler.transform(test)
    
    if use_trends:
        test_w_trends = pd.concat([test_sc, trends_test], axis=1, join='inner')
    else:
        train_w_trends = train
        test_w_trends = test_sc
    
    df4diff = df.iloc[len(train)-1:len(train)].copy()
    dfdiff = df4diff.append(test)
    dfdiff = (dfdiff - dfdiff.shift(1)).dropna()
    dfdiff = (dfdiff>0).astype(int)
    
    inputs = []

    for i in range(n_input, len(train)):
        inputs.append((train_w_trends.iloc[i-n_input:i].values,
                       train_w_trends.iloc[i].values))
    
    losses = []

    for i in range(epochs):
        if i % eval_every == 0:
            model.train()
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
        print('epoch: {}/{} loss: {:.7f}'.format(i+1, epochs, L))
        
        if (i + 1) % eval_every == 0:
            preds_unsc = evaluate(model, train_w_trends, test_w_trends,
                                  test_size, scaler)
            get_binary_errors(preds_unsc, test, stocks, dfdiff, diffed)
    
    if (i + 1) % eval_every != 0:
        preds_unsc = evaluate(model, train_w_trends, test_w_trends, test_size,
                              scaler)
        get_binary_errors(preds_unsc, test, stocks, dfdiff, diffed)
    
    colors = ['blue', 'orange', 'green']

    preds = pd.DataFrame(preds_unsc, index=test.index, columns=stocks)
    
    if diffed:
        for stock in stocks:
            preds = pd.concat([df.iloc[len(train):len(train)+1], preds], axis=0)
            preds[stock] = preds[stock].cumsum()
            preds = preds.iloc[1:]
        
    for i in range(len(stocks)):
        plt.figure('{}_{}in_{}ep_{}h'.format(stocks[i].lower(),
                                             n_input,
                                             epochs,
                                             hidden_layer))
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
                          preds.iloc[n_preds*j:n_preds*j+n_preds, i])
            
            plt.title(title)
            plt.plot(x, y, color=colors[i], linestyle='--', label=label)
        
        plt.show()
    
    preds_diff = (preds_unsc > 0).astype(int)
    
    for stock in stocks:
        print('{} RMSE : {}'.format(stock, rmse(test[stock], preds[stock])))

