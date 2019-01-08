import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import cython


class DataLoader():
    def __init__(self, cols, testcols):
        self.cols = cols
        self.testcols = testcols

    def get_train_data(self, df, assetCode=None, assetName=None, seq_len=5, normalize=True):
        '''
        Create x, y train sliding sequence windows 
        *not generative, so only use with one company*
        '''
        if assetName:
            data_train = df.loc[df.assetName==assetName, self.cols].values
        else:
            data_train = df.loc[df.assetCode==assetCossde, self.cols].values

        self.len_train = len(data_train)
        x_train = []
        y_train = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(data_train, i, seq_len, normalize)
            x_train.append(x)
            y_train.append(y)
        return np.array(x_train), np.array(y_train)

    def generate_test_batch(self, df, assetCodes, seq_len, normalize=True):
        data_windows = []
        for i,asset in enumerate(assetCodes):
            window = df.loc[df.assetCode==asset, self.testcols].tail(seq_len).values
            window = np.array(window).astype(float)
            if window.shape[0] != seq_len:
                pad = np.zeros((seq_len-window.shape[0],len(self.testcols)))
                window = np.vstack((pad,window))
            data_windows.append(window)
        data_windows = np.array(data_windows).astype(float)
        print(f'normalizing the days data of shape {data_windows.shape}')
        data_windows = self.normalize_windows(data_windows, single_window=False) if normalize else data_windows
        return np.array(data_windows)

    def _next_window(self, data, i, seq_len, normalize):
        '''Generates the next data window from the given index location i'''
        window = data[i:i+seq_len]
        window = self.normalize_windows(window, single_window=True)[0] if normalize else window
        x = window[:,1:]
        y = np.where(window[-1, [0]]>0,1,0)
        return x, y

    def normalize_windows(self, window_data, single_window=False):
        '''normalize window with a base value of zero'''
        normalized_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            scaler = StandardScaler()
            normalized_window = scaler.fit_transform(window)
            normalized_data.append(normalized_window)
        return np.array(normalized_data)

if __name__ == '__main__':
    pass
    
 