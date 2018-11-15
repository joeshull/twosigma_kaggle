import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

class DataPrepper():
    """A class for loading and transforming data for stock data
    Instantiates with:

    INPUTS:
    Columns to drop from the market df - List of strings
    Columns to drop from the news df - List of strings
    A Split date for Cross Validation data - integer YYYYMMDD
    """

    def __init__(self, drop_market=None, drop_news=None, split=20151231):
        self.train_cutoff = 20081231
        self.test_split = split
        self.train_data = None

    def load_data(self, market_file, news_file):
        """
        Load data into class for processing:

        Inputs:
        market_file - string location of pickled df
        news_file - string location of pickled df


        Outputs:
        None
        """

        self.market_train = pd.read_pickle(market_file)
        self.news_train = pd.read_pickle(news_file)

    def make_price_diff(self, market_train):
        """eda function to find outliers
        Inputs:
        market_train - df of financial data

        Output:
        Dataframe with new columns:
        'closeOverOpen' - Close/Open
        'priceDiff' - Close minus Open

        """
        market_train['closeOverOpen'] = market_train['Close']/market_train['Open']
        market_train['priceDiff'] = np.abs(market_train['Close'] - market_train['Open'])
        return market_train

    def _replace_price_outliers(self, market_train=None):
        """
        Hidden Function to replace outlier/incorrect open and close data

        """

        if market_train is None:
            market_train = self.market_train
            trainprep = True
        market_train['dailychange'] = market_train['close']/market_train['open']
        market_train.loc[market_train['dailychange'] < .33,'open'] = market_train['close']
        market_train.loc[market_train['dailychange'] > 2, 'close'] = market_train['open']
        if trainprep:
            self.market_train = market_train
        else:
            return market_train

    def prepare_market(self, market_train=None):
        """
        Prepares the market_train dataframe for merging.
        Performs all aggregation and datacleaning functions

        Input:
        market_train - (optional) Dataframe


        Output:
        (optional) Dataframe of prepared data (or stored to object)

        """


        if market_train is None:
            market_train = self.market_train
            self._replace_price_outliers()
            trainprep = True
        else:
            market_train = self._replace_price_outliers(market_train)
        market_train['time'] = market_train['time'].dt.strftime("%Y%m%d").astype(int)
        market_train = market_train[market_train.time >= self.train_cutoff]
        market_train['todayreturnraw'] = market_train['close']/market_train['open']
        market_train['pricevolume'] = market_train['volume']/market_train['close']
        self.tradingdays = market_train['time'].unique()
        if trainprep:
            self.market_train = market_train
        else:
            return market_train

    def prepare_news(self, news_train=None, market_train=None):

        """
        Prepares the news_train dataframe for merging.
        Performs all aggregation and datacleaning functions

        Input:
        news_train - (optional) Dataframe
        market_train - (optional) Dataframe. If news_train, is passed, market_train must also be passed
        for trading-day news merge to work.

        Output:
        (optional) Dataframe of prepared data (or stored to object)

        """
        if news_train is None:
            news_train = self.news_train
            self.tradingdays = self.market_train['time'].unique()
            trainprep= True
        else:
            self.tradingdays = market_train['time'].unique()
        news_train['time'] = news_train['time'].dt.strftime("%Y%m%d").astype(int)
        news_train = news_train[news_train.time >= self.train_cutoff]
        news_train['time'] = news_train['time'].apply(self._map_trading_day)
        news_train['coverage'] = news_train['sentimentWordCount']/news_train['wordCount']
        if trainprep:
            self.news_train = news_train
        else:
            return market_train

    def _map_trading_day(self, news_date):
        """
        Hidden function for datafame.map.
        Maps the news_date to its respective trading day.

        """
        if news_date in self.tradingdays:
            return news_date
        else:   
            values = self.tradingdays - news_date
            mask = values >= 0
            try:
                return self.tradingdays[mask][0]
            except:
                return 0

    def merge_data(self, market_df=None, news_df=None):
        """
        Merges Market and News Data

        Input:
        market_df - (optional) previously prepared market dataframe
        news_df - (optional) previously prepared news dataframe


        Output:
        Dataframe 

        """

        if market_df is None and news_df is None:
            market_df = self.market_train
            news_df = self.news_train
            trainprep = True
        newsgroup = news_df.groupby(['time', 'assetName'], sort=False).agg(np.mean).reset_index()
        merged = pd.merge(market_df, newsgroup, how='left', on=['time', 'assetName'], copy=False)
        merged.fillna(value=0, inplace=True)
        if trainprep:
            self.train_data = merged
        else:
            return merged
    def prepare_train_data(self):
        """
        If data is training data, run this after calling load_data()

        """
        self.prepare_market()
        self.prepare_news()
        self.merge_data()

    def train_test_split(self, X, y, split_date=20151231):
        """
        Splitting function to create a validation set from Training Data,

        Inputs:
        X - Dataframe of feature data including 'time' as integer
        y - Np.array or Dataframe - The target of the same length as X
        split_date - (Integer) Date to make split

        Outputs:
        X_train, X_test, y_train, y_test

        """

        mask = X.time <= split_date
        return X[mask], X[~mask], y[mask], y[~mask]



class DataLoader():
    def __init__(self, filename, split, cols):
        self.dataframe = pd.read_csv(filename)
        i_split = int(len(self.dataframe) * split)
        self.data_train = self.dataframe.loc[:,cols].iloc[:i_split].values
        self.data_test  = self.dataframe.loc[:,cols].iloc[i_split:].values
        self.len_train  = len(self.data_train)
        self.len_test   = len(self.data_test)
        self.len_train_windows = None

    def get_test_data(self, seq_len, normalize):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])
        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalize_windows(data_windows, single_window=False) if normalize else data_windows

        x = data_windows[:,:,1:]
        y = np.where(data_windows[:, -1, [0]]>0, 1,0)
        return x,y

    def get_train_data(self, seq_len, normalize):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalize)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalize):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalize)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalize):
        '''Generates the next data window from the given index location i'''
        window = self.data_train[i:i+seq_len]
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

    env_market = '../data/market_train_env.pkl'
    env_news = '../data/news_train_env.pkl'

    data = DataPrepper()
    data.load_data(env_market, env_news)
    data.prepare_train_data()

