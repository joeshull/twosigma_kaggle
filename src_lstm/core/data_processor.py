import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class DataPrepper():
    """A class for loading and merging the stock and news data
    Instantiates with:

    INPUTS:
    Columns to drop from the market df - List of strings
    Columns to drop from the news df - List of strings
    A Split date for Cross Validation data - integer YYYYMMDD
    """

    def __init__(self,train_cutoff=20100101):
        self.train_cutoff = train_cutoff
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


if __name__ == '__main__':

    env_market = '../data/market_train_env.pkl'
    env_news = '../data/news_train_env.pkl'

    data = DataPrepper()
    data.load_data(env_market, env_news)
    data.prepare_train_data()
    data.merge_data()
    # data.train_data.to_pickle('../data/original_merged_train.pkl')

