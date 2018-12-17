from datetime import datetime, timedelta
import numpy as np 
import pandas as pd 
from pandas.api.types import is_numeric_dtype
import matplotlib as plt 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split as sk_train_test_split

from multiprocessing import Pool
import gc


class Featurizer():
    def __init__(self, assetId='assetCode',
                       n_lag=[3,7,14],
                       shift_size=1, 
                       return_features=['returnsClosePrevMktres10','returnsClosePrevRaw10',
                                        'returnsOpenPrevMktres1', 'returnsOpenPrevRaw1',
                                        'open','close']
                ):
        self.assetId = assetId
        self.n_lag = n_lag
        self.shift_size = shift_size
        self.return_features = return_features

    def transform(self, df):
        new_df = self.generate_lag_features(df)
        df = pd.merge(df, new_df, how='left', on=['time', self.assetId])
        df = self.mis_impute(df)
        df.dropna(axis=0, inplace=True)
        return df


    def create_lag(self, df_code):
        # code = df_code.loc[:,self.assetId].unique()
        for col in self.return_features:
            for window in self.n_lag:
                rolled = df_code[col].shift(self.shift_size).rolling(window=window)
                lag_mean = rolled.mean()
                lag_max = rolled.max()
                lag_min = rolled.min()
                lag_std = rolled.std()
                df_code['%s_lag_%s_mean'%(col,window)] = lag_mean
                df_code['%s_lag_%s_max'%(col,window)] = lag_max
                df_code['%s_lag_%s_min'%(col,window)] = lag_min
                # df_code['%s_lag_%s_std'%(col,window)] = lag_std
        return df_code.fillna(-1)

    def generate_lag_features(self,df):
        features = ['time', self.assetId, 'volume', 'close', 'open',
       'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
       'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
       'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
       'returnsClosePrevMktres10', 'returnsOpenPrevMktres10']
        df = df.loc[:,features]
    
        assetCodes = df[self.assetId].unique()
        df_codes = df.groupby(self.assetId)
        df_codes = [df_code[1][['time', self.assetId]+self.return_features] for df_code in df_codes]
        pool = Pool(6)
        all_df = pool.map(self.create_lag, df_codes)
        new_df = pd.concat(all_df)  
        new_df.drop(self.return_features,axis=1,inplace=True)
        pool.close()

        return new_df

    def mis_impute(self, df):
        for i in df.columns:
            if df[i].dtype == "object":
                df[i] = df[i].fillna("other")
            elif (df[i].dtype == "int64" or df[i].dtype == "float64"):
                df[i] = df[i].fillna(df[i].mean())
            else:
                pass
        return df



def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if is_numeric_dtype(col_type):
            col_type = col_type.name
            
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


if __name__ == '__main__':

    featurizer = Featurizer()
    
    print("loading X")
    X_test = pd.read_pickle('../data/test_data.pkl')

    X_test = reduce_mem_usage(X_test)
    print("transform X")
    X_test = featurizer.transform(X_test)
    X_test = reduce_mem_usage(X_test)
    print("saved")
    X_test.to_pickle('../data/X_test_featurized.pkl')








