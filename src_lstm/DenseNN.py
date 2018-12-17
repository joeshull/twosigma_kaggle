from core.data_processor import DataPrepper
from datetime import datetime, timedelta
import numpy as np 
import pandas as pd 
from pandas.api.types import is_numeric_dtype
import matplotlib as plt 
from plot_helpers.classifier_performance_plothelper import plot_coefficient_importances
from plot_helpers.model_plots import plot_confusion_matrix, plot_roc
from plot_helpers.tree_plothelper import *
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split as sk_train_test_split
import joblib

from keras.layers import Input, Dense, Activation, Dropout, LSTM, Embedding, Reshape, Flatten, concatenate, BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint

from multiprocessing import Pool

from sklearn.metrics import log_loss

import gc

class Featurizer():
    def __init__(self, assetId='assetCode',
                       n_lag=[3,7,14,],
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
       'returnsClosePrevMktres10', 'returnsOpenPrevMktres10',
        'universe']
        df = df.loc[:,features]
    
        assetCodes = df[self.assetId].unique()
        df_codes = df.groupby(self.assetId)
        df_codes = [df_code[1][['time', self.assetId]+self.return_features] for df_code in df_codes]
        pool = Pool(4)
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


class AssetEncoder():
    def __init__(self):
        self.assetId = 'assetCode'
        self.le = LabelEncoder()

    def fit(self, X):
        labels = X[self.assetId].values
        self.labels = np.append(labels, 'le_unknown')
        self.le.fit(self.labels)

    def transform(self, X):
        X[self.assetId] = X[self.assetId].apply(self._map_known_labels)
        X['assetCodeT'] = self.le.transform(X)
        return X

    def _map_known_labels(self,x):
        if x in self.labels:
            return x
        else:
            return 'le_unknown'

def build_nn(n_real=5, n_cat=5):
    
    #categorical
    cat_input = Input((1,), dtype='int16')
    emb = Embedding(n_cat,n_real)(cat_input)
    cat_logit = Flatten()(emb)

    #Real
    real_input = Input((n_real,))
    real_logit = BatchNormalization()(real_input)
    real_logit = Dense(256, activation='relu')(real_input)
    real_logit = Dense(128, activation='relu')(real_logit)


    #merged
    merged_logits = concatenate([cat_logit, real_logit])
    merged_logits = Dense(64, activation='relu')(merged_logits)
    merged_logits = Dense(32, activation='relu')(merged_logits)
    out = Dense(1, activation='sigmoid')(merged_logits)

    #compile
    model = Model(inputs=[real_input, cat_input], outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage


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

class NNDataGenerator():
    def __init__(self, labelEncoder, real_cols, assetId='assetCode'):
        self.labelEncoder = labelEncoder
        self.real_cols = real_cols
        self.assetId = assetId
        self.assetEnc = 'assetCodeT'
    def flow_from_dataset(self, X, y, batch_size=1000):
        end = len(X)//batch_size
        for i in range(len(X)//batch_size):
            if i == end-1:
                real = X.iloc[i*batch_size:,self.real_cols]
                label = self.labelEncoder.transform(X.loc[i*batch_size:, self.assetId])
                y = y[i*batch_size:]
            else:
                real = X.iloc[i*batch_size:(i+1)*batch_size,self.real_cols]
                label = self.labelEncoder.transform(i*batch_size:(i+1)*batch_size, self.assetId])
                y = y[i*batch_size:(i+1)*batch_size]




    def get_data(self, X):





if __name__ == '__main__':
    data = DataPrepper()
    featurizer = Featurizer()
    df = pd.read_pickle('../data/original_merged_train.pkl')
    df = df.loc[df.time>=20100101]
    target = df.pop('returnsOpenNextMktres10').values
    X = df
    X = featurizer.transform(X)
    X.to_pickle('../data/lag_features/X_all.pkl')
    np.save('../data/lag_features/target_all', target)


    X = pd.read_pickle('..data/lag_features/X_all.pkl')
    target = np.load('../data/lag_features/target_all.npy')
    drop_cols = ['assetCode','assetName','marketCommentary', 'time']
    X_features = [c for c in X.columns.values if c not in drop_cols]
    
    #scale
    X_scaler = StandardScaler()
    X_scaler.fit(X.loc[:,X_features])
    X.loc[:,X_features] = X_scaler.transform(X.loc[:,X_features])

    #encode assets
    assetencoder = AssetEncoder()
    assetencoder.fit(X)



    X_train, X_val, target_train, target_val = sk_train_test_split(X, target)

    #make binary y
    y_train = np.where(target_train>0, 1, 0).astype(int)
    y_val = np.where(target_val>0, 1, 0).astype(int)

    #make X with feature selected
    cat_cols = ['assetCode','assetCodeT','assetName','marketCommentary', 'time']
    real_cols = [c for c in X.columns.values if c not in cat_cols]
    real_train = X_train.loc[:,real_cols]
    real_val = X_val.loc[:,real_cols]
    cat_train = X_train.loc[:,'assetCodeT']
    cat_val = X_val.loc[:,'assetCodeT']


    n_real = len(real_cols)
    n_cat = len(X)

    model = build_nn(n_real, n_cat)
    print (model.summary())
    checkpoint = ModelCheckpoint('dense_nn.hdf5', verbose=True, save_best_only=True)
    earlystop = EarlyStopping(patience=5, verbose=True)

    history = model.fit(x=[real_train,cat_train],y=y_train,
             validation_data=([real_val,cat_val],y_val),
             epochs=10, verbose=True, callbacks=[earlystop, checkpoint])




