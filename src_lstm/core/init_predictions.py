from core.data_processor import DataHandler
from datetime import datetime, timedelta
import numpy as np 
import pandas as pd 
import matplotlib as plt 
from plot_helpers.classifier_performance_plothelper import plot_coefficient_importances
from plot_helpers.model_plots import plot_confusion_matrix, plot_roc
from plot_helpers.tree_plothelper import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from lightgbm import LGBMClassifier
import joblib

from keras.layers import Input, Dense, Activation, Dropout, LSTM, Embedding, Reshape, concatenate
from keras.models import Model


class InitialLogit():
    def __init__(self):
        self.model = LogisticRegression()
        self.scaler = StandardScaler()
    def fit(self, X_train, y_train):
        self.scaler = self.scaler.fit(X_train)
        self.model.fit(self.scaler.transform(X_train), y_train)

    def show_roc(self, X_test, y_test):
        X_test = self.scaler.transform(X_test)
        fig , ax = plt.subplots(figsize=(12,12))
        plot_roc(self.model, X_test, y_test, ax)
        ax.set_title("Logistic Regression on AAPL Stock")
    def dump_model(self):
        joblib.dump(self.model, 'init_aapl_logit.joblib')
    def show_coefficients(self, features, title):
        fig, ax = plt.subplot(figsize=(12,8))
        plot_coefficient_importances(self.model, features,  ax)
        plt.title(title)
        

class InitialGBM():
    # # https://www.kaggle.com/rabaman/0-64-in-100-lines
    def __init__(self):
        self.model = LGBMClassifier(num_leaves=60, max_depth=4, learning_rate=0.01, n_estimators=500,
                                min_child_samples=20, random_state=42, n_jobs=-1, bagging_fraction=0.9,
                                feature_fraction=0.9, bagging_freq=5, bagging_seed=2018)
    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train, eval_set=(X_val,y_val),
                        eval_metric=['auc', 'binary_logloss'], verbose=True)

    def show_roc(self, X_test, y_test):
        fig , ax = plt.subplots(figsize=(12,12))
        plot_roc(self.model, X_test, y_test, ax)
        ax.set_title("Initial Model Performance: Stock Prices and News")
    def dump_model(self):
        joblib.dump(self.model, 'init_lgb.joblib')
    def show_partial_dependence(self, data, labels):
        plot_partial_dependence_skater(self.model, data, labels)

class InitialRNN():
    def build_model(self):
        lstm_neurons = 300
        n_steps = 60
        n_features = 29
        len_vocab = 2
        len_embed = 300

        #real input is n_iterations, n_timesteps, n_features
        #cat input is n_iterations, n_timesteps, 1

        real_input = Input(shape=(n_steps, n_features,))
        cat_input = Input(shape=(n_steps,1), dtype='int16')
        emb = Embedding(len_vocab,len_embed)(cat_input)
        emb = Reshape((n_steps, len_embed))(emb)
        merged = concatenate([emb, real_input])
        rnn = LSTM(300, input_shape=(n_steps, len_embed+n_features),return_sequences=True)(merged)
        drop = Dropout(.2)(rnn)
        rnn = LSTM(300, input_shape=(n_steps, len_embed+n_features),return_sequences=True)(drop)
        rnn = LSTM(300, input_shape=(n_steps, len_embed+n_features),return_sequences=False)(rnn)
        drop = Dropout(.2)(rnn)
        dense = Dense(1, activation='sigmoid')(drop)
        M = Model(input=[real_input, cat_input], output=[dense])
        M.compile(loss='binary_crossentropy', optimizer='adam')

if __name__ == '__main__':
    le = LabelEncoder()
    data = DataHandler()
    df = pd.read_pickle('../data/init_train_data.pkl')
    df_aapl = df[df.assetName == 'Apple Inc'].reset_index()
    y_apl = np.where(df_aapl.pop('returnsOpenNextMktres10').values>0, 1, 0).astype(int)
    X_apl = df_aapl
    Xapl_train, Xapl_test, yapl_train, yapl_test = data.train_test_split(X_apl, y_apl, 20151231)
    Xapl_train.drop(columns=['assetName', 'time'], inplace=True)
    Xapl_test.drop(columns=['assetName', 'time'], inplace=True)


    #Single Stock Performance




