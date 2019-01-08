import os
import json
import time
import math
import pandas as pd
import matplotlib.pyplot as plt
from core.data_loader import DataLoader
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
import numpy as np

import tensorflow, tensorboard
from keras.callbacks import TensorBoard
from keras.layers import Input, Dense, Activation, Dropout, LSTM
from keras.models import Model
from keras.optimizers import Adam

from core.evaluator import ReturnsEvaluator

def build_model(lstm_neurons=300,n_steps=5, n_features=23):
    #real input is n_iterations, n_timesteps, n_features
    real_input = Input(shape=(n_steps, n_features,))
    rnn = LSTM(300, input_shape=(n_steps, n_features),
                    return_sequences=True, 
                    dropout=.2, 
                    recurrent_dropout=.2)(real_input)
    rnn = LSTM(300, input_shape=(n_steps, n_features),
                    return_sequences=True, 
                    dropout=.2, 
                    recurrent_dropout=.2)(rnn)
    rnn = LSTM(300, input_shape=(n_steps, n_features),
                    return_sequences=False, 
                    dropout=.2, 
                    recurrent_dropout=.2)(rnn)
    dense = Dense(300, activation='relu')(rnn)
    dense = Dense(1, activation='sigmoid')(dense)
    M = Model(inputs=[real_input], outputs=[dense])
    adam = Adam(lr=0.0001)
    M.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return M


def predict_looper_lstm(model, test_df, dataloader, seq_len=5, 
                   pred_start=20170101, pred_end=20181101, 
                   predcol='confidenceValue'):
    test_df = test_df.sort_values(['time', 'assetCode'])
    pred_days = test_df['time'].loc[(test_df['time']>=pred_start) & (test_df['time']<=pred_end)].unique()
    test_df[predcol] = 0
    for pred_day in pred_days:
        print(f'___________________________________')
        print(f'Getting new data and preparing it')
        assetCodes = test_df['assetCode'].loc[test_df.time == pred_day].values
        pred_df = test_df.loc[(test_df.time<=pred_day) & (test_df.time>=(pred_day - 10))]
        print(f'Got it. ready to make some predictions')
        print(f'Getting company windows for {pred_day}')
        X_test = dataloader.generate_test_batch(pred_df, assetCodes, 5)
        predictions = (model.predict(X_test) - 0.5) * 2
        print("Submitting predictions!")
        test_df[predcol].loc[test_df.time == pred_day] = predictions.flatten()
    return test_df



if __name__ == '__main__':



    train_df = pd.read_pickle('../data/init_train_data.pkl')
    test_df = pd.read_pickle('../data/test_data.pkl')
    test_df = test_df.sort_values(['time', 'assetCode'])
    test_df.to_pickle('../data/test_data.pkl')

    #Gotta have features. These are good ones
    cols = ["returnsOpenNextMktres10","returnsClosePrevRaw1",
               "returnsOpenPrevRaw1", "returnsClosePrevMktres1",
               "returnsOpenPrevMktres1","returnsClosePrevMktres10",
               "returnsOpenPrevMktres10", "dailychange","companyCount", "relevance",
               "sentimentNegative", "sentimentNeutral", "sentimentPositive",
               "noveltyCount12H", "noveltyCount24H", "noveltyCount3D",
               "noveltyCount5D", "noveltyCount7D", "volumeCounts12H",
               "volumeCounts24H", "volumeCounts3D", "volumeCounts5D",
               "volumeCounts7D", "coverage"]

    #Removing the target column for the prediction phase
    test_cols = ["returnsClosePrevRaw1",
               "returnsOpenPrevRaw1", "returnsClosePrevMktres1",
               "returnsOpenPrevMktres1","returnsClosePrevMktres10",
               "returnsOpenPrevMktres10", "dailychange","companyCount", "relevance",
               "sentimentNegative", "sentimentNeutral", "sentimentPositive",
               "noveltyCount12H", "noveltyCount24H", "noveltyCount3D",
               "noveltyCount5D", "noveltyCount7D", "volumeCounts12H",
               "volumeCounts24H", "volumeCounts3D", "volumeCounts5D",
               "volumeCounts7D", "coverage"]

    loader = DataLoader(cols, test_cols)

    X, y = loader.get_train_data(train_df, assetName='Apple Inc', seq_len=5, normalize=True)

    model = build_model()
    model.fit(X, y, epochs=5, batch_size=50)

    pred_df = predict_looper_lstm(model, test_df, loader)

    evaluator = ReturnsEvaluator()

    print("kaggle mean variance", evaluator.get_kaggle_mean_variance(pred_df))

    metrics_dict = evaluator.get_returns(pred_df)


