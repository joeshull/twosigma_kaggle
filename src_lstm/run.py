__author__ = "Jakob Aungiers"
__copyright__ = "Jakob Aungiers 2018"
__version__ = "2.0.0"
__license__ = "MIT"

import os
import json
import time
import math
import matplotlib.pyplot as plt
from core.data_processor import DataLoader
from core.model import Model
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib as mpl
import numpy as np

from keras.layers import Input, Dense, Activation, Dropout, LSTM
from keras.models import Model
from keras.optimizers import Adam

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()

def plot_roc(probs, y_test,ax):
    fpr, tpr, thresholds = roc_curve(y_test,probs)
    auc_score = round(roc_auc_score(y_test, probs),4)
    ax.plot(fpr, tpr, label=f'Initial LSTM = {auc_score} AUC')
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
         label='Luck')
    ax.set_xlabel("False Positive Rate (1-Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity, Recall)")
    ax.set_title("ROC/AUC: AAPL - Trained AAPL Only - 5day sequence_length")
    ax.legend()

def build_model():
    lstm_neurons = 300
    n_steps = 5
    n_features = 24
    len_vocab = 3
    len_embed = 300

    #real input is n_iterations, n_timesteps, n_features
    #cat input is n_iterations, n_timesteps, 1

    real_input = Input(shape=(n_steps, n_features,))
    cat_input = Input(shape=(n_steps,1), dtype='int16')
    emb = Embedding(len_vocab,len_embed)(cat_input)
    emb = Reshape((n_steps, len_embed))(emb)
    merged = concatenate([emb, real_input])
    rnn = LSTM(300, input_shape=(n_steps, len_embed+n_features),return_sequences=True)(merged)
    drop = Dropout(.3)(rnn)
    rnn = LSTM(300, input_shape=(n_steps, len_embed+n_features),return_sequences=True)(drop)
    rnn = LSTM(300, input_shape=(n_steps, len_embed+n_features),return_sequences=False)(rnn)
    drop = Dropout(.3)(rnn)
    dense = Dense(300, activation='relu')(drop)
    dense = Dense(1, activation='sigmoid')(dense)
    M = Model(input=[real_input, cat_input], output=[dense])
    adam = Adam(lr=0.0001)
    M.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return M



if __name__ == '__main__':

    configs = json.load(open('config.json', 'r'))
    # if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )



    # #Get Embedded X,y for each company
    # #Configs
    config_aapl = json.load(open('config_aapl.json', 'r'))
    config_advance = json.load(open('config_advance.json', 'r'))
    config_allstate = json.load(open('config_allstate.json', 'r'))

    data_aapl = DataLoader(
    os.path.join('data', config_aapl['data']['filename']),
    config_aapl['data']['train_test_split'],
    config_aapl['data']['columns']
    )

    data_adv = DataLoader(
    os.path.join('data', config_advance['data']['filename']),
    config_advance['data']['train_test_split'],
    config_advance['data']['columns']
    )

    data_alls = DataLoader(
    os.path.join('data', config_allstate['data']['filename']),
    config_allstate['data']['train_test_split'],
    config_allstate['data']['columns']
    )
    #AAPL Data
    xapl, yapl = data_aapl.get_train_data(
        seq_len=config_aapl['data']['sequence_length'],
        normalise=config_aapl['data']['normalise']
    )

    Xapl = [xapl,np.zeros((xapl.shape[0],xapl.shape[1],1))]

    #Advance Data
    xadv, yadv = data_adv.get_train_data(
        seq_len=config_advance['data']['sequence_length'],
        normalise=config_advance['data']['normalise']
    )

    Xadv = [xadv,np.ones((xadv.shape[0],xadv.shape[1],1))]

    #Allstate Data
    xalls, yalls = data_alls.get_train_data(
        seq_len=config_allstate['data']['sequence_length'],
        normalise=config_allstate['data']['normalise']
    )
    allemb = np.ones((xalls.shape[0],xalls.shape[1],1))+1
    Xalls = [xadv,allemb]

    # #Test With Embedding
    # #AAPL
    x_test_apl, y_test_apl = data_aapl.get_test_data(
        seq_len=config_aapl['data']['sequence_length'],
        normalise=config_aapl['data']['normalise']
    )
    em_apl = np.zeros((x_test_apl.shape[0], x_test_apl.shape[1],1))
    X_test_apl = [x_test_apl, em_apl]

    #Advance
    x_test_adv, y_test_adv = data_adv.get_test_data(
        seq_len=config_advance['data']['sequence_length'],
        normalise=config_advance['data']['normalise']
    )
    em_adv = np.ones((x_test_adv.shape[0], x_test_adv.shape[1],1))
    X_test_adv = [x_test_adv, em_adv]

    # #Allstate
    x_test_alls, y_test_alls = data_alls.get_test_data(
        seq_len=config_allstate['data']['sequence_length'],
        normalise=config_allstate['data']['normalise']
    )
    em_alls = np.ones((x_test_alls.shape[0], x_test_alls.shape[1],1)) + 1
    X_test_alls = [x_test_alls, em_alls]

 
    #Build Model for Embedding
    model = build_model()

    model.fit(Xapl, yapl, epochs=5, batch_size=50, validation_data=[X_test_apl, y_test_apl])
    # for X,y in zip([Xapl, Xadv, Xalls],[yapl, yadv, yalls]):
    #     model.fit(X,y, 
    #         epochs=config_allstate['training']['epochs'],
    #         batch_size=config_allstate['training']['batch_size'],
    #         validation_data=(X_test_alls, y_test_alls))


    ##NO EMBED
    # # out-of memory generative training
    # data = DataLoader(
    # os.path.join('data', configs['data']['filename']),
    # configs['data']['train_test_split'],
    # configs['data']['columns']
    # )
    # model = Model()
    # model.build_model(configs)
    # steps_per_epoch = math.ceil((data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
    # model.train_generator(
    #     data_gen=data.generate_train_batch(
    #         seq_len=configs['data']['sequence_length'],
    #         batch_size=configs['training']['batch_size'],
    #         normalise=configs['data']['normalise']
    #     ),
    #     epochs=configs['training']['epochs'],
    #     batch_size=configs['training']['batch_size'],
    #     steps_per_epoch=steps_per_epoch,
    #     save_dir=configs['model']['save_dir']
    # )

    # x_test, y_test = data.get_test_data(
    #     seq_len=configs['data']['sequence_length'],
    #     normalise=configs['data']['normalise']
    # )
    #no embedding
    # predictions = model.predict_point_by_point(x_test)




    predictions = model.predict(X_test_apl)
    predictions = np.reshape(predictions, (predictions.size,))

    fig, ax = plt.subplots(figsize=(12,12))
    plot_roc(predictions, y_test_apl, ax)
    plt.show()