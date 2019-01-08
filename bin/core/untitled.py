from keras.layers import Input, Flatten, Dense, Activation, Dropout, LSTM, Embedding, Reshape, concatenate, merge
from keras.models import Model
import numpy as np
a = Input((1,), dtype='int16')
b = Input((1,))
emb = Embedding(4,20)(a)
emb = Flatten()(emb)
merged = concatenate([emb, b])
out = Dense(1, activation='linear')(merged)
M = Model(input=[a,b], output=[out])
M.compile(loss='mse', optimizer='SGD')
testA = np.random.randint(0,4,(6,1))
testB = np.random.random((6,1))
testY = np.random.random((6,1))
M.evaluate([testA, testB], testY)