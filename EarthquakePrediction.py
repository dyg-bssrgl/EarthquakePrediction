from tensorflow import keras
import tensorflow as tf
from keras.utils import plot_model
from keras import layers
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold
from keras.layers import SimpleRNN, LeakyReLU, PReLU, Input, LSTM, GRU, Conv1D, Dense, concatenate, Activation, Flatten, Lambda, GlobalAveragePooling1D, Dropout, Embedding, BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
from keras import backend as K

matplotlib.use('TkAgg')

# Model parametreleri
sequence_length = 5
features = 5  # Özellik sayısı, veri setinize göre ayarlayın

# Eğitim verilerini yükleme
df_train = pd.read_csv('train.csv')
df_train['tarih-saat'] = pd.to_datetime(df_train['tarih-saat'], format='%d-%m-%Y %H:%M:%S').astype(np.int64)

# Test verilerini yükleme
df_test = pd.read_csv('test.csv')
df_test['tarih-saat'] = pd.to_datetime(df_test['tarih-saat'], format='%d-%m-%Y %H:%M:%S').astype(np.int64)

# Verileri ölçeklendirme
scaler = MinMaxScaler()
df_train_scaled = scaler.fit_transform(df_train)
df_test_scaled = scaler.transform(df_test)

# Sıralı veri oluşturma fonksiyonu
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        label = data[i + sequence_length]
        X.append(seq)
        y.append(label)
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(df_train_scaled, sequence_length)
X_test, y_test = create_sequences(df_test_scaled, sequence_length)

# # Model yapısını oluşturma
# input_layer = Input(shape=(sequence_length, features))
#
# # input katmanları
# simpleRNN_t = SimpleRNN(1024, return_sequences=True)(input_layer)
# simpleRNN_act = Activation(activation='softplus')(simpleRNN_t)
# simpleRNN_b = BatchNormalization()(simpleRNN_act)
# simpleRNN4 = Dropout(rate=0.4)(simpleRNN_b)
#
# simpleRNN_t2 = SimpleRNN(512, return_sequences=True)(input_layer)
# simpleRNN2 = Activation('softplus')(simpleRNN_t2)
#
# lstm4_t = LSTM(4096, return_sequences=True)(input_layer)
# lstm4_a = LeakyReLU()(lstm4_t)
# lstm4_b = BatchNormalization()(lstm4_a)
# lstm4 = Dropout(rate=0.25)(lstm4_b)
#
# lstm2_t = LSTM(2048, return_sequences=True)(input_layer)
# lstm2 = Activation('softplus')(lstm2_t)
#
# gru4_t = GRU(4096, return_sequences=True)(input_layer)
# gru4_a = Activation('relu')(gru4_t)
# gru4_b = BatchNormalization()(gru4_a)
# gru4 = Dropout(rate=0.3)(gru4_b)
#
# gru2_t = GRU(2048, return_sequences=True)(input_layer)
# gru2 = PReLU()(gru2_t)
#
# conv4_t = Conv1D(4096, kernel_size=3, padding='same')(input_layer)
# conv4_a = Activation('softplus')(conv4_t)
# conv4_b = BatchNormalization()(conv4_a)
# conv4 = Dropout(rate=0.5)(conv4_b)
#
# conv2_t = Conv1D(2048, kernel_size=3, padding='same')(input_layer)
# conv2 = Activation('softplus')(conv2_t)
#
# # LSTMArm ################################################################
# # L2 Arm
# # İlk dal
# lstm_arm_1_1_lstm_t = LSTM(2048, return_sequences=True)(lstm2)
# lstm_arm_1_1_lstm = PReLU()(lstm_arm_1_1_lstm_t)
#
# lstm_arm_1_2_lstm_t = LSTM(2048, return_sequences=True)(lstm2)
# lstm_arm_1_2_lstm = PReLU()(lstm_arm_1_2_lstm_t)
#
# lstm_arm_1_3_gru_t = GRU(2048, return_sequences=True)(lstm2)
# lstm_arm_1_3_gru = PReLU()(lstm_arm_1_3_gru_t)
#
# lstm_arm_1_4_conv_t = Conv1D(2048, kernel_size=3, padding='same')(lstm2)
# lstm_arm_1_4_conv = PReLU()(lstm_arm_1_4_conv_t)
# # İkinci dal
# lstm_arm_2_1_lstm_t = LSTM(1024, return_sequences=True)(lstm_arm_1_1_lstm)
# lstm_arm_2_1_lstm = PReLU()(lstm_arm_2_1_lstm_t)
# lstm_arm_2_2_gru_t = GRU(1024, return_sequences=True)(lstm_arm_1_1_lstm)
# lstm_arm_2_2_gru = PReLU()(lstm_arm_2_2_gru_t)
#
# lstm_arm_2_3_lstm_t = LSTM(1024, return_sequences=True)(lstm_arm_1_2_lstm)
# lstm_arm_2_3_lstm = PReLU()(lstm_arm_2_3_lstm_t)
# lstm_arm_2_4_lstm_t = LSTM(1024, return_sequences=True)(lstm_arm_1_2_lstm)
# lstm_arm_2_4_lstm = PReLU()(lstm_arm_2_4_lstm_t)
#
# lstm_arm_2_5_lstm_t = LSTM(1024, return_sequences=True)(lstm_arm_1_3_gru)
# lstm_arm_2_5_lstm = PReLU()(lstm_arm_2_5_lstm_t)
# lstm_arm_2_6_conv_t = Conv1D(1024, kernel_size=3, padding='same')(lstm_arm_1_3_gru)
# lstm_arm_2_6_conv = PReLU()(lstm_arm_2_6_conv_t)
#
# lstm_arm_2_7_gru_t = GRU(1024, return_sequences=True)(lstm_arm_1_4_conv)
# lstm_arm_2_7_gru = PReLU()(lstm_arm_2_7_gru_t)
# lstm_arm_2_8_lstm_t = LSTM(1024, return_sequences=True)(lstm_arm_1_4_conv)
# lstm_arm_2_8_lstm = PReLU()(lstm_arm_2_8_lstm_t)
# # Üçüncü dal
# lstm_arm_3_1_lstm_t = LSTM(512, return_sequences=True)(lstm_arm_2_1_lstm)
# lstm_arm_3_1_lstm = LeakyReLU()(lstm_arm_3_1_lstm_t)
# lstm_arm_3_2_lstm_t = LSTM(512, return_sequences=True)(lstm_arm_2_1_lstm)
# lstm_arm_3_2_lstm = LeakyReLU()(lstm_arm_3_2_lstm_t)
#
# lstm_arm_3_3_conv_t = Conv1D(512, kernel_size=3, padding='same')(lstm_arm_2_2_gru)
# lstm_arm_3_3_conv = LeakyReLU()(lstm_arm_3_3_conv_t)
# lstm_arm_3_4_lstm_t = LSTM(512, return_sequences=True)(lstm_arm_2_2_gru)
# lstm_arm_3_4_lstm = LeakyReLU()(lstm_arm_3_4_lstm_t)
#
# lstm_arm_3_5_gru_t = GRU(512, return_sequences=True)(lstm_arm_2_3_lstm)
# lstm_arm_3_5_gru = LeakyReLU()(lstm_arm_3_5_gru_t)
# lstm_arm_3_6_conv_t = Conv1D(512, kernel_size=3, padding='same')(lstm_arm_2_3_lstm)
# lstm_arm_3_6_conv = LeakyReLU()(lstm_arm_3_6_conv_t)
#
# lstm_arm_3_7_gru_t = GRU(512, return_sequences=True)(lstm_arm_2_4_lstm)
# lstm_arm_3_7_gru = LeakyReLU()(lstm_arm_3_7_gru_t)
# lstm_arm_3_8_conv_t = Conv1D(512, kernel_size=3, padding='same')(lstm_arm_2_4_lstm)
# lstm_arm_3_8_conv = LeakyReLU()(lstm_arm_3_8_conv_t)
#
# lstm_arm_3_9_lstm_t = LSTM(512, return_sequences=True)(lstm_arm_2_5_lstm)
# lstm_arm_3_9_lstm = LeakyReLU()(lstm_arm_3_9_lstm_t)
# lstm_arm_3_10_lstm_t = LSTM(512, return_sequences=True)(lstm_arm_2_5_lstm)
# lstm_arm_3_10_lstm = LeakyReLU()(lstm_arm_3_10_lstm_t)
#
# lstm_arm_3_11_lstm_t = LSTM(512, return_sequences=True)(lstm_arm_2_6_conv)
# lstm_arm_3_11_lstm = LeakyReLU()(lstm_arm_3_11_lstm_t)
# lstm_arm_3_12_lstm_t = LSTM(512, return_sequences=True)(lstm_arm_2_6_conv)
# lstm_arm_3_12_lstm = LeakyReLU()(lstm_arm_3_12_lstm_t)
#
# lstm_arm_3_13_lstm_t = LSTM(512, return_sequences=True)(lstm_arm_2_7_gru)
# lstm_arm_3_13_lstm = LeakyReLU()(lstm_arm_3_13_lstm_t)
# lstm_arm_3_14_lstm_t = LSTM(512, return_sequences=True)(lstm_arm_2_7_gru)
# lstm_arm_3_14_lstm = LeakyReLU()(lstm_arm_3_14_lstm_t)
#
# lstm_arm_3_15_lstm_t = LSTM(512, return_sequences=True)(lstm_arm_2_8_lstm)
# lstm_arm_3_15_lstm = LeakyReLU()(lstm_arm_3_15_lstm_t)
# lstm_arm_3_16_gru_t = GRU(512, return_sequences=True)(lstm_arm_2_8_lstm)
# lstm_arm_3_16_gru = LeakyReLU()(lstm_arm_3_16_gru_t)
# # L2 çıkışları
# concatenate_L2_1_4 = concatenate([lstm_arm_3_1_lstm, lstm_arm_3_2_lstm, lstm_arm_3_3_conv, lstm_arm_3_4_lstm])
# concatenate_L2_1_4_d = Dense(256)(concatenate_L2_1_4)
# concatenate_L2_5_8 = concatenate([lstm_arm_3_5_gru, lstm_arm_3_6_conv, lstm_arm_3_7_gru, lstm_arm_3_8_conv])
# concatenate_L2_5_8_d = Dense(256)(concatenate_L2_5_8)
# concatenate_L2_9_12 = concatenate([lstm_arm_3_9_lstm, lstm_arm_3_10_lstm, lstm_arm_3_11_lstm, lstm_arm_3_12_lstm])
# concatenate_L2_9_12_d = Dense(256)(concatenate_L2_9_12)
# concatenate_L2_13_16 = concatenate([lstm_arm_3_13_lstm, lstm_arm_3_14_lstm, lstm_arm_3_15_lstm, lstm_arm_3_16_gru])
# concatenate_L2_13_16_d = Dense(256)(concatenate_L2_13_16)
#
# concatenate_L2_1_8 = concatenate([concatenate_L2_1_4_d, concatenate_L2_5_8_d])
# concatenate_L2_1_8_d = Dense(256)(concatenate_L2_1_8)
# concatenate_L2_9_16 = concatenate([concatenate_L2_9_12_d, concatenate_L2_13_16_d])
# concatenate_L2_9_16_d = Dense(256)(concatenate_L2_9_16)
#
# concatenate_L2_pre = concatenate([concatenate_L2_1_8_d, concatenate_L2_9_16_d])
# concatenate_L2 = Dense(256)(concatenate_L2_pre)
#
# # L4 Arm
# # İlk dal
# lstm_arm_1_1_lstm_t4 = LSTM(2048, return_sequences=True)(lstm4)
# lstm_arm_1_1_lstm4 = PReLU()(lstm_arm_1_1_lstm_t4)
#
# lstm_arm_1_2_lstm_t4 = LSTM(2048, return_sequences=True)(lstm4)
# lstm_arm_1_2_lstm4 = PReLU()(lstm_arm_1_2_lstm_t4)
#
# lstm_arm_1_3_gru_t4 = GRU(2048, return_sequences=True)(lstm4)
# lstm_arm_1_3_gru4 = PReLU()(lstm_arm_1_3_gru_t4)
#
# lstm_arm_1_4_conv_t4 = Conv1D(2048, kernel_size=3, padding='same')(lstm4)
# lstm_arm_1_4_conv4 = PReLU()(lstm_arm_1_4_conv_t4)
# # İkinci dal
# lstm_arm_2_1_lstm_t4 = LSTM(1024, return_sequences=True)(lstm_arm_1_1_lstm4)
# lstm_arm_2_1_lstm4 = PReLU()(lstm_arm_2_1_lstm_t4)
# lstm_arm_2_2_gru_t4 = GRU(1024, return_sequences=True)(lstm_arm_1_1_lstm4)
# lstm_arm_2_2_gru4 = PReLU()(lstm_arm_2_2_gru_t4)
#
# lstm_arm_2_3_lstm_t4 = LSTM(1024, return_sequences=True)(lstm_arm_1_2_lstm4)
# lstm_arm_2_3_lstm4 = PReLU()(lstm_arm_2_3_lstm_t4)
# lstm_arm_2_4_lstm_t4 = LSTM(1024, return_sequences=True)(lstm_arm_1_2_lstm4)
# lstm_arm_2_4_lstm4 = PReLU()(lstm_arm_2_4_lstm_t4)
#
# lstm_arm_2_5_lstm_t4 = LSTM(1024, return_sequences=True)(lstm_arm_1_3_gru4)
# lstm_arm_2_5_lstm4 = PReLU()(lstm_arm_2_5_lstm_t4)
# lstm_arm_2_6_conv_t4 = Conv1D(1024, kernel_size=3, padding='same')(lstm_arm_1_3_gru4)
# lstm_arm_2_6_conv4 = PReLU()(lstm_arm_2_6_conv_t4)
#
# lstm_arm_2_7_gru_t4 = GRU(1024, return_sequences=True)(lstm_arm_1_4_conv4)
# lstm_arm_2_7_gru4 = PReLU()(lstm_arm_2_7_gru_t4)
# lstm_arm_2_8_lstm_t4 = LSTM(1024, return_sequences=True)(lstm_arm_1_4_conv4)
# lstm_arm_2_8_lstm4 = PReLU()(lstm_arm_2_8_lstm_t4)
# # Üçüncü dal
# lstm_arm_3_1_lstm_t4 = LSTM(512, return_sequences=True)(lstm_arm_2_1_lstm4)
# lstm_arm_3_1_lstm4 = LeakyReLU()(lstm_arm_3_1_lstm_t4)
# lstm_arm_3_2_lstm_t4 = LSTM(512, return_sequences=True)(lstm_arm_2_1_lstm4)
# lstm_arm_3_2_lstm4 = LeakyReLU()(lstm_arm_3_2_lstm_t4)
#
# lstm_arm_3_3_conv_t4 = Conv1D(512, kernel_size=3, padding='same')(lstm_arm_2_2_gru4)
# lstm_arm_3_3_conv4 = LeakyReLU()(lstm_arm_3_3_conv_t4)
# lstm_arm_3_4_lstm_t4 = LSTM(512, return_sequences=True)(lstm_arm_2_2_gru4)
# lstm_arm_3_4_lstm4 = LeakyReLU()(lstm_arm_3_4_lstm_t4)
#
# lstm_arm_3_5_gru_t4 = GRU(512, return_sequences=True)(lstm_arm_2_3_lstm4)
# lstm_arm_3_5_gru4 = LeakyReLU()(lstm_arm_3_5_gru_t4)
# lstm_arm_3_6_conv_t4 = Conv1D(512, kernel_size=3, padding='same')(lstm_arm_2_3_lstm4)
# lstm_arm_3_6_conv4 = LeakyReLU()(lstm_arm_3_6_conv_t4)
#
# lstm_arm_3_7_gru_t4 = GRU(512, return_sequences=True)(lstm_arm_2_4_lstm4)
# lstm_arm_3_7_gru4 = LeakyReLU()(lstm_arm_3_7_gru_t4)
# lstm_arm_3_8_conv_t4 = Conv1D(512, kernel_size=3, padding='same')(lstm_arm_2_4_lstm4)
# lstm_arm_3_8_conv4 = LeakyReLU()(lstm_arm_3_8_conv_t4)
#
# lstm_arm_3_9_lstm_t4 = LSTM(512, return_sequences=True)(lstm_arm_2_5_lstm4)
# lstm_arm_3_9_lstm4 = LeakyReLU()(lstm_arm_3_9_lstm_t4)
# lstm_arm_3_10_lstm_t4 = LSTM(512, return_sequences=True)(lstm_arm_2_5_lstm4)
# lstm_arm_3_10_lstm4 = LeakyReLU()(lstm_arm_3_10_lstm_t4)
#
# lstm_arm_3_11_lstm_t4 = LSTM(512, return_sequences=True)(lstm_arm_2_6_conv4)
# lstm_arm_3_11_lstm4= LeakyReLU()(lstm_arm_3_11_lstm_t4)
# lstm_arm_3_12_lstm_t4 = LSTM(512, return_sequences=True)(lstm_arm_2_6_conv4)
# lstm_arm_3_12_lstm4 = LeakyReLU()(lstm_arm_3_12_lstm_t4)
#
# lstm_arm_3_13_lstm_t4 = LSTM(512, return_sequences=True)(lstm_arm_2_7_gru4)
# lstm_arm_3_13_lstm4 = LeakyReLU()(lstm_arm_3_13_lstm_t4)
# lstm_arm_3_14_lstm_t4 = LSTM(512, return_sequences=True)(lstm_arm_2_7_gru4)
# lstm_arm_3_14_lstm4 = LeakyReLU()(lstm_arm_3_14_lstm_t4)
#
# lstm_arm_3_15_lstm_t4 = LSTM(512, return_sequences=True)(lstm_arm_2_8_lstm4)
# lstm_arm_3_15_lstm4 = LeakyReLU()(lstm_arm_3_15_lstm_t4)
# lstm_arm_3_16_gru_t4 = GRU(512, return_sequences=True)(lstm_arm_2_8_lstm4)
# lstm_arm_3_16_gru4 = LeakyReLU()(lstm_arm_3_16_gru_t4)
# # L4 çıkışları
# concatenate_L4_1_4 = concatenate([lstm_arm_3_1_lstm4, lstm_arm_3_2_lstm4, lstm_arm_3_3_conv4, lstm_arm_3_4_lstm4])
# concatenate_L4_1_4_d = Dense(256)(concatenate_L4_1_4)
# concatenate_L4_5_8 = concatenate([lstm_arm_3_5_gru4, lstm_arm_3_6_conv4, lstm_arm_3_7_gru4, lstm_arm_3_8_conv4])
# concatenate_L4_5_8_d = Dense(256)(concatenate_L4_5_8)
# concatenate_L4_9_12 = concatenate([lstm_arm_3_9_lstm4, lstm_arm_3_10_lstm4, lstm_arm_3_11_lstm4, lstm_arm_3_12_lstm4])
# concatenate_L4_9_12_d = Dense(256)(concatenate_L4_9_12)
# concatenate_L4_13_16 = concatenate([lstm_arm_3_13_lstm4, lstm_arm_3_14_lstm4, lstm_arm_3_15_lstm4, lstm_arm_3_16_gru4])
# concatenate_L4_13_16_d = Dense(256)(concatenate_L4_13_16)
#
# concatenate_L4_1_8 = concatenate([concatenate_L4_1_4_d, concatenate_L4_5_8_d])
# concatenate_L4_1_8_d = Dense(256)(concatenate_L4_1_8)
# concatenate_L4_9_16 = concatenate([concatenate_L4_9_12_d, concatenate_L4_13_16_d])
# concatenate_L4_9_16_d = Dense(256)(concatenate_L4_9_16)
#
# concatenate_L4_pre = concatenate([concatenate_L4_1_8_d, concatenate_L4_9_16_d])
# concatenate_L4 = Dense(256)(concatenate_L4_pre)
#
#
# # GRUArm ################################################################
# # G2 Arm
# # İlk dal
# gru_arm_1_1_lstm_t = LSTM(2048, return_sequences=True)(gru2)
# gru_arm_1_1_lstm = PReLU()(gru_arm_1_1_lstm_t)
#
# gru_arm_1_2_lstm_t = LSTM(2048, return_sequences=True)(gru2)
# gru_arm_1_2_lstm = PReLU()(gru_arm_1_2_lstm_t)
#
# gru_arm_1_3_gru_t = GRU(2048, return_sequences=True)(gru2)
# gru_arm_1_3_gru = PReLU()(gru_arm_1_3_gru_t)
#
# gru_arm_1_4_conv_t = Conv1D(2048, kernel_size=3, padding='same')(gru2)
# gru_arm_1_4_conv = PReLU()(gru_arm_1_4_conv_t)
# # İkinci dal
# gru_arm_2_1_lstm_t = LSTM(1024, return_sequences=True)(gru_arm_1_1_lstm)
# gru_arm_2_1_lstm = PReLU()(gru_arm_2_1_lstm_t)
# gru_arm_2_2_gru_t = GRU(1024, return_sequences=True)(gru_arm_1_1_lstm)
# gru_arm_2_2_gru = PReLU()(gru_arm_2_2_gru_t)
#
# gru_arm_2_3_lstm_t = LSTM(1024, return_sequences=True)(gru_arm_1_2_lstm)
# gru_arm_2_3_lstm = PReLU()(gru_arm_2_3_lstm_t)
# gru_arm_2_4_lstm_t = LSTM(1024, return_sequences=True)(gru_arm_1_2_lstm)
# gru_arm_2_4_lstm = PReLU()(gru_arm_2_4_lstm_t)
#
# gru_arm_2_5_lstm_t = LSTM(1024, return_sequences=True)(gru_arm_1_3_gru)
# gru_arm_2_5_lstm = PReLU()(gru_arm_2_5_lstm_t)
# gru_arm_2_6_conv_t = Conv1D(1024, kernel_size=3, padding='same')(gru_arm_1_3_gru)
# gru_arm_2_6_conv = PReLU()(gru_arm_2_6_conv_t)
#
# gru_arm_2_7_gru_t = GRU(1024, return_sequences=True)(gru_arm_1_4_conv)
# gru_arm_2_7_gru = PReLU()(gru_arm_2_7_gru_t)
# gru_arm_2_8_lstm_t = LSTM(1024, return_sequences=True)(gru_arm_1_4_conv)
# gru_arm_2_8_lstm = PReLU()(gru_arm_2_8_lstm_t)
# # Üçüncü dal
# gru_arm_3_1_lstm_t = LSTM(512, return_sequences=True)(gru_arm_2_1_lstm)
# gru_arm_3_1_lstm = LeakyReLU()(gru_arm_3_1_lstm_t)
# gru_arm_3_2_lstm_t = LSTM(512, return_sequences=True)(gru_arm_2_1_lstm)
# gru_arm_3_2_lstm = LeakyReLU()(gru_arm_3_2_lstm_t)
#
# gru_arm_3_3_conv_t = Conv1D(512, kernel_size=3, padding='same')(gru_arm_2_2_gru)
# gru_arm_3_3_conv = LeakyReLU()(gru_arm_3_3_conv_t)
# gru_arm_3_4_lstm_t = LSTM(512, return_sequences=True)(gru_arm_2_2_gru)
# gru_arm_3_4_lstm = LeakyReLU()(gru_arm_3_4_lstm_t)
#
# gru_arm_3_5_gru_t = GRU(512, return_sequences=True)(gru_arm_2_3_lstm)
# gru_arm_3_5_gru = LeakyReLU()(gru_arm_3_5_gru_t)
# gru_arm_3_6_conv_t = Conv1D(512, kernel_size=3, padding='same')(gru_arm_2_3_lstm)
# gru_arm_3_6_conv = LeakyReLU()(gru_arm_3_6_conv_t)
#
# gru_arm_3_7_gru_t = GRU(512, return_sequences=True)(gru_arm_2_4_lstm)
# gru_arm_3_7_gru = LeakyReLU()(gru_arm_3_7_gru_t)
# gru_arm_3_8_conv_t = Conv1D(512, kernel_size=3, padding='same')(gru_arm_2_4_lstm)
# gru_arm_3_8_conv = LeakyReLU()(gru_arm_3_8_conv_t)
#
# gru_arm_3_9_lstm_t = LSTM(512, return_sequences=True)(gru_arm_2_5_lstm)
# gru_arm_3_9_lstm = LeakyReLU()(gru_arm_3_9_lstm_t)
# gru_arm_3_10_lstm_t = LSTM(512, return_sequences=True)(gru_arm_2_5_lstm)
# gru_arm_3_10_lstm = LeakyReLU()(gru_arm_3_10_lstm_t)
#
# gru_arm_3_11_lstm_t = LSTM(512, return_sequences=True)(gru_arm_2_6_conv)
# gru_arm_3_11_lstm = LeakyReLU()(gru_arm_3_11_lstm_t)
# gru_arm_3_12_lstm_t = LSTM(512, return_sequences=True)(gru_arm_2_6_conv)
# gru_arm_3_12_lstm = LeakyReLU()(gru_arm_3_12_lstm_t)
#
# gru_arm_3_13_lstm_t = LSTM(512, return_sequences=True)(gru_arm_2_7_gru)
# gru_arm_3_13_lstm = LeakyReLU()(gru_arm_3_13_lstm_t)
# gru_arm_3_14_lstm_t = LSTM(512, return_sequences=True)(gru_arm_2_7_gru)
# gru_arm_3_14_lstm = LeakyReLU()(gru_arm_3_14_lstm_t)
#
# gru_arm_3_15_lstm_t = LSTM(512, return_sequences=True)(gru_arm_2_8_lstm)
# gru_arm_3_15_lstm = LeakyReLU()(gru_arm_3_15_lstm_t)
# gru_arm_3_16_gru_t = GRU(512, return_sequences=True)(gru_arm_2_8_lstm)
# gru_arm_3_16_gru = LeakyReLU()(gru_arm_3_16_gru_t)
# # G2 çıkışları
# concatenate_G2_1_4 = concatenate([gru_arm_3_1_lstm, gru_arm_3_2_lstm, gru_arm_3_3_conv, gru_arm_3_4_lstm])
# concatenate_G2_1_4_d = Dense(256)(concatenate_G2_1_4)
# concatenate_G2_5_8 = concatenate([gru_arm_3_5_gru, gru_arm_3_6_conv, gru_arm_3_7_gru, gru_arm_3_8_conv])
# concatenate_G2_5_8_d = Dense(256)(concatenate_G2_5_8)
# concatenate_G2_9_12 = concatenate([gru_arm_3_9_lstm, gru_arm_3_10_lstm, gru_arm_3_11_lstm, gru_arm_3_12_lstm])
# concatenate_G2_9_12_d = Dense(256)(concatenate_G2_9_12)
# concatenate_G2_13_16 = concatenate([gru_arm_3_13_lstm, gru_arm_3_14_lstm, gru_arm_3_15_lstm, gru_arm_3_16_gru])
# concatenate_G2_13_16_d = Dense(256)(concatenate_G2_13_16)
#
# concatenate_G2_1_8 = concatenate([concatenate_G2_1_4_d, concatenate_G2_5_8_d])
# concatenate_G2_1_8_d = Dense(256)(concatenate_G2_1_8)
# concatenate_G2_9_16 = concatenate([concatenate_G2_9_12_d, concatenate_G2_13_16_d])
# concatenate_G2_9_16_d = Dense(256)(concatenate_G2_1_8)
#
# concatenate_G2_pre = concatenate([concatenate_G2_1_8_d, concatenate_G2_9_16_d])
# concatenate_G2 = Dense(256)(concatenate_G2_pre)
#
# # G4 Arm
# # İlk dal
# gru_arm_1_1_lstm_t4 = LSTM(2048, return_sequences=True)(gru4)
# gru_arm_1_1_lstm4 = PReLU()(gru_arm_1_1_lstm_t4)
#
# gru_arm_1_2_lstm_t4 = LSTM(2048, return_sequences=True)(gru4)
# gru_arm_1_2_lstm4 = PReLU()(gru_arm_1_2_lstm_t4)
#
# gru_arm_1_3_gru_t4 = GRU(2048, return_sequences=True)(gru4)
# gru_arm_1_3_gru4 = PReLU()(gru_arm_1_3_gru_t4)
#
# gru_arm_1_4_conv_t4 = Conv1D(2048, kernel_size=3, padding='same')(gru4)
# gru_arm_1_4_conv4 = PReLU()(gru_arm_1_4_conv_t4)
# # İkinci dal
# gru_arm_2_1_lstm_t4 = LSTM(1024, return_sequences=True)(gru_arm_1_1_lstm4)
# gru_arm_2_1_lstm4 = PReLU()(gru_arm_2_1_lstm_t4)
# gru_arm_2_2_gru_t4 = GRU(1024, return_sequences=True)(gru_arm_1_1_lstm4)
# gru_arm_2_2_gru4 = PReLU()(gru_arm_2_2_gru_t4)
#
# gru_arm_2_3_lstm_t4 = LSTM(1024, return_sequences=True)(gru_arm_1_2_lstm4)
# gru_arm_2_3_lstm4 = PReLU()(gru_arm_2_3_lstm_t4)
# gru_arm_2_4_lstm_t4 = LSTM(1024, return_sequences=True)(gru_arm_1_2_lstm4)
# gru_arm_2_4_lstm4 = PReLU()(gru_arm_2_4_lstm_t4)
#
# gru_arm_2_5_lstm_t4 = LSTM(1024, return_sequences=True)(gru_arm_1_3_gru4)
# gru_arm_2_5_lstm4 = PReLU()(gru_arm_2_5_lstm_t4)
# gru_arm_2_6_conv_t4 = Conv1D(1024, kernel_size=3, padding='same')(gru_arm_1_3_gru4)
# gru_arm_2_6_conv4 = PReLU()(gru_arm_2_6_conv_t4)
#
# gru_arm_2_7_gru_t4 = GRU(1024, return_sequences=True)(gru_arm_1_4_conv4)
# gru_arm_2_7_gru4 = PReLU()(gru_arm_2_7_gru_t4)
# gru_arm_2_8_lstm_t4 = LSTM(1024, return_sequences=True)(gru_arm_1_4_conv4)
# gru_arm_2_8_lstm4 = PReLU()(gru_arm_2_8_lstm_t4)
# # Üçüncü dal
# gru_arm_3_1_lstm_t4 = LSTM(512, return_sequences=True)(gru_arm_2_1_lstm4)
# gru_arm_3_1_lstm4 = LeakyReLU()(gru_arm_3_1_lstm_t4)
# gru_arm_3_2_lstm_t4 = LSTM(512, return_sequences=True)(gru_arm_2_1_lstm4)
# gru_arm_3_2_lstm4 = LeakyReLU()(gru_arm_3_2_lstm_t4)
#
# gru_arm_3_3_conv_t4 = Conv1D(512, kernel_size=3, padding='same')(gru_arm_2_2_gru4)
# gru_arm_3_3_conv4 = LeakyReLU()(gru_arm_3_3_conv_t4)
# gru_arm_3_4_lstm_t4 = LSTM(512, return_sequences=True)(gru_arm_2_2_gru4)
# gru_arm_3_4_lstm4 = LeakyReLU()(gru_arm_3_4_lstm_t4)
#
# gru_arm_3_5_gru_t4 = GRU(512, return_sequences=True)(gru_arm_2_3_lstm4)
# gru_arm_3_5_gru4 = LeakyReLU()(gru_arm_3_5_gru_t4)
# gru_arm_3_6_conv_t4 = Conv1D(512, kernel_size=3, padding='same')(gru_arm_2_3_lstm4)
# gru_arm_3_6_conv4 = LeakyReLU()(gru_arm_3_6_conv_t4)
#
# gru_arm_3_7_gru_t4 = GRU(512, return_sequences=True)(gru_arm_2_4_lstm4)
# gru_arm_3_7_gru4 = LeakyReLU()(gru_arm_3_7_gru_t4)
# gru_arm_3_8_conv_t4 = Conv1D(512, kernel_size=3, padding='same')(gru_arm_2_4_lstm4)
# gru_arm_3_8_conv4 = LeakyReLU()(gru_arm_3_8_conv_t4)
#
# gru_arm_3_9_lstm_t4 = LSTM(512, return_sequences=True)(gru_arm_2_5_lstm4)
# gru_arm_3_9_lstm4 = LeakyReLU()(gru_arm_3_9_lstm_t4)
# gru_arm_3_10_lstm_t4 = LSTM(512, return_sequences=True)(gru_arm_2_5_lstm4)
# gru_arm_3_10_lstm4 = LeakyReLU()(gru_arm_3_10_lstm_t4)
#
# gru_arm_3_11_lstm_t4 = LSTM(512, return_sequences=True)(gru_arm_2_6_conv4)
# gru_arm_3_11_lstm4= LeakyReLU()(gru_arm_3_11_lstm_t4)
# gru_arm_3_12_lstm_t4 = LSTM(512, return_sequences=True)(gru_arm_2_6_conv4)
# gru_arm_3_12_lstm4 = LeakyReLU()(gru_arm_3_12_lstm_t4)
#
# gru_arm_3_13_lstm_t4 = LSTM(512, return_sequences=True)(gru_arm_2_7_gru4)
# gru_arm_3_13_lstm4 = LeakyReLU()(gru_arm_3_13_lstm_t4)
# gru_arm_3_14_lstm_t4 = LSTM(512, return_sequences=True)(gru_arm_2_7_gru4)
# gru_arm_3_14_lstm4 = LeakyReLU()(gru_arm_3_14_lstm_t4)
#
# gru_arm_3_15_lstm_t4 = LSTM(512, return_sequences=True)(gru_arm_2_8_lstm4)
# gru_arm_3_15_lstm4 = LeakyReLU()(gru_arm_3_15_lstm_t4)
# gru_arm_3_16_gru_t4 = GRU(512, return_sequences=True)(gru_arm_2_8_lstm4)
# gru_arm_3_16_gru4 = LeakyReLU()(gru_arm_3_16_gru_t4)
# # G4 çıkışları
# concatenate_G4_1_4 = concatenate([gru_arm_3_1_lstm4, gru_arm_3_2_lstm4, gru_arm_3_3_conv4, gru_arm_3_4_lstm4])
# concatenate_G4_1_4_d = Dense(256)(concatenate_G4_1_4)
# concatenate_G4_5_8 = concatenate([gru_arm_3_5_gru4, gru_arm_3_6_conv4, gru_arm_3_7_gru4, gru_arm_3_8_conv4])
# concatenate_G4_5_8_d = Dense(256)(concatenate_G4_5_8)
# concatenate_G4_9_12 = concatenate([gru_arm_3_9_lstm4, gru_arm_3_10_lstm4, gru_arm_3_11_lstm4, gru_arm_3_12_lstm4])
# concatenate_G4_9_12_d = Dense(256)(concatenate_G4_9_12)
# concatenate_G4_13_16 = concatenate([gru_arm_3_13_lstm4, gru_arm_3_14_lstm4, gru_arm_3_15_lstm4, gru_arm_3_16_gru4])
# concatenate_G4_13_16_d = Dense(256)(concatenate_G4_13_16)
#
# concatenate_G4_1_8 = concatenate([concatenate_G4_1_4_d, concatenate_G4_5_8_d])
# concatenate_G4_1_8_d = Dense(256)(concatenate_G4_1_8)
# concatenate_G4_9_16 = concatenate([concatenate_G4_9_12_d, concatenate_G4_13_16_d])
# concatenate_G4_9_16_d = Dense(256)(concatenate_G4_9_16)
#
# concatenate_G4_pre = concatenate([concatenate_G4_1_8_d, concatenate_G4_9_16_d])
# concatenate_G4 = Dense(256)(concatenate_G4_pre)
#
# # ConvArm
# # C2
# # birinci dal
# conv_arm_1_1_lstm_t = LSTM(1024, return_sequences=True)(conv2)
# conv_arm_1_1_lstm = PReLU()(conv_arm_1_1_lstm_t)
#
# conv_arm_1_2_gru_t = GRU(1024, return_sequences=True)(conv2)
# conv_arm_1_2_gru = PReLU()(conv_arm_1_2_gru_t)
#
# conv_arm_1_3_conv_t = Conv1D(1024, kernel_size=3, padding='same')(conv2)
# conv_arm_1_3_conv = PReLU()(conv_arm_1_3_conv_t)
# # ikinci dal
# conv_arm_2_1_conv_t = Conv1D(512, kernel_size=3, padding='same')(conv_arm_1_1_lstm)
# conv_arm_2_1_conv = LeakyReLU()(conv_arm_2_1_conv_t)
# conv_arm_2_2_conv_t = Conv1D(512, kernel_size=3, padding='same')(conv_arm_1_1_lstm)
# conv_arm_2_2_conv = PReLU()(conv_arm_2_2_conv_t)
#
# conv_arm_2_3_conv_t = Conv1D(512, kernel_size=3, padding='same')(conv_arm_1_2_gru)
# conv_arm_2_3_conv = LeakyReLU()(conv_arm_2_3_conv_t)
# conv_arm_2_4_conv_t = Conv1D(512, kernel_size=3, padding='same')(conv_arm_1_2_gru)
# conv_arm_2_4_conv = PReLU()(conv_arm_2_4_conv_t)
#
# conv_arm_2_5_lstm_t = LSTM(512, return_sequences=True)(conv_arm_1_3_conv)
# conv_arm_2_5_lstm = LeakyReLU()(conv_arm_2_5_lstm_t)
# conv_arm_2_6_gru_t = GRU(512, return_sequences=True)(conv_arm_1_3_conv)
# conv_arm_2_6_gru = PReLU()(conv_arm_2_6_gru_t)
# # üçüncü dal
# concatenate_conv2_pre = concatenate([conv_arm_2_1_conv, conv_arm_2_2_conv, conv_arm_2_3_conv, conv_arm_2_4_conv, conv_arm_2_5_lstm, conv_arm_2_6_gru])
# concatenate_C2 = Dense(256)(concatenate_conv2_pre)
#
# # C4
# # birinci dal
# conv_arm_1_1_lstm_t4 = LSTM(1024, return_sequences=True)(conv4)
# conv_arm_1_1_lstm4 = PReLU()(conv_arm_1_1_lstm_t4)
#
# conv_arm_1_2_gru_t4 = GRU(1024, return_sequences=True)(conv4)
# conv_arm_1_2_gru4 = PReLU()(conv_arm_1_2_gru_t4)
#
# conv_arm_1_3_conv_t4 = Conv1D(1024, kernel_size=3, padding='same')(conv4)
# conv_arm_1_3_conv4 = PReLU()(conv_arm_1_3_conv_t4)
# # ikinci dal
# conv_arm_2_1_conv_t4 = Conv1D(512, kernel_size=3, padding='same')(conv_arm_1_1_lstm4)
# conv_arm_2_1_conv4 = LeakyReLU()(conv_arm_2_1_conv_t4)
# conv_arm_2_2_conv_t4 = Conv1D(512, kernel_size=3, padding='same')(conv_arm_1_1_lstm4)
# conv_arm_2_2_conv4 = PReLU()(conv_arm_2_2_conv_t4)
#
# conv_arm_2_3_conv_t4 = Conv1D(512, kernel_size=3, padding='same')(conv_arm_1_2_gru4)
# conv_arm_2_3_conv4 = LeakyReLU()(conv_arm_2_3_conv_t4)
# conv_arm_2_4_conv_t4 = Conv1D(512, kernel_size=3, padding='same')(conv_arm_1_2_gru4)
# conv_arm_2_4_conv4 = PReLU()(conv_arm_2_4_conv_t4)
#
# conv_arm_2_5_lstm_t4 = LSTM(512, return_sequences=True)(conv_arm_1_3_conv4)
# conv_arm_2_5_lstm4 = LeakyReLU()(conv_arm_2_5_lstm_t4)
# conv_arm_2_6_gru_t4 = GRU(512, return_sequences=True)(conv_arm_1_3_conv4)
# conv_arm_2_6_gru4 = PReLU()(conv_arm_2_6_gru_t4)
# # üçüncü dal
# concatenate_conv4_pre = concatenate([conv_arm_2_1_conv4, conv_arm_2_2_conv4, conv_arm_2_3_conv4, conv_arm_2_4_conv4, conv_arm_2_5_lstm4, conv_arm_2_6_gru4])
# concatenate_C4 = Dense(256)(concatenate_conv4_pre)
#
#
# # RNN Arm
# # R2 - R4
# rnn_arm_1_rnn_t = SimpleRNN(512, return_sequences=True)(simpleRNN2)
# rnn_arm_1_rnn = LeakyReLU()(rnn_arm_1_rnn_t)
#
# rnn_arm_2_gru_t = GRU(512, return_sequences=True)(simpleRNN2)
# rnn_arm_2_gru = LeakyReLU()(rnn_arm_2_gru_t)
#
# rnn_arm_3_rnn_t = SimpleRNN(512, return_sequences=True)(simpleRNN2)
# rnn_arm_3_rnn = LeakyReLU()(rnn_arm_1_rnn_t)
#
# rnn_arm_4_lstm_t = LSTM(512, return_sequences=True)(simpleRNN2)
# rnn_arm_4_lstm = LeakyReLU()(rnn_arm_4_lstm_t)
#
# concatenate_R2_pre = concatenate([rnn_arm_1_rnn, rnn_arm_2_gru, rnn_arm_3_rnn, rnn_arm_4_lstm])
# concatenate_R2 = Dense(256)(concatenate_R2_pre)
#
# rnn_arm_1_rnn_t4 = SimpleRNN(512, return_sequences=True)(simpleRNN4)
# rnn_arm_1_rnn4 = LeakyReLU()(rnn_arm_1_rnn_t4)
#
# rnn_arm_2_gru_t4 = GRU(512, return_sequences=True)(simpleRNN4)
# rnn_arm_2_gru4 = LeakyReLU()(rnn_arm_2_gru_t4)
#
# rnn_arm_3_rnn_t4 = SimpleRNN(512, return_sequences=True)(simpleRNN4)
# rnn_arm_3_rnn4 = LeakyReLU()(rnn_arm_1_rnn_t4)
#
# rnn_arm_4_lstm_t4 = LSTM(512, return_sequences=True)(simpleRNN4)
# rnn_arm_4_lstm4 = LeakyReLU()(rnn_arm_4_lstm_t4)
#
# concatenate_R4_pre = concatenate([rnn_arm_1_rnn4, rnn_arm_2_gru4, rnn_arm_3_rnn4, rnn_arm_4_lstm4])
# concatenate_R4 = Dense(256)(concatenate_R4_pre)
#
# # ARM çıkışları
#
# LSTMArm_pre = concatenate([concatenate_L4, concatenate_L2])
# LSTMArm = Dense(256)(LSTMArm_pre)
#
# GRUArm_pre = concatenate([concatenate_G4, concatenate_G2])
# GRUArm = Dense(256)(GRUArm_pre)
#
# ConvArm_pre = concatenate([concatenate_C4, concatenate_C2])
# ConvArm = Dense(256)(ConvArm_pre)
#
# RNNArm_pre = concatenate([concatenate_R4, concatenate_R2])
# RNNArm = Dense(256)(RNNArm_pre)
#
# # LSTM Booster
# lstm_booster_1_t = LSTM(1024, return_sequences=True)(lstm2)
# lstm_booster_1 = PReLU()(lstm_booster_1_t)
#
# lstm_booster_2_t = LSTM(1024, return_sequences=True)(lstm2)
# lstm_booster_2 = LeakyReLU()(lstm_booster_2_t)
#
# lstm_booster_3_t = LSTM(1024, return_sequences=True)(lstm2)
# lstm_booster_3 = Activation('relu')(lstm_booster_3_t)
#
# concatenate_lstm_booster = concatenate([lstm_booster_1, lstm_booster_2, lstm_booster_3])
# LSTMBooster = Dense(256)(concatenate_lstm_booster)
#
# # LastStep
#
# concatenate_GRU_Conv = concatenate([GRUArm, ConvArm])
# total_GRU_Conv = Dense(256)(concatenate_GRU_Conv)
#
# concatenate_Conv_RNN_LSTM = concatenate([LSTMArm, ConvArm, RNNArm])
# total_Conv_RNN_LSTM = Dense(256)(concatenate_Conv_RNN_LSTM)
#
# concatenate_CRLG = concatenate([total_GRU_Conv, total_Conv_RNN_LSTM])
# total_CRLG = Dense(256)(concatenate_CRLG)
#
# concatenate_All = concatenate([LSTMArm, GRUArm, LSTMBooster, total_CRLG])
# LastStep = Dense(256)(concatenate_All)
#
# # Final
#
# concatenate_LG = concatenate([LSTMArm, GRUArm])
# total_LG = Dense(256)(concatenate_LG)
#
# concatenate_LSLB = concatenate([LastStep, LSTMBooster])
# total_LSLB = Dense(256)(concatenate_LSLB)
#
# concatenate_Final = concatenate([total_LG, total_LSLB])
# FinalArm = Dense(256)(concatenate_Final)
#
# # Çıkış katmanı
# concatenate_FinalLastStep = concatenate([FinalArm, LastStep])
# output = Dense(256)(concatenate_FinalLastStep)
#
# output2_t = LSTM(256, return_sequences=False)(output)
# output2 = PReLU()(output2_t)
#
# final_output = Dense(5)(output2)




# # Model yapısını oluşturma
# input_layer = Input(shape=(sequence_length, features))
#
# # İlk katmanlar
# lstm_1_1_t = LSTM(2560, return_sequences=True)(input_layer)
# lstm_1_1 = Activation(activation='softplus')(lstm_1_1_t)
#
# lstm_1_2_t = LSTM(1280, return_sequences=True)(input_layer)
# lstm_1_2_a = PReLU()(lstm_1_2_t)
# lstm_1_2_d = Dropout(rate=0.3)(lstm_1_2_a)
# lstm_1_2 = BatchNormalization()(lstm_1_2_d)
#
# gru_1_t = GRU(2560, return_sequences=True)(input_layer)
# gru_1 = PReLU()(gru_1_t)
#
# conv1d_1_t = Conv1D(1280, kernel_size=3, padding='same')(input_layer)
# conv1d_1 = LeakyReLU()(conv1d_1_t)
#
# # İkinci katmanlar
# concatenate_LGC = concatenate([lstm_1_1, gru_1, conv1d_1])
# total_LGC = Dense(1280)(concatenate_LGC)
#
# concatenate_LSTM = concatenate([lstm_1_1, lstm_1_2])
# total_LSTM = Dense(1280)(concatenate_LSTM)
#
# concatenate_total = concatenate([total_LGC, total_LSTM])
# total_all = Dense(1280)(concatenate_total)
#
# lstm_2_t = LSTM(2560, return_sequences=True)(lstm_1_1)
# lstm_2 = Activation(activation='softplus')(lstm_2_t)
#
# # üçüncü katmanlar
# concatenate_final = concatenate([lstm_1_1, lstm_1_2, total_all])
# total_final = Dense(1280)(concatenate_final)
#
# concatenate_final2 = concatenate([lstm_2, total_final])
# output_t = LSTM(1280, return_sequences=False)(concatenate_final2)
# output = PReLU()(output_t)
#
# #toplam katmanı
# final_output = Dense(5)(output)



# Model yapısını oluşturma

input_layer = Input(shape=(sequence_length, features))

lstm1_t = LSTM(50, return_sequences=True)(input_layer) # return_sequences=True iken 2D veri döner
lstm1 = PReLU()(lstm1_t)

lstm2_t = LSTM(40, return_sequences=True)(lstm1)
lstm2 = LeakyReLU()(lstm2_t)

lstm3_t = LSTM(30, return_sequences=True)(lstm2)
lstm3 = PReLU()(lstm3_t)

lstm4_t = LSTM(20, return_sequences=True)(lstm3)
lstm4 = LeakyReLU()(lstm4_t)

lstm5_t = LSTM(10, return_sequences=True)(lstm4)
lstm5 = PReLU()(lstm5_t)

lstm6_t = LSTM(25, return_sequences=False)(lstm5) # return_sequences=False iken 1D veri döner
lstm6 = LeakyReLU()(lstm6_t)

final_output = Dense(features)(lstm6)

# input_layer = Input(shape=(sequence_length, features))
# lstm_t = GRU(256, return_sequences=True)(input_layer)
# lstm = PReLU()(lstm_t)
# lstm_t2 = GRU(256, return_sequences=True)(lstm)
# lstm2 = PReLU()(lstm_t2)
# lstm_t3 = GRU(256, return_sequences=False)(lstm2)
# lstm3 = PReLU()(lstm_t3)
# final_output = Dense(5)(lstm3)

# input_layer = Input(shape=(sequence_length, features))
# lstm_t = LSTM(1024, return_sequences=True)(input_layer)
# lstm = PReLU()(lstm_t)
# lstm_t2 = GRU(1024, return_sequences=True)(input_layer)
# lstm2 = PReLU()(lstm_t2)
# concatenated = concatenate([lstm, lstm2])
# lstm_t3 = LSTM(256, return_sequences=False)(concatenated)
# lstm3 = LeakyReLU()(lstm_t3)
# final_output = Dense(5)(lstm3)

# input_layer = Input(shape=(sequence_length, features))
# conv1_t = Conv1D(256, kernel_size=3, padding='same')(input_layer)
# conv1 = PReLU()(conv1_t)
# conv2_t = Conv1D(256, kernel_size=3, padding='same')(conv1)
# conv2 = PReLU()(conv2_t)
# conv3_t = Conv1D(256, kernel_size=3, padding='same')(conv2)
# conv3 = PReLU()(conv3_t)
# flattened = Flatten()(conv3)
# final_output = Dense(5)(flattened)

# input_layer = Input(shape=(sequence_length, features))
#
# lstm_t = LSTM(256, return_sequences=True)(input_layer)
# lstm_a = PReLU()(lstm_t)
# lstm = Flatten()(lstm_a)
#
# gru_t = GRU(256, return_sequences=True)(input_layer)
# gru_a = LeakyReLU()(gru_t)
# gru = Flatten()(gru_a)
#
# conv_t = Conv1D(256, kernel_size=3, padding='same')(input_layer)
# conv_a = PReLU()(conv_t)
# conv = Flatten()(conv_a)
#
# concatenated = concatenate([lstm, gru])
# concatenated2 = concatenate([lstm,conv])
# concatenated3 = concatenate([concatenated, concatenated2])
#
# flattened = Dense(256)(concatenated3)
#
# final_output = Dense(5)(flattened)

# Modeli oluşturma

model = Model(inputs=input_layer, outputs=final_output)

def custom_loss_function(y_true, y_pred, scaler):
    # Ölçeklendirilmiş tolerans değerleri
    time_tolerance_scaled = (3600000000000 - scaler.min_[0]) * scaler.scale_[0] # 1 saat tolerans
    lat_max = (1 / scaler.scale_[1]) + scaler.min_[1]
    lon_max = (1 / scaler.scale_[2]) + scaler.min_[2]
    depth_max = (1 / scaler.scale_[3]) + scaler.min_[3]
    mag_max = (1 / scaler.scale_[4]) + scaler.min_[4]

    lat_lon_tolerance_percentage = 0.05  # Enlem ve boylam için yüzde 5 tolerans
    depth_tolerance_percentage = 0.3  # Derinlik için yüzde 30 tolerans
    mag_tolerance_percentage = 0.2  # Magnitude için yüzde 20 tolerans

    lat_tolerance_scaled = lat_lon_tolerance_percentage * (lat_max - scaler.min_[1]) * scaler.scale_[1]
    lon_tolerance_scaled = lat_lon_tolerance_percentage * (lon_max - scaler.min_[2]) * scaler.scale_[2]
    depth_tolerance_scaled = depth_tolerance_percentage * (depth_max - scaler.min_[3]) * scaler.scale_[3]
    mag_tolerance_scaled = mag_tolerance_percentage * (mag_max - scaler.min_[4]) * scaler.scale_[4]

    # Tarih-saat toleransı ve cezası
    time_diff = y_pred[:, 0] - y_true[:, 0]  # Pozitif değerler ileriye, negatif değerler geçmişe işaret eder
    time_penalty = K.maximum(time_diff - time_tolerance_scaled, 0)  # İleriye dönük tolerans
    past_penalty = K.maximum(-time_diff, 0)  # Geçmişe dönük ceza iki katı

    # Enlem ve boylam toleransı ve cezası
    lat_penalty = K.maximum(K.abs(y_pred[:, 1] - y_true[:, 1]) - lat_tolerance_scaled, 0)
    lon_penalty = K.maximum(K.abs(y_pred[:, 2] - y_true[:, 2]) - lon_tolerance_scaled, 0)

    # Derinlik toleransı ve cezası
    depth_penalty = K.maximum(K.abs(y_pred[:, 3] - y_true[:, 3]) - depth_tolerance_scaled, 0)

    # Magnitude toleransı ve cezası
    mag_penalty = K.maximum(K.abs(y_pred[:, 4] - y_true[:, 4]) - mag_tolerance_scaled, 0)

    # Toplam kaybı hesaplama
    total_loss = K.mean(15 * time_penalty + 30 * past_penalty + 3 * lat_penalty +
                                   3 * lon_penalty + 0.1 * depth_penalty + mag_penalty)

    return total_loss

def custom_accuracy(y_true, y_pred, scaler):
    # Ölçeklendirilmiş tolerans değerleri
    time_tolerance_scaled = (3600000000000 - scaler.min_[0]) * scaler.scale_[0]
    lat_lon_tolerance_percentage = 0.05
    depth_tolerance_percentage = 0.3
    mag_tolerance_percentage = 0.2

    lat_max = (1 / scaler.scale_[1]) + scaler.min_[1]
    lon_max = (1 / scaler.scale_[2]) + scaler.min_[2]
    depth_max = (1 / scaler.scale_[3]) + scaler.min_[3]
    mag_max = (1 / scaler.scale_[4]) + scaler.min_[4]

    lat_tolerance_scaled = lat_lon_tolerance_percentage * (lat_max - scaler.min_[1]) * scaler.scale_[1]
    lon_tolerance_scaled = lat_lon_tolerance_percentage * (lon_max - scaler.min_[2]) * scaler.scale_[2]
    depth_tolerance_scaled = depth_tolerance_percentage * (depth_max - scaler.min_[3]) * scaler.scale_[3]
    mag_tolerance_scaled = mag_tolerance_percentage * (mag_max - scaler.min_[4]) * scaler.scale_[4]

    # Doğruluk hesaplamaları
    correct_time = K.less_equal(K.abs(y_pred[:, 0] - y_true[:, 0]), time_tolerance_scaled)
    correct_lat = K.less_equal(K.abs(y_pred[:, 1] - y_true[:, 1]), lat_tolerance_scaled)
    correct_lon = K.less_equal(K.abs(y_pred[:, 2] - y_true[:, 2]), lon_tolerance_scaled)
    correct_depth = K.less_equal(K.abs(y_pred[:, 3] - y_true[:, 3]), depth_tolerance_scaled)
    correct_mag = K.less_equal(K.abs(y_pred[:, 4] - y_true[:, 4]), mag_tolerance_scaled)

    # Ağırlıklar
    weights = {
        'time': 15.0,   # Zaman için ağırlık
        'lat_lon': 3.0, # Enlem ve boylam için ağırlık
        'depth': 0.1,   # Derinlik için ağırlık
        'mag': 1.0      # Magnitude için ağırlık
    }

    # Ağırlıklı doğruluk hesaplama
    accuracy = (
        weights['time'] * tf.cast(correct_time, tf.float32) +
        weights['lat_lon'] * (tf.cast(correct_lat, tf.float32) + tf.cast(correct_lon, tf.float32)) / 2 +
        weights['depth'] * tf.cast(correct_depth, tf.float32) +
        weights['mag'] * tf.cast(correct_mag, tf.float32)
    ) / sum(weights.values())

    return K.mean(accuracy)

def custom_loss(y_true, y_pred):
    return custom_loss_function(y_true, y_pred, scaler)

def custom_acc(y_true, y_pred):
    return custom_accuracy(y_true, y_pred, scaler)

# Modeli derleme
model.compile(optimizer='adam', loss=custom_loss,
              metrics=[custom_acc])

# Modelin görsel şemasını çizdirme
plot_model(model, to_file='model_diagram.png', show_shapes=True, show_layer_names=True)

checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')

history = None

from keras.callbacks import Callback

class Validate(Callback):
    def __init__(self, prediction_data, scaler):
        super(Validate, self).__init__()
        self.prediction_data = prediction_data
        self.scaler = scaler
        self.predict_must_be = pd.DataFrame({
            'tarih-saat': pd.to_datetime(['10-11-2023 05:06:26'], format='%d-%m-%Y %H:%M:%S'),
            'latitude': [39.034],
            'longitude': [27.772],
            'depth': [6.89],
            'magnitude': [1.2]
        })

    def on_epoch_end(self, epoch, logs=None):
        # Tarih-saat sütununu dönüştürme
        self.prediction_data['tarih-saat'] = self.prediction_data['tarih-saat'].astype(np.int64).astype(np.float64)
        self.predict_must_be['tarih-saat'] = self.predict_must_be['tarih-saat'].astype(np.int64).astype(np.float64)

        # Veriyi ölçeklendirme
        self.new_data_scaled = self.scaler.transform(self.prediction_data)
        self.must_be_data_scaled = self.scaler.transform(self.predict_must_be)

        # Tahmin yapma
        self.predicted = self.model.predict(np.array([self.new_data_scaled]))
        self.predicted2 = self.scaler.inverse_transform(self.predicted)  # Ölçeklendirmeyi geri al

        # Tahmini tarih-saat değerini geri dönüştürme
        self.predicted_timestamp = pd.to_datetime(self.predicted2[0][0], unit='ns')
        self.must_be_timestamp = pd.to_datetime(self.predict_must_be['tarih-saat'][0], unit='ns')

        print(f"Beklenen:\t\t{self.must_be_timestamp:%Y-%m-%d %H:%M:%S}\t"
              f"{self.predict_must_be['latitude'][0]:.4f}\t"
              f"{self.predict_must_be['longitude'][0]:.4f}\t"
              f"{self.predict_must_be['depth'][0]:.2f}\t"
              f"{self.predict_must_be['magnitude'][0]:.2f}")
        print(f"Gerçekleşen:\t{self.predicted_timestamp:%Y-%m-%d %H:%M:%S}\t"
              f"{self.predicted2[0][1]:.4f}\t"
              f"{self.predicted2[0][2]:.4f}\t"
              f"{self.predicted2[0][3]:.2f}\t"
              f"{self.predicted2[0][4]:.2f}")

# Yeni veri: [Tarih-saat, longitude, latitude, depth, magnitude]
data_to_sent_callback = pd.DataFrame({
    'tarih-saat': pd.to_datetime([
        '10-11-2023 03:48:57',
        '10-11-2023 03:50:32',
        '10-11-2023 04:09:30',
        '10-11-2023 04:18:45',
        '10-11-2023 04:35:41'
    ], format='%d-%m-%Y %H:%M:%S'),
    'latitude': [35.268, 37.824, 38.712, 38.859, 38.079],
    'longitude': [26.691, 36.076, 39.972, 37.341, 36.607],
    'depth': [8.9, 7, 7, 4.78, 7.13],
    'magnitude': [2.9, 1.8, 1.4, 2.0, 2.1]
})

callback1 = Validate(prediction_data=data_to_sent_callback, scaler=scaler)

try:
    # Modeli eğit
    history = model.fit(
        X_train, y_train,
        epochs=2000,
        batch_size=8,
        verbose=1,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint, callback1]
    )
except KeyboardInterrupt:
    # Eğitim Ctrl+C ile durdurulursa
    print("\nEğitim klavye kesmesi ile durduruldu.")

# Modeli kaydetme
model.save('deprem_tahmin_modeli.h5')
joblib.dump(scaler, 'scaler.pkl')

# Tarihçeden verileri al
if history:
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['custom_acc']
    val_acc = history.history['val_custom_acc']
    epochs = range(1, len(train_loss) + 1)

    # Eğitim kaybı ve doğruluk değerleri için grafik
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='g')
    ax1.plot(epochs, train_loss, 'g-', label='Eğitim Kaybı')
    ax1.tick_params(axis='y', labelcolor='g')

    ax2 = ax1.twinx()  # Ortak x eksenini paylaşan ikinci bir y eksenini oluştur
    ax2.set_ylabel('Accuracy', color='b')
    ax2.plot(epochs, train_acc, 'b-', label='Eğitim Doğruluğu')
    ax2.tick_params(axis='y', labelcolor='b')

    fig.tight_layout()  # Düzeni otomatik ayarla
    plt.title('Eğitim Kaybı ve Doğruluk')
    plt.savefig('train_loss_accuracy.png')  # Grafikleri PNG olarak kaydet
    plt.show()

    # Doğrulama kaybı ve doğruluk değerleri için grafik
    fig, ax3 = plt.subplots()

    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Val Loss', color='r')
    ax3.plot(epochs, val_loss, 'r-', label='Doğrulama Kaybı')
    ax3.tick_params(axis='y', labelcolor='r')

    ax4 = ax3.twinx()
    ax4.set_ylabel('Val Accuracy', color='c')
    ax4.plot(epochs, val_acc, 'c-', label='Doğrulama Doğruluğu')
    ax4.tick_params(axis='y', labelcolor='c')

    fig.tight_layout()
    plt.title('Doğrulama Kaybı ve Doğruluk')
    plt.savefig('val_loss_accuracy.png')  # Grafikleri PNG olarak kaydet
    plt.show()

# Test seti üzerinde performansı değerlendirme
test_loss = model.evaluate(X_test, y_test)
print(f"Test Kaybı: {test_loss}")

# Performans metrikleri
y_pred = model.predict(X_test)
y_pred = y_pred.squeeze()
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'MSE: {mse}, R2: {r2}, RMSE: {rmse}')









########################################################################################################################
from keras.models import load_model

# Modeli yükleme
model = load_model('deprem_tahmin_modeli.h5', custom_objects={'custom_loss': custom_loss,
                                                              'custom_acc': custom_acc})
# Ölçeklendirici yükleme (Bu örnekte, ölçeklendiriciyi 'scaler.pkl' olarak kaydettiğinizi varsayıyorum)
import joblib
scaler = joblib.load('scaler.pkl')

# Yeni veri: [Tarih-saat, longitude, latitude, depth, magnitude]
new_data = pd.DataFrame({
    'tarih-saat': pd.to_datetime([
        '10-11-2023 03:48:57',
        '10-11-2023 03:50:32',
        '10-11-2023 04:09:30',
        '10-11-2023 04:18:45',
        '10-11-2023 04:35:41'
    ], format='%d-%m-%Y %H:%M:%S'),
    'latitude': [35.268, 37.824, 38.712, 38.859, 38.079],
    'longitude': [26.691, 36.076, 39.972, 37.341, 36.607],
    'depth': [8.9, 7, 7, 4.78, 7.13],
    'magnitude': [2.9, 1.8, 1.4, 2.0, 2.1]
})

predict_must_be = pd.DataFrame({
    'tarih-saat' : pd.to_datetime(['10-11-2023 05:06:26'], format='%d-%m-%Y %H:%M:%S').astype(np.int64),
    'latitude' : [39.034],
    'longitude' : [27.772],
    'depth' : [6.89],
    'magnitude' : [1.2]
})

predict_must_be_scaled = scaler.transform(predict_must_be)

# Tarih-saat sütununu Unix timestamp'e dönüştürme
new_data['tarih-saat'] = new_data['tarih-saat'].astype(np.int64)

# Veriyi ölçeklendirme
new_data_scaled = scaler.transform(new_data)

# Tahmin yapma
predicted = model.predict(np.array([new_data_scaled]))
predicted2 = scaler.inverse_transform(predicted)  # Ölçeklendirmeyi geri al

# Tahmini tarih-saat değerini geri dönüştürme
predicted_timestamp = pd.to_datetime(predicted2[0][0], unit='ns')
print(f"Tahmini Tarih-Saat: {predicted_timestamp}, sapma: %{(predicted[0][0]-predict_must_be_scaled[0][0])/predict_must_be_scaled[0][0]*100}")
print(f"Tahmini Latitude: {predicted2[0][1]}, sapma: %{(predicted[0][1]-predict_must_be_scaled[0][1])/predict_must_be_scaled[0][1]*100}")
print(f"Tahmini Longitude: {predicted2[0][2]}, sapma: %{(predicted[0][2]-predict_must_be_scaled[0][2])/predict_must_be_scaled[0][2]*100}")
print(f"Tahmini Depth: {predicted2[0][3]}, sapma: %{(predicted[0][3]-predict_must_be_scaled[0][3])/predict_must_be_scaled[0][3]*100}")
print(f"Tahmini Magnitude: {predicted2[0][4]}, sapma: %{(predicted[0][4]-predict_must_be_scaled[0][4])/predict_must_be_scaled[0][4]*100}")







# from sklearn.inspection import permutation_importance
# from sklearn.model_selection import cross_val_score
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.svm import SVR
# from sklearn.neural_network import MLPRegressor
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.multioutput import MultiOutputRegressor
# import matplotlib.pyplot as plt
#
# # Veri setini yükleme ve hazırlama
# df = pd.read_csv('veriler.csv')
# df['tarih-saat'] = pd.to_datetime(df['tarih-saat'], format='%d-%m-%Y %H:%M:%S').astype(np.int64) // 10**9
#
# # custom_epoch = pd.to_datetime('11-10-2023 15:00:00', format='%d-%m-%Y %H:%M:%S')
# # df['tarih-saat'] = (pd.to_datetime(df['tarih-saat'], format='%d-%m-%Y %H:%M:%S') - custom_epoch).astype(np.int64) // 10**9
#
#
# # tüm veriler girdi / tüm veriler çıktı
# X = df
# y = df
#
#
# # sadece GBR
#
# # tarih-saat girdi / magnitude çıktı
# # X = df.drop(columns=['latitude', 'longitude', 'depth', 'magnitude'])
# # y = df.drop(columns=['tarih-saat', 'latitude', 'longitude', 'depth'])
#
# # konum girdi / magnitude çıktı
# # X = df.drop(columns=['tarih-saat', 'depth', 'magnitude'])
# # y = df.drop(columns=['tarih-saat', 'latitude', 'longitude', 'depth'])
#
# # magnitude girdi / derinlik çıktı
# # X = df.drop(columns=['tarih-saat', 'latitude', 'longitude', 'depth'])
# # y = df.drop(columns=['tarih-saat', 'latitude', 'longitude', 'magnitude'])
#
# # magnitude harici girdi / magnitude çıktı
# # X = df.drop(columns=['magnitude'])
# # y = df.drop(columns=['tarih-saat', 'latitude', 'longitude', 'depth'])
#
#
#
#
#
# # zamansız
#
# # konum girdi / magnitude ve depth çıktı
# # X = df.drop(columns=['tarih-saat', 'magnitude', 'depth'])
# # y = df.drop(columns=['tarih-saat','latitude', 'longitude'])
#
# # konum ve depth girdi / magnitude çıktı
# # X = df.drop(columns=['tarih-saat', 'magnitude'])
# # y = df.drop(columns=['tarih-saat', 'latitude', 'longitude', 'depth'])
#
#
#
#
# # MLP/RFR için
#
# # tarih saat girdi / magnitude ve derinlik
# # X = df.drop(columns=['latitude', 'longitude', 'depth', 'magnitude'])
# # y = df.drop(columns=['tarih-saat', 'latitude', 'longitude'])
#
# # konum girdi / depth magnitude
# # X = df.drop(columns=['tarih-saat', 'depth', 'magnitude'])
# # y = df.drop(columns=['tarih-saat', 'latitude', 'longitude'])
#
# # magnitude hariç girdi / magnitude depth çıktı
# # X = df.drop(columns=['magnitude'])
# # y = df.drop(columns=['tarih-saat', 'latitude', 'longitude'])
#
#
# # Veri setini eğitim ve test setlerine ayırma %20-%80
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Modeli oluşturma ve eğitme
# model = MultiOutputRegressor(GradientBoostingRegressor(random_state=42))
# # model = RandomForestRegressor(random_state=42)
# # model = MLPRegressor(random_state=42, max_iter=1000)
#
# model.fit(X_train, y_train)
#
# # Permütasyon Özelliği Önemi
# perm_importance = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)
#
# print('Permutation Feature Importance')
#
# # Permütasyon özelliğin önemini ve standart sapmasını yazdırma
# for i in perm_importance.importances_mean.argsort()[::-1]:
#     if perm_importance.importances_mean[i] - 2 * perm_importance.importances_std[i] > 0:
#         print(f"{X.columns[i]}: {perm_importance.importances_mean[i]:.5f} +/- {perm_import*/ance.importances_std[i]:.5f}")
#
# print('\n')
#
# # Çapraz doğrulama ile modeli değerlendirme
# cv_scores = cross_val_score(model, X, y, cv=5)
# print(f'Cross-validation scores: {cv_scores}\n')
# print(f'Average cross-validation score: {cv_scores.mean()}\n')
#
# # Modelin MSE'sini hesaplama
# y_pred = model.predict(X_test)
# for i, col in enumerate(y.columns):
#     mse = mean_squared_error(y_test[col], y_pred[:, i])
#     print(f'Mean Squared Error for {col}: {mse}')
#
# print('\n')
#
# # Normalize Error değerlerinin yazdırılması
# for col in y.columns:
#     mse = mean_squared_error(y_test[col], y_pred[:, i])
#     range_of_values = y[col].max() - y[col].min()
#     normalized_error = mse / range_of_values
#     print(f'Normalized Error for {col}: {normalized_error}')
#
# print('\n')
#
# # MAE (Mean Absolute Error)
# mae = mean_absolute_error(y_test, y_pred)
# print(f"Mean Absolute Error (MAE): {mae}")
#
# # RMSE (Root Mean Squared Error)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print(f"Root Mean Squared Error (RMSE): {rmse}")
#
# # R² (R-squared, Determination Coefficient)
# r_squared = r2_score(y_test, y_pred)
# print(f"R-squared (R²): {r_squared}")
#
#
# try:
#     # Gerçek ve tahmin edilen değerler
#     true_values = y_test['depth']
#     predicted_values = y_pred[:, y.columns.get_loc('depth')]
#
#     # Scatter plot
#     plt.figure(figsize=(10, 6))
#     plt.scatter(true_values, predicted_values, alpha=0.5)
#     plt.title('Derinlik Gerçek Değerler vs. Tahmin Edilen Değerler')
#     plt.xlabel('Gerçek Değerler')
#     plt.ylabel('Tahmin Edilen Değerler')
#     plt.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], 'k--') # Ideal line
#     plt.savefig('depth.png')
#     plt.show()
#
# except:
#     pass
#
#
# try:
#     # Gerçek ve tahmin edilen değerler
#     true_values = y_test['tarih-saat']
#     predicted_values = y_pred[:, y.columns.get_loc('tarih-saat')]
#
#     # Scatter plot
#     plt.figure(figsize=(10, 6))
#     plt.scatter(true_values, predicted_values, alpha=0.5)
#     plt.title('Tarih-Saat Gerçek Değerler vs. Tahmin Edilen Değerler')
#     plt.xlabel('Gerçek Değerler')
#     plt.ylabel('Tahmin Edilen Değerler')
#     plt.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], 'k--') # Ideal line
#     plt.savefig('time.png')
#     plt.show()
# except:
#     pass
#
# try:
#     # Gerçek ve tahmin edilen değerler
#     true_values = y_test['longitude']
#     predicted_values = y_pred[:, y.columns.get_loc('longitude')]
#
#     # Scatter plot
#     plt.figure(figsize=(10, 6))
#     plt.scatter(true_values, predicted_values, alpha=0.5)
#     plt.title('Boylam Gerçek Değerler vs. Tahmin Edilen Değerler')
#     plt.xlabel('Gerçek Değerler')
#     plt.ylabel('Tahmin Edilen Değerler')
#     plt.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], 'k--') # Ideal line
#     plt.savefig('longitude.png')
#     plt.show()
# except:
#     pass
#
#
# try:
#     # Gerçek ve tahmin edilen değerler
#     true_values = y_test['latitude']
#     predicted_values = y_pred[:, y.columns.get_loc('latitude')]
#
#     # Scatter plot
#     plt.figure(figsize=(10, 6))
#     plt.scatter(true_values, predicted_values, alpha=0.5)
#     plt.title('Enlem Gerçek Değerler vs. Tahmin Edilen Değerler')
#     plt.xlabel('Gerçek Değerler')
#     plt.ylabel('Tahmin Edilen Değerler')
#     plt.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], 'k--') # Ideal line
#     plt.savefig('latitude.png')
#     plt.show()
# except:
#     pass
#
# try:
#     # Gerçek ve tahmin edilen değerler
#     true_values = y_test['magnitude']
#     predicted_values = y_pred[:, y.columns.get_loc('magnitude')]
#
#     # Scatter plot
#     plt.figure(figsize=(10, 6))
#     plt.scatter(true_values, predicted_values, alpha=0.5)
#     plt.title('Büyüklük Gerçek Değerler vs. Tahmin Edilen Değerler')
#     plt.xlabel('Gerçek Değerler')
#     plt.ylabel('Tahmin Edilen Değerler')
#     plt.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], 'k--') # Ideal line
#     plt.savefig('magnitude.png')
#     plt.show()
# except:
#     pass
#
# try:
#     importances = perm_importance.importances_mean
#     indices = np.argsort(importances)[::-1]
#
#     plt.figure(figsize=(10, 6))
#     plt.title('Özellik Önemlilikleri')
#     plt.bar(range(X.shape[1]), importances[indices], align='center', alpha=0.5)
#     plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
#     plt.xlabel('Özellikler')
#     plt.ylabel('Önemlilik Derecesi')
#     plt.savefig('perm_importance.png')
#     plt.show()
# except:
#     pass
