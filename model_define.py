import tensorflow as tf
#from tensorflow import keras
#from keras.layers import *

from tensorflow.keras.layers import Conv1D, Dropout, concatenate, LSTM, Dense, Lambda
from tensorflow.keras import Input, Model

'''
%matplotlib inline
%load_ext autoreload
%autoreload 2
%config InlineBackend.figure_format ='retina'
'''


class CNNLSTM_model:
    def __init__(self,
    input_shape):
        ## TODO: Hyperparameter 넣는 방법 추가
        self.input_data = Input(shape=input_shape)
        
        #model define
        CNNLSTM_layer_1 = self.CNNLSTM_layer(self.input_data)
        concatenate_1 = concatenate([self.input_data, CNNLSTM_layer_1])

        CNNLSTM_layer_2 = self.CNNLSTM_layer(concatenate_1)
        concatenate_2 = concatenate([CNNLSTM_layer_1, CNNLSTM_layer_2])

        CNNLSTM_layer_3 = self.CNNLSTM_layer(concatenate_2)
        concatenate_3  = concatenate([CNNLSTM_layer_2, CNNLSTM_layer_3])

        CNNLSTM_layer_4 = self.CNNLSTM_layer(concatenate_3)
        concatenate_4 = concatenate([CNNLSTM_layer_3, CNNLSTM_layer_4])
        
        CNNLSTM_layer_5 = self.CNNLSTM_layer(concatenate_4)
        concatenate_5 = concatenate([CNNLSTM_layer_4, CNNLSTM_layer_5])

        LSTM_middle_1 = self.select_last_LSTM_cell(concatenate_3)
        LSTM_middle_2 = self.select_last_LSTM_cell(concatenate_4)
        self.LSTM_middle_final = self.select_last_LSTM_cell(concatenate_5)

        self.classification_layer_1 = self.classification_layer(LSTM_middle_1, layer_name='1')
        self.classification_layer_2 = self.classification_layer(LSTM_middle_2, layer_name='2')
        self.MTL_classification_layer, self.MTL_regression_layer = \
            self.MTL_layer(self.LSTM_middle_final, layer_name='final')

        self._MTL_model = Model(self.input_data, [self.classification_layer_1, self.classification_layer_2,
                                        self.MTL_classification_layer, self.MTL_regression_layer])
        self.classification_model = Model(self.input_data, self.MTL_classification_layer)
        self._regression_model = Model(self.input_data, self.MTL_regression_layer)
        self._clustering_model = Model(self.input_data, self.LSTM_middle_final)

    def MTL_model(self):
        
        return self._MTL_model

    def classification_model(self):
        
        return self.classification_model

    def regression_model(self):
        
        return self._regression_model

    def clustering_model(self):
        
        return self._clustering_model

    def CNNLSTM_layer(self, layer_input):
        kernal_size = 128
        RNN_cell_size = 256
        dropout = 0.5

        input_data = layer_input

        CNN_1 = Conv1D(kernal_size, [1], activation='relu', padding='same', kernel_initializer='he_normal')(input_data)
        CNN_1 = Dropout(dropout)(CNN_1)

        CNN_2 = Conv1D(kernal_size, [2], activation='relu', padding='same', kernel_initializer='he_normal')(input_data)
        CNN_2 = Dropout(dropout)(CNN_2)

        CNN_3 = Conv1D(kernal_size, [3], activation='relu', padding='same', kernel_initializer='he_normal')(input_data)
        CNN_3 = Dropout(dropout)(CNN_3)

        CNN_4 = Conv1D(kernal_size, [4], activation='relu', padding='same', kernel_initializer='he_normal')(input_data)
        CNN_4 = Dropout(dropout)(CNN_4)

        CNN_5 = Conv1D(kernal_size, [4], activation='relu', padding='same', kernel_initializer='he_normal')(input_data)
        CNN_5 = Dropout(dropout)(CNN_5)

        CNN_concatenate = concatenate([CNN_1, CNN_2, CNN_3, CNN_4, CNN_5])

        LSTM_layer = LSTM(RNN_cell_size, return_sequences=True)(CNN_concatenate)
        LSTM_layer = Dropout(dropout)(LSTM_layer)

        return LSTM_layer

    def select_last_LSTM_cell(self,LSTM):
        last_LSTM = Lambda(lambda x: x[:,-1,:])(LSTM)
        return last_LSTM

    def classification_layer(self, layer_input, layer_name='1'):
        DNN_1_cell = 100
        DNN_2_cell = 100
        DNN_final_cell = 11
        dropout = 0.5

        input_data = layer_input

        DNN_class_layer_1 = Dense(DNN_1_cell, activation='relu')(input_data)
        DNN_class_layer_1 = Dropout(dropout)(DNN_class_layer_1)

        DNN_class_layer_2 = Dense(DNN_2_cell, activation='relu')(DNN_class_layer_1)
        DNN_class_layer_2 = Dropout(dropout)(DNN_class_layer_2)

        DNN_class_layer_3 = Dense(DNN_final_cell,
                                  activation='softmax',
                                  name='class_{}'.format(layer_name))(DNN_class_layer_2)

        return DNN_class_layer_3

    def MTL_layer(self, layer_input, layer_name='3'):
        DNN_class_1_cell = 100
        DNN_class_2_cell = 100
        DNN_class_final_cell = 11

        DNN_rate_1_cell = 100
        DNN_rate_final_cell = 1

        dropout = 0.5

        input_data = layer_input

        DNN_class_layer_1 = Dense(DNN_class_1_cell, activation='relu')(input_data)
        DNN_class_layer_1 = Dropout(dropout)(DNN_class_layer_1)

        DNN_class_layer_2 = Dense(DNN_class_2_cell, activation='relu')(DNN_class_layer_1)
        DNN_class_layer_2 = Dropout(dropout)(DNN_class_layer_2)

        DNN_class_layer_3 = Dense(DNN_class_final_cell,
                                  activation='softmax',
                                  name='class_{}'.format(layer_name))(DNN_class_layer_2)

        DNN_rate_layer_1 = Dense(DNN_rate_1_cell, activation='relu')(DNN_class_layer_2)
        DNN_rate_layer_1 = Dropout(dropout)(DNN_rate_layer_1)

        DNN_rate_layer_2 = Dense(DNN_rate_final_cell, activation='linear', name='rate')(DNN_rate_layer_1)

        return DNN_class_layer_3, DNN_rate_layer_2
