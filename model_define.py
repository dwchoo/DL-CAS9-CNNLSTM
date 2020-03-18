import tensorflow as tf
#from tensorflow import keras
#from keras.layers import *

from tensorflow.keras.layers import Conv1D, Conv2D,Dropout, concatenate, LSTM, Dense, Lambda, Reshape
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import Adam

'''
%matplotlib inline
%load_ext autoreload
%autoreload 2
%config InlineBackend.figure_format ='retina'
'''

default_HP = {
    'CNN_1_kernal' : 16,
    'CNN_1_RNN' : 64,
    'CNN_2_kernal' : 32,
    'CNN_2_RNN' : 128,
    'CNN_3_kernal' : 64,
    'CNN_3_RNN' : 256,
    'CNN_4_kernal' : 128,
    'CNN_4_RNN' : 256,
    'CNN_5_kernal' : 128,
    'CNN_5_RNN' : 256,

    'DNN_1' : 100,
    'DNN_2' : 100,
    'DNN_rate' : 100,

    'CNN_dropout' : 0.5,
    'DNN_dropout' : 0.5,

    'learning_rate' : 0.0001
}


class CNNLSTM_model:
    def __init__(self, input_shape, HP = default_HP):
        self.input_data = Input(shape=input_shape)
        self._HP = HP


        self.HP_define()
        self.training_model = self.model_define()
        self.model_compile()


    def HP_define(self):
        self.CNN_1_kernal = self._HP['CNN_1_kernal']
        self.CNN_1_RNN    = self._HP['CNN_1_RNN']
        self.CNN_2_kernal = self._HP['CNN_2_kernal']
        self.CNN_2_RNN    = self._HP['CNN_2_RNN']
        self.CNN_3_kernal = self._HP['CNN_3_kernal']
        self.CNN_3_RNN    = self._HP['CNN_3_RNN']
        self.CNN_4_kernal = self._HP['CNN_4_kernal']
        self.CNN_4_RNN    = self._HP['CNN_4_RNN']
        self.CNN_5_kernal = self._HP['CNN_5_kernal']
        self.CNN_5_RNN    = self._HP['CNN_5_RNN']
        self.DNN_1        = self._HP['DNN_1']
        self.DNN_2        = self._HP['DNN_2']
        self.DNN_rate     = self._HP['DNN_rate']
        self.CNN_dropout  = self._HP['CNN_dropout' ]
        self.DNN_dropout  = self._HP['DNN_dropout' ]
        self.learning_rate  =  self._HP['learning_rate']

        
    def model_define(self):
        #model define
        CNNLSTM_layer_1 = self.CNNLSTM_layer(self.input_data,
                                            kernal_size=self.CNN_1_kernal,
                                            RNN_cell_size=self.CNN_1_RNN,
                                            dropout=self.CNN_dropout)
        concatenate_1 = concatenate([self.input_data, CNNLSTM_layer_1])

        CNNLSTM_layer_2 = self.CNNLSTM_layer(self.input_data,
                                            kernal_size=self.CNN_2_kernal,
                                            RNN_cell_size=self.CNN_2_RNN,
                                            dropout=self.CNN_dropout)
        concatenate_2 = concatenate([CNNLSTM_layer_1, CNNLSTM_layer_2])

        CNNLSTM_layer_3 = self.CNNLSTM_layer(self.input_data,
                                            kernal_size=self.CNN_3_kernal,
                                            RNN_cell_size=self.CNN_3_RNN,
                                            dropout=self.CNN_dropout)
        concatenate_3  = concatenate([CNNLSTM_layer_2, CNNLSTM_layer_3])

        CNNLSTM_layer_4 = self.CNNLSTM_layer(self.input_data,
                                            kernal_size=self.CNN_4_kernal,
                                            RNN_cell_size=self.CNN_4_RNN,
                                            dropout=self.CNN_dropout)
        concatenate_4 = concatenate([CNNLSTM_layer_3, CNNLSTM_layer_4])
        
        CNNLSTM_layer_5 = self.CNNLSTM_layer(self.input_data,
                                            kernal_size=self.CNN_5_kernal,
                                            RNN_cell_size=self.CNN_5_RNN,
                                            dropout=self.CNN_dropout)
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
        self._classification_model = Model(self.input_data, self.MTL_classification_layer)
        self._regression_model = Model(self.input_data, self.MTL_regression_layer)
        self._clustering_model = Model(self.input_data, self.LSTM_middle_final)


        return self._MTL_model

    def model_compile(self):
        # compile
        optimizer = Adam(learning_rate = self.learning_rate)
        self.training_model.compile(optimizer=optimizer,
                                    loss={
                                        #'num_mis': 'categorical_crossentropy',
                                        'class_1': 'categorical_crossentropy',
                                        'class_2': 'categorical_crossentropy',
                                        'class_final': 'categorical_crossentropy',
                                        'rate': 'mean_squared_error'},
                                    loss_weights={
                                        #'num_mis': 0.5,
                                        'class_1': 1,
                                        'class_2': 1,
                                        'class_final': 0.5,
                                        'rate': 1},
                                    metrics={
                                        #'num_mis': 'accuracy',
                                        'class_1': "accuracy",
                                        'class_2': "accuracy",
                                        'class_final': "accuracy"}
                                    )


    def MTL_model(self,):
        return self._MTL_model

    def classification_model(self,):
        return self._classification_model

    def regression_model(self):
        return self._regression_model

    def clustering_model(self):
        return self._clustering_model

    def CNNLSTM_layer(self, layer_input, kernal_size = 128, RNN_cell_size = 256, dropout = 0.5):
        kernal_size = kernal_size
        RNN_cell_size = RNN_cell_size
        dropout = dropout

        input_data = layer_input

        CNN_1 = Conv1D(kernal_size, [1], activation='relu', padding='same', kernel_initializer='he_normal')(input_data)
        CNN_1 = Dropout(dropout)(CNN_1)

        CNN_2 = Conv1D(kernal_size, [2], activation='relu', padding='same', kernel_initializer='he_normal')(input_data)
        CNN_2 = Dropout(dropout)(CNN_2)

        CNN_3 = Conv1D(kernal_size, [3], activation='relu', padding='same', kernel_initializer='he_normal')(input_data)
        CNN_3 = Dropout(dropout)(CNN_3)

        CNN_4 = Conv1D(kernal_size, [4], activation='relu', padding='same', kernel_initializer='he_normal')(input_data)
        CNN_4 = Dropout(dropout)(CNN_4)

        CNN_5 = Conv1D(kernal_size, [5], activation='relu', padding='same', kernel_initializer='he_normal')(input_data)
        CNN_5 = Dropout(dropout)(CNN_5)

        CNN_concatenate = concatenate([CNN_1, CNN_2, CNN_3, CNN_4, CNN_5])

        LSTM_layer = LSTM(RNN_cell_size, return_sequences=True)(CNN_concatenate)
        LSTM_layer = Dropout(dropout)(LSTM_layer)

        return LSTM_layer

    def select_last_LSTM_cell(self,LSTM):
        last_LSTM = Lambda(lambda x: x[:,-1,:])(LSTM)
        return last_LSTM

    def classification_layer(self, layer_input, layer_name='1'):
        DNN_1_cell = self.DNN_1
        DNN_2_cell = self.DNN_2
        DNN_final_cell = 11
        dropout = self.DNN_dropout

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
        DNN_class_1_cell = self.DNN_1
        DNN_class_2_cell = self.DNN_2
        DNN_class_final_cell = 11

        DNN_rate_1_cell = self.DNN_rate
        DNN_rate_final_cell = 1

        dropout = self.DNN_dropout

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








class CNNLSTM_3_layer_model(CNNLSTM_model):
    def __init__(self, input_shape, HP=default_HP):
        super().__init__(input_shape, HP=HP)

    def HP_define(self):
        self.CNN_1_kernal = self._HP['CNN_1_kernal']
        self.CNN_1_RNN    = self._HP['CNN_1_RNN']
        self.CNN_2_kernal = self._HP['CNN_2_kernal']
        self.CNN_2_RNN    = self._HP['CNN_2_RNN']
        self.CNN_3_kernal = self._HP['CNN_3_kernal']
        self.CNN_3_RNN    = self._HP['CNN_3_RNN']
        #self.CNN_4_kernal = self._HP['CNN_4_kernal']
        #self.CNN_4_RNN    = self._HP['CNN_4_RNN']
        #self.CNN_5_kernal = self._HP['CNN_5_kernal']
        #self.CNN_5_RNN    = self._HP['CNN_5_RNN']
        self.DNN_1        = self._HP['DNN_1']
        self.DNN_2        = self._HP['DNN_2']
        self.DNN_rate     = self._HP['DNN_rate']
        self.CNN_dropout  = self._HP['CNN_dropout' ]
        self.DNN_dropout  = self._HP['DNN_dropout' ]
        self.learning_rate  =  self._HP['learning_rate']

    def model_define(self):
        #model define
        CNNLSTM_layer_1 = self.CNNLSTM_layer(self.input_data,
                                            kernal_size=self.CNN_1_kernal,
                                            RNN_cell_size=self.CNN_1_RNN,
                                            dropout=self.CNN_dropout)
        concatenate_1 = concatenate([self.input_data, CNNLSTM_layer_1])

        CNNLSTM_layer_2 = self.CNNLSTM_layer(self.input_data,
                                            kernal_size=self.CNN_2_kernal,
                                            RNN_cell_size=self.CNN_2_RNN,
                                            dropout=self.CNN_dropout)
        concatenate_2 = concatenate([CNNLSTM_layer_1, CNNLSTM_layer_2])

        CNNLSTM_layer_3 = self.CNNLSTM_layer(self.input_data,
                                            kernal_size=self.CNN_3_kernal,
                                            RNN_cell_size=self.CNN_3_RNN,
                                            dropout=self.CNN_dropout)
        concatenate_3  = concatenate([CNNLSTM_layer_2, CNNLSTM_layer_3])

        #CNNLSTM_layer_4 = self.CNNLSTM_layer(self.input_data,
        #                                    kernal_size=self.CNN_4_kernal,
        #                                    RNN_cell_size=self.CNN_4_RNN,
        #                                    dropout=self.CNN_dropout)
        #concatenate_4 = concatenate([CNNLSTM_layer_3, CNNLSTM_layer_4])
        #
        #CNNLSTM_layer_5 = self.CNNLSTM_layer(self.input_data,
        #                                    kernal_size=self.CNN_5_kernal,
        #                                    RNN_cell_size=self.CNN_5_RNN,
        #                                    dropout=self.CNN_dropout)
        #concatenate_5 = concatenate([CNNLSTM_layer_4, CNNLSTM_layer_5])

        LSTM_middle_1 = self.select_last_LSTM_cell(concatenate_1)
        LSTM_middle_2 = self.select_last_LSTM_cell(concatenate_2)
        self.LSTM_middle_final = self.select_last_LSTM_cell(concatenate_3)

        self.classification_layer_1 = self.classification_layer(LSTM_middle_1, layer_name='1')
        self.classification_layer_2 = self.classification_layer(LSTM_middle_2, layer_name='2')
        self.MTL_classification_layer, self.MTL_regression_layer = \
            self.MTL_layer(self.LSTM_middle_final, layer_name='final')

        self._MTL_model = Model(self.input_data, [self.classification_layer_1, self.classification_layer_2,
                                        self.MTL_classification_layer, self.MTL_regression_layer])
        self._classification_model = Model(self.input_data, self.MTL_classification_layer)
        self._regression_model = Model(self.input_data, self.MTL_regression_layer)
        self._clustering_model = Model(self.input_data, self.LSTM_middle_final)

        return self._MTL_model

    def MTL_model(self):
        return super().MTL_model()
    
    def regression_model(self):
        return super().regression_model()




class CNNLSTM_regression_3_layer_model(CNNLSTM_model):
    def __init__(self, input_shape, HP=default_HP):
        super().__init__(input_shape, HP=HP)

    def HP_define(self):
        self.CNN_1_kernal = self._HP['CNN_1_kernal']
        self.CNN_1_RNN    = self._HP['CNN_1_RNN']
        self.CNN_2_kernal = self._HP['CNN_2_kernal']
        self.CNN_2_RNN    = self._HP['CNN_2_RNN']
        self.CNN_3_kernal = self._HP['CNN_3_kernal']
        self.CNN_3_RNN    = self._HP['CNN_3_RNN']
        #self.CNN_4_kernal = self._HP['CNN_4_kernal']
        #self.CNN_4_RNN    = self._HP['CNN_4_RNN']
        #self.CNN_5_kernal = self._HP['CNN_5_kernal']
        #self.CNN_5_RNN    = self._HP['CNN_5_RNN']
        self.DNN_1        = self._HP['DNN_1']
        self.DNN_2        = self._HP['DNN_2']
        self.DNN_rate     = self._HP['DNN_rate']
        self.CNN_dropout  = self._HP['CNN_dropout' ]
        self.DNN_dropout  = self._HP['DNN_dropout' ]
        self.learning_rate  =  self._HP['learning_rate']

    def model_define(self):
        #model define
        CNNLSTM_layer_1 = self.CNNLSTM_layer(self.input_data,
                                            kernal_size=self.CNN_1_kernal,
                                            RNN_cell_size=self.CNN_1_RNN,
                                            dropout=self.CNN_dropout)
        concatenate_1 = concatenate([self.input_data, CNNLSTM_layer_1])

        CNNLSTM_layer_2 = self.CNNLSTM_layer(self.input_data,
                                            kernal_size=self.CNN_2_kernal,
                                            RNN_cell_size=self.CNN_2_RNN,
                                            dropout=self.CNN_dropout)
        concatenate_2 = concatenate([CNNLSTM_layer_1, CNNLSTM_layer_2])

        CNNLSTM_layer_3 = self.CNNLSTM_layer(self.input_data,
                                            kernal_size=self.CNN_3_kernal,
                                            RNN_cell_size=self.CNN_3_RNN,
                                            dropout=self.CNN_dropout)
        concatenate_3  = concatenate([CNNLSTM_layer_2, CNNLSTM_layer_3])

        #CNNLSTM_layer_4 = self.CNNLSTM_layer(self.input_data,
        #                                    kernal_size=self.CNN_4_kernal,
        #                                    RNN_cell_size=self.CNN_4_RNN,
        #                                    dropout=self.CNN_dropout)
        #concatenate_4 = concatenate([CNNLSTM_layer_3, CNNLSTM_layer_4])
        #
        #CNNLSTM_layer_5 = self.CNNLSTM_layer(self.input_data,
        #                                    kernal_size=self.CNN_5_kernal,
        #                                    RNN_cell_size=self.CNN_5_RNN,
        #                                    dropout=self.CNN_dropout)
        #concatenate_5 = concatenate([CNNLSTM_layer_4, CNNLSTM_layer_5])

        LSTM_middle_1 = self.select_last_LSTM_cell(concatenate_1)
        LSTM_middle_2 = self.select_last_LSTM_cell(concatenate_2)
        self.LSTM_middle_final = self.select_last_LSTM_cell(concatenate_3)

        self.regression_layer_1 = self.regression_layer(LSTM_middle_1, layer_name='1')
        self.regression_layer_2 = self.regression_layer(LSTM_middle_2, layer_name='2')
        self.final_regression_layer = self.regression_layer(self.LSTM_middle_final, layer_name=None)

        self._regression_train_model = Model(self.input_data, [self.regression_layer_1, self.regression_layer_2, self.final_regression_layer])
        self._regression_model = Model(self.input_data, self.final_regression_layer)

        self._clustering_model = Model(self.input_data, self.LSTM_middle_final)

        return self._regression_train_model


    def model_compile(self):
        # compile
        optimizer = Adam(learning_rate = self.learning_rate)
        self.training_model.compile(optimizer=optimizer,
                                    loss={
                                        #'num_mis': 'categorical_crossentropy',
                                        'rate_1': 'mean_squared_error',
                                        'rate_2': 'mean_squared_error',
                                        'rate': 'mean_squared_error'},
                                    loss_weights={
                                        #'num_mis': 0.5,
                                        'rate_1': 1,
                                        'rate_2': 1,
                                        'rate': 1},
            )


    def regression_layer(self, layer_input, layer_name='3'):
        
        if layer_name is None:
            final_layer_named = 'rate'
        else:
            final_layer_named = 'rate_{}'.format(layer_name)


        DNN_rate_1_cell = self.DNN_rate
        DNN_rate_final_cell = 1

        dropout = self.DNN_dropout

        input_data = layer_input

        DNN_rate_layer_1 = Dense(DNN_rate_1_cell, activation='relu')(input_data)
        DNN_rate_layer_1 = Dropout(dropout)(DNN_rate_layer_1)

        DNN_rate_layer_2 = Dense(DNN_rate_final_cell, activation='linear', name=final_layer_named)(DNN_rate_layer_1)

        return DNN_rate_layer_2


    def regression_train_model(self):
        return self._regression_train_model
    
    def regression_model(self):
        return self._regression_model











class CNNLSTM_offtarget_model:
    def __init__(self,
    input_shape):
        ## TODO: Hyperparameter 넣는 방법 추가
        self.input_data = Input(shape=input_shape)
        CNNLSTM_input_reshape = Reshape((23,8))(self.input_data)
        
        #model define
        CNNLSTM_layer_1 = self.initial_CNNLSTM_layer(self.input_data)
        concatenate_1 = concatenate([CNNLSTM_input_reshape, CNNLSTM_layer_1])

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
        self._classification_model = Model(self.input_data, self.MTL_classification_layer)
        self._regression_model = Model(self.input_data, self.MTL_regression_layer)
        self._clustering_model = Model(self.input_data, self.LSTM_middle_final)

    def MTL_model(self,):
        return self._MTL_model

    def classification_model(self,):
        return self._classification_model

    def regression_model(self):
        return self._regression_model

    def clustering_model(self):
        return self._clustering_model

    def initial_CNNLSTM_layer(self, layer_input):
        kernal_size = 128
        RNN_cell_size = 256
        dropout = 0.5

        input_data = layer_input

        CNN_1 = Conv2D(kernal_size, kernel_size = (1,4), strides=(1,4), padding='same', activation='relu',kernel_initializer='he_normal')(input_data)
        CNN_1 = Dropout(dropout)(CNN_1)

        CNN_2 = Conv2D(kernal_size, kernel_size = (2,4), strides=(1,4), padding='same', activation='relu',kernel_initializer='he_normal')(input_data)
        CNN_2 = Dropout(dropout)(CNN_2)

        CNN_3 = Conv2D(kernal_size, kernel_size = (3,4), strides=(1,4), padding='same', activation='relu',kernel_initializer='he_normal')(input_data)
        CNN_3 = Dropout(dropout)(CNN_3)

        CNN_4 = Conv2D(kernal_size, kernel_size = (4,4), strides=(1,4), padding='same', activation='relu',kernel_initializer='he_normal')(input_data)
        CNN_4 = Dropout(dropout)(CNN_4)

        CNN_5 = Conv2D(kernal_size, kernel_size = (5,4), strides=(1,4), padding='same', activation='relu',kernel_initializer='he_normal')(input_data)
        CNN_5 = Dropout(dropout)(CNN_5)

        CNN_concatenate = concatenate([CNN_1, CNN_2, CNN_3, CNN_4, CNN_5])

        squeezed = Lambda(lambda x: tf.keras.backend.squeeze(x,axis=2))(CNN_concatenate)

        LSTM_layer = LSTM(RNN_cell_size, return_sequences=True)(squeezed)
        LSTM_layer = Dropout(dropout)(LSTM_layer)

        return LSTM_layer


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

        CNN_5 = Conv1D(kernal_size, [5], activation='relu', padding='same', kernel_initializer='he_normal')(input_data)
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
