import tensorflow as tf
import numpy as np

from data_import_preprocessing import import_data_preprocessing
from model_define import *


from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io


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



class CNNLSTM_SpCas9:
    def __init__(self, HP = default_HP, random_seed = 1234):
        # training and benchmark
        self.train_data_name = 'data/HT_Cas9_train.csv'
        self.test_data_name = 'data/HT_Cas9_test.csv'
        
        self.benchmark_data = 'data/Endo_Cas9.csv'

        # SpCas9 data
        SpCas9_data_preprocessing = import_data_preprocessing(train_data_file_name=self.train_data_name,
                                                               test_data_file_name=self.test_data_name,
                                                               RD_seed=random_seed
                                                               )
        self.SpCas9_data = SpCas9_data_preprocessing(sgRNA_column='Target context sequence',
                                                        indel_column='Background subtracted indel',
                                                        split_data=0.1)
        
        # Benchmark data
        Endo_data_preprocessing = import_data_preprocessing(train_data_file_name=self.benchmark_data,
                                                            RD_seed=random_seed
                                                            )
        self.bench_data_endo = Endo_data_preprocessing(sgRNA_column='30 bp target sequence (4 bp + 20 bp Protospacer + PAM + 3 bp)',
                                                        indel_column='Averge indel frequency (%)',
                                                        split_data=0
                                                        )

        
        # Data
        self.X_train        = self.SpCas9_data['train']['seq']
        self.class_train    = self.SpCas9_data['train']['indel_class']
        self.rate_train     = self.SpCas9_data['train']['indel_rate']

        self.X_val          = self.SpCas9_data['val']['seq']
        self.class_val      = self.SpCas9_data['val']['indel_class']
        self.rate_val       = self.SpCas9_data['val']['indel_rate']

        self.X_test         = self.SpCas9_data['test']['seq']
        self.class_test     = self.SpCas9_data['test']['indel_class']
        self.rate_test      = self.SpCas9_data['test']['indel_rate']

        self.X_total        = self.SpCas9_data['total']['seq']
        self.class_total    = self.SpCas9_data['total']['indel_class']
        self.rate_total     = self.SpCas9_data['total']['indel_rate']

        self.X_bench_endo    = self.bench_data_endo['total']['seq']
        self.class_bench_endo    = self.bench_data_endo['total']['indel_class']
        self.rate_bench_endo    = self.bench_data_endo['total']['indel_rate']


        self.input_shape = self.X_train.shape[1:]


        # Model
        self.CNNLSTM = CNNLSTM_model(input_shape=self.input_shape, HP = HP)
        self.CNNLSTM_MTL_model = self.CNNLSTM.MTL_model()
        self.CNNLSTM_regression_model = self.CNNLSTM.regression_model()

    def model_train(self, callback = [],verbose=1):

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        min_delta=0.0001,
                                                        patience=30, verbose=0, mode='auto')

        callbacks = [early_stopping] + callback

        self.CNNLSTM_MTL_model.fit(x=self.X_train,
                                    y={
                                        #'num_mis': num_mis_train,
                                        'class_1': self.class_train,
                                        'class_2': self.class_train,
                                        'class_final': self.class_train,
                                        'rate': self.rate_train},
                                    validation_data=(self.X_val, {#'num_mis': num_mis_val,
                                                            'class_1': self.class_val,
                                                            'class_2': self.class_val,
                                                            'class_final': self.class_val,
                                                            'rate': self.rate_val}),
                                    #class_weight={
                                    #    'class_1' : class_weights_dict,
                                    #    'class_2' : class_weights_dict,
                                    #    'class_final' : class_weights_dict},
                                    #sample_weight={'rate' : sample_weight},
                                    shuffle=True,
                                    epochs=200,
                                    batch_size=64,
                                    verbose=verbose,
                                    callbacks=callbacks)


    def model_evaluate(self,verbose=0):
        # model result
        class_result_acc = self.CNNLSTM_MTL_model.evaluate(self.X_test,
                                      {
                                          #'num_mis': num_mis_test,
                                          'class_1': self.class_test,
                                          'class_2': self.class_test,
                                          'class_final': self.class_test,
                                          'rate': self.rate_test},
                                      verbose=0
                                     )


        #print(MTL_model.metrics_names)
        #print(class_result_acc)
        result_label = self.CNNLSTM_MTL_model.metrics_names
        
        # loss
        label_index_loss = result_label.index('loss')
        label_index_rate_loss = result_label.index('rate_loss')
        total_loss = class_result_acc[label_index_loss]
        rate_loss = class_result_acc[label_index_rate_loss]

        # display evaluate
        if verbose == 1:
            for label, result in zip(result_label, class_result_acc):
                print('{:16} : {:>6.4f}'.format(label,result))
        
        # Correlation
        self.test_data_pearson, self.test_data_spearman = self.calc_correlation(
            model=self.CNNLSTM_regression_model,
            input_data=self.X_test,
            true_data=self.rate_test,
            verbose=0
        )
        self.bench_data_pearson, self.bench_data_spearman = self.calc_correlation(
            model=self.CNNLSTM_regression_model,
            input_data=self.X_bench_endo,
            true_data=self.rate_bench_endo,
            verbose=0
        )

        return {
            'total_loss' : total_loss, 'test_rate_loss' : rate_loss,
            'test_data_pearson' : self.test_data_pearson, 'test_data_spearman' : self.test_data_spearman,
            'bench_data_pearson' : self.bench_data_pearson, 'bench_data_spearman' : self.bench_data_spearman
            }



    def calc_correlation(self, model, input_data, true_data, verbose = 0):
        def nan_return_0(value):
            import math
            if math.isnan(value):
                return float(0)
            else:
                return value

        test_prediction = model.predict(input_data)
        test_prediction = np.array(test_prediction).reshape(-1,)
        test_true = true_data.reshape(-1,)
        
        #test_correlation = np.corrcoef(test_prediction, test_true).flatten()[1]
        pearson_corr = pearsonr(test_prediction, test_true)
        spearman_corr = spearmanr(test_prediction, test_true)

        pearson_corr_value = nan_return_0(pearson_corr[0])
        spearman_corr_value = nan_return_0(spearman_corr[0])


        if verbose == 1:
            #print('Correlation : {}'.format(test_correlation))
            print('Pearson Correlation : {}'.format(pearson_corr_value))
            print('Spearman Correlation : {}'.format(spearman_corr_value))
        return pearson_corr_value, spearman_corr_value


    def plot_scatter(self, prediction_data, true_data, plot_name='Test'):
        figure = plt.figure(figsize=(8,8))
        plt.title(plot_name, fontsize=20)
        plt.ylabel('Prediction', fontsize=15)
        plt.xlabel('True', fontsize=15)
        plt.scatter(x=true_data, y=prediction_data)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.grid(True)

        return figure

    def plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    def log_plot(self, epoch, model, input_data, true_data, plot, plot_name='Test scatter'):
        prediction = model.predict(input_data)
        if prediction.shape[1] >= 2:
            prediction = np.argmax(prediction, axis=1)
        else:
            prediction = prediction.reshape(-1,)

        figure = plot(prediction_data=prediction, true_data=true_data, plot_name=plot_name)
        fig_image = self.plot_to_image(figure)
        return fig_image
        #with self.file_writer_plot.as_default():
        #    tf.summary.image(plot_name, fig_image, step=epoch)




class CNNLSTM_short_SpCas9:
    def __init__(self, HP = default_HP, random_seed = 1234):
        # training and benchmark
        self.train_data_name = 'data/HT_Cas9_train.csv'
        self.test_data_name = 'data/HT_Cas9_test.csv'
        
        self.benchmark_data = 'data/Endo_Cas9.csv'

        # SpCas9 data
        SpCas9_data_preprocessing = import_data_preprocessing(train_data_file_name=self.train_data_name,
                                                               test_data_file_name=self.test_data_name,
                                                               RD_seed=random_seed
                                                               )
        self.SpCas9_data = SpCas9_data_preprocessing(sgRNA_column='Target context sequence',
                                                        indel_column='Background subtracted indel',
                                                        split_data=0.1)
        
        # Benchmark data
        Endo_data_preprocessing = import_data_preprocessing(train_data_file_name=self.benchmark_data,
                                                            RD_seed=random_seed
                                                            )
        self.bench_data_endo = Endo_data_preprocessing(sgRNA_column='30 bp target sequence (4 bp + 20 bp Protospacer + PAM + 3 bp)',
                                                        indel_column='Averge indel frequency (%)',
                                                        split_data=0
                                                        )

        
        # Data
        self.X_train        = self.SpCas9_data['train']['seq']
        self.class_train    = self.SpCas9_data['train']['indel_class']
        self.rate_train     = self.SpCas9_data['train']['indel_rate']

        self.X_val          = self.SpCas9_data['val']['seq']
        self.class_val      = self.SpCas9_data['val']['indel_class']
        self.rate_val       = self.SpCas9_data['val']['indel_rate']

        self.X_test         = self.SpCas9_data['test']['seq']
        self.class_test     = self.SpCas9_data['test']['indel_class']
        self.rate_test      = self.SpCas9_data['test']['indel_rate']

        self.X_total        = self.SpCas9_data['total']['seq']
        self.class_total    = self.SpCas9_data['total']['indel_class']
        self.rate_total     = self.SpCas9_data['total']['indel_rate']

        self.X_bench_endo    = self.bench_data_endo['total']['seq']
        self.class_bench_endo    = self.bench_data_endo['total']['indel_class']
        self.rate_bench_endo    = self.bench_data_endo['total']['indel_rate']


        self.input_shape = self.X_train.shape[1:]


        # Model
        self.CNNLSTM = CNNLSTM_short_model(input_shape=self.input_shape, HP = HP)
        self.CNNLSTM_MTL_model = self.CNNLSTM.MTL_model()
        self.CNNLSTM_regression_model = self.CNNLSTM.regression_model()

    def model_train(self, callback = [],verbose=1):

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        min_delta=0.001,
                                                        patience=30, verbose=0, mode='auto')

        callbacks = [early_stopping] + callback

        self.CNNLSTM_MTL_model.fit(x=self.X_train,
                                    y={
                                        #'num_mis': num_mis_train,
                                        'class_1': self.class_train,
                                        #'class_2': self.class_train,
                                        'class_final': self.class_train,
                                        'rate': self.rate_train},
                                    validation_data=(self.X_val, {#'num_mis': num_mis_val,
                                                            'class_1': self.class_val,
                                                            #'class_2': self.class_val,
                                                            'class_final': self.class_val,
                                                            'rate': self.rate_val}),
                                    #class_weight={
                                    #    'class_1' : class_weights_dict,
                                    #    'class_2' : class_weights_dict,
                                    #    'class_final' : class_weights_dict},
                                    #sample_weight={'rate' : sample_weight},
                                    shuffle=True,
                                    epochs=200,
                                    batch_size=64,
                                    verbose=verbose,
                                    callbacks=callbacks)


    def model_evaluate(self,verbose=0):
        # model result
        class_result_acc = self.CNNLSTM_MTL_model.evaluate(self.X_test,
                                      {
                                          #'num_mis': num_mis_test,
                                          'class_1': self.class_test,
                                          #'class_2': self.class_test,
                                          'class_final': self.class_test,
                                          'rate': self.rate_test},
                                      verbose=0
                                     )


        #print(MTL_model.metrics_names)
        #print(class_result_acc)
        result_label = self.CNNLSTM_MTL_model.metrics_names
        
        # loss
        label_index_loss = result_label.index('loss')
        label_index_rate_loss = result_label.index('rate_loss')
        total_loss = class_result_acc[label_index_loss]
        rate_loss = class_result_acc[label_index_rate_loss]

        # display evaluate
        if verbose == 1:
            for label, result in zip(result_label, class_result_acc):
                print('{:16} : {:>6.4f}'.format(label,result))
        
        # Correlation
        self.test_data_pearson, self.test_data_spearman = self.calc_correlation(
            model=self.CNNLSTM_regression_model,
            input_data=self.X_test,
            true_data=self.rate_test,
            verbose=0
        )
        self.bench_data_pearson, self.bench_data_spearman = self.calc_correlation(
            model=self.CNNLSTM_regression_model,
            input_data=self.X_bench_endo,
            true_data=self.rate_bench_endo,
            verbose=0
        )

        return {
            'total_loss' : total_loss, 'test_rate_loss' : rate_loss,
            'test_data_pearson' : self.test_data_pearson, 'test_data_spearman' : self.test_data_spearman,
            'bench_data_pearson' : self.bench_data_pearson, 'bench_data_spearman' : self.bench_data_spearman
            }



    def calc_correlation(self, model, input_data, true_data, verbose = 0):
        test_prediction = model.predict(input_data)
        test_prediction = np.array(test_prediction).reshape(-1,)
        test_true = true_data.reshape(-1,)
        
        #test_correlation = np.corrcoef(test_prediction, test_true).flatten()[1]
        pearson_corr = pearsonr(test_prediction, test_true)
        spearman_corr = spearmanr(test_prediction, test_true)
        if verbose == 1:
            #print('Correlation : {}'.format(test_correlation))
            print('Pearson Correlation : {}'.format(pearson_corr[0]))
            print('Spearman Correlation : {}'.format(spearman_corr[0]))
        return pearson_corr[0], spearman_corr[0]


    def plot_scatter(self, prediction_data, true_data, plot_name='Test'):
        figure = plt.figure(figsize=(8,8))
        plt.title(plot_name, fontsize=20)
        plt.ylabel('Prediction', fontsize=15)
        plt.xlabel('True', fontsize=15)
        plt.scatter(x=true_data, y=prediction_data)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.grid(True)

        return figure

    def plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    def log_plot(self, epoch, model, input_data, true_data, plot, plot_name='Test scatter'):
        prediction = model.predict(input_data)
        if prediction.shape[1] >= 2:
            prediction = np.argmax(prediction, axis=1)
        else:
            prediction = prediction.reshape(-1,)

        figure = plot(prediction_data=prediction, true_data=true_data, plot_name=plot_name)
        fig_image = self.plot_to_image(figure)
        return fig_image
        #with self.file_writer_plot.as_default():
        #    tf.summary.image(plot_name, fig_image, step=epoch)






class SpCas9_MODEL:
    def __init__(self, HP = default_HP, random_seed = 1234):

        # init
        self.__HP = HP
        self.__random_seed = random_seed
        self.epochs = 200

        # data
        self.Data()
        

        self.model = self.model_define(input_shape=self.input_shape, HP = self.__HP)
        self.__model = self.model()
        self.__model_regression = self.model.regression_model



    def Data(self):
        # training and benchmark
        self.train_data_name = 'data/HT_Cas9_train.csv'
        self.test_data_name = 'data/HT_Cas9_test.csv'
        
        self.benchmark_data = 'data/Endo_Cas9.csv'

        # SpCas9 data
        SpCas9_data_preprocessing = import_data_preprocessing(train_data_file_name=self.train_data_name,
                                                               test_data_file_name=self.test_data_name,
                                                               RD_seed=self.__random_seed
                                                               )
        self.SpCas9_data = SpCas9_data_preprocessing(sgRNA_column='Target context sequence',
                                                        indel_column='Background subtracted indel',
                                                        split_data=0.1)
        
        # Benchmark data
        Endo_data_preprocessing = import_data_preprocessing(train_data_file_name=self.benchmark_data,
                                                            RD_seed=self.__random_seed
                                                            )
        self.bench_data_endo = Endo_data_preprocessing(sgRNA_column='30 bp target sequence (4 bp + 20 bp Protospacer + PAM + 3 bp)',
                                                        indel_column='Averge indel frequency (%)',
                                                        split_data=0
                                                        )

        
        # Data
        self.X_train        = self.SpCas9_data['train']['seq']
        self.class_train    = self.SpCas9_data['train']['indel_class']
        self.rate_train     = self.SpCas9_data['train']['indel_rate']

        self.X_val          = self.SpCas9_data['val']['seq']
        self.class_val      = self.SpCas9_data['val']['indel_class']
        self.rate_val       = self.SpCas9_data['val']['indel_rate']

        self.X_test         = self.SpCas9_data['test']['seq']
        self.class_test     = self.SpCas9_data['test']['indel_class']
        self.rate_test      = self.SpCas9_data['test']['indel_rate']

        self.X_total        = self.SpCas9_data['total']['seq']
        self.class_total    = self.SpCas9_data['total']['indel_class']
        self.rate_total     = self.SpCas9_data['total']['indel_rate']

        self.X_bench_endo    = self.bench_data_endo['total']['seq']
        self.class_bench_endo    = self.bench_data_endo['total']['indel_class']
        self.rate_bench_endo    = self.bench_data_endo['total']['indel_rate']


        self.input_shape = self.X_train.shape[1:]



    def model_define(self,input_shape, HP):
        class model:
            def __init__(self, input_shape, HP):
                self.model = CNNLSTM_model(input_shape=input_shape, HP = HP)
                self.MTL_model = self.model.MTL_model()
                self.regression_model = self.model.regression_model()

            def __call__(self,):
                return self.MTL_model
        # Model
        _model = model(input_shape=input_shape, HP = HP)

        return _model


    def model_train(self, callback = [],verbose=1):

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        min_delta=0.001,
                                                        patience=30, verbose=0, mode='auto')

        callbacks = [early_stopping] + callback

        self.__model.fit(x=self.X_train,
                        y={
                            #'num_mis': num_mis_train,
                            'class_1': self.class_train,
                            'class_2': self.class_train,
                            'class_final': self.class_train,
                            'rate': self.rate_train},
                        validation_data=(self.X_val, {#'num_mis': num_mis_val,
                                                'class_1': self.class_val,
                                                'class_2': self.class_val,
                                                'class_final': self.class_val,
                                                'rate': self.rate_val}),
                        #class_weight={
                        #    'class_1' : class_weights_dict,
                        #    'class_2' : class_weights_dict,
                        #    'class_final' : class_weights_dict},
                        #sample_weight={'rate' : sample_weight},
                        shuffle=True,
                        epochs=self.epochs,
                        batch_size=64,
                        verbose=verbose,
                        callbacks=callbacks)


    def model_evaluate(self,verbose=0):
        # model result
        class_result_acc = self.__model.evaluate(self.X_test,
                                               {
                                                   #'num_mis': num_mis_test,
                                                   'class_1': self.class_test,
                                                   'class_2': self.class_test,
                                                   'class_final': self.class_test,
                                                   'rate': self.rate_test},
                                               verbose=0
                                                )


        #print(MTL_model.metrics_names)
        #print(class_result_acc)
        result_label = self.__model.metrics_names
        
        # loss
        label_index_loss = result_label.index('loss')
        label_index_rate_loss = result_label.index('rate_loss')
        total_loss = class_result_acc[label_index_loss]
        rate_loss = class_result_acc[label_index_rate_loss]

        # display evaluate
        if verbose == 1:
            for label, result in zip(result_label, class_result_acc):
                print('{:16} : {:>6.4f}'.format(label,result))
        
        # Correlation
        self.test_data_pearson, self.test_data_spearman = self.calc_correlation(
            model=self.__model_regression,
            input_data=self.X_test,
            true_data=self.rate_test,
            verbose=0
        )
        self.bench_data_pearson, self.bench_data_spearman = self.calc_correlation(
            model=self.__model_regression,
            input_data=self.X_bench_endo,
            true_data=self.rate_bench_endo,
            verbose=0
        )

        return {
            'total_loss' : total_loss, 'test_rate_loss' : rate_loss,
            'test_data_pearson' : self.test_data_pearson, 'test_data_spearman' : self.test_data_spearman,
            'bench_data_pearson' : self.bench_data_pearson, 'bench_data_spearman' : self.bench_data_spearman
            }



    def calc_correlation(self, model, input_data, true_data, verbose = 0):
        def nan_return_0(value):
            import math
            if math.isnan(value):
                return float(0)
            else:
                return value

        test_prediction = model.predict(input_data)
        test_prediction = np.array(test_prediction).reshape(-1,)
        test_true = true_data.reshape(-1,)
        
        #test_correlation = np.corrcoef(test_prediction, test_true).flatten()[1]
        pearson_corr = pearsonr(test_prediction, test_true)
        spearman_corr = spearmanr(test_prediction, test_true)

        pearson_corr_value = nan_return_0(pearson_corr[0])
        spearman_corr_value = nan_return_0(spearman_corr[0])


        if verbose == 1:
            #print('Correlation : {}'.format(test_correlation))
            print('Pearson Correlation : {}'.format(pearson_corr_value))
            print('Spearman Correlation : {}'.format(spearman_corr_value))
        return pearson_corr_value, spearman_corr_value

    def model_regression(self):
        return self.__model_regression

    '''
    def plot_scatter(self, prediction_data, true_data, plot_name='Test'):
        figure = plt.figure(figsize=(8,8))
        plt.title(plot_name, fontsize=20)
        plt.ylabel('Prediction', fontsize=15)
        plt.xlabel('True', fontsize=15)
        plt.scatter(x=true_data, y=prediction_data)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.grid(True)

        return figure
    
    def plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    def log_plot(self, epoch, model, input_data, true_data, plot, plot_name='Test scatter'):
        prediction = model.predict(input_data)
        if prediction.shape[1] >= 2:
            prediction = np.argmax(prediction, axis=1)
        else:
            prediction = prediction.reshape(-1,)

        figure = plot(prediction_data=prediction, true_data=true_data, plot_name=plot_name)
        fig_image = self.plot_to_image(figure)
        return fig_image
        #with self.file_writer_plot.as_default():
        #    tf.summary.image(plot_name, fig_image, step=epoch)
    '''




class CNNLSTM_3_layer_SpCas9(SpCas9_MODEL):
    def __init__(self, HP=default_HP, random_seed=1234):
        super().__init__(HP=HP, random_seed=random_seed)

    def Data(self):
        return super().Data()

    def model_define(self, input_shape, HP):
        class model:
            def __init__(self, input_shape, HP):
                self.model = CNNLSTM_3_layer_model(input_shape=input_shape, HP = HP)
                self.MTL_model = self.model.MTL_model()
                self.regression_model = self.model.regression_model()

            def __call__(self):
                return self.MTL_model
        # Model
        _model = model(input_shape=input_shape, HP = HP)

        return _model

    def model_train(self, callback=[], verbose=1):
        return super().model_train(callback=callback, verbose=verbose)

    def model_evaluate(self, verbose=0):
        return super().model_evaluate(verbose=verbose)

    def calc_correlation(self, model, input_data, true_data, verbose=0):
        return super().calc_correlation(model, input_data, true_data, verbose=verbose)