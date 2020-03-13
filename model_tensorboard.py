import tensorflow as tf
#import tensorboard
from tensorboard.plugins.hparams import api as hp

from datetime import datetime
import matplotlib.pyplot as plt
import io

from HP_tunning.plot import *
from HP_tunning.my_tensorboard import HP_tunning_tensorboard_callback
from Model import *




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


class CNNLSTM_tensorboard:
    def __init__(self, HP_dict, log_dir = 'tensorboard/', GPU=1, random_seed = 1234, verbose=1):
        time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        GPU_stamp = '_GPU_{}'.format(GPU)
        self.log_dir =  log_dir + time_stamp + GPU_stamp
        self.verbose = verbose

        # HP
        self.HP_dict = HP_dict


        self.CNN_1_kernal = hp.HParam('CNN_1_kernal',   hp.IntInterval(1, 1024))
        self.CNN_1_RNN    = hp.HParam('CNN_1_RNN',      hp.IntInterval(1, 1024))
        self.CNN_2_kernal = hp.HParam('CNN_2_kernal',   hp.IntInterval(1, 1024))
        self.CNN_2_RNN    = hp.HParam('CNN_2_RNN',      hp.IntInterval(1, 1024))
        self.CNN_3_kernal = hp.HParam('CNN_3_kernal',   hp.IntInterval(1, 1024))
        self.CNN_3_RNN    = hp.HParam('CNN_3_RNN',      hp.IntInterval(1, 1024))
        self.CNN_4_kernal = hp.HParam('CNN_4_kernal',   hp.IntInterval(1, 1024))
        self.CNN_4_RNN    = hp.HParam('CNN_4_RNN',      hp.IntInterval(1, 1024))
        self.CNN_5_kernal = hp.HParam('CNN_5_kernal',   hp.IntInterval(1, 1024))
        self.CNN_5_RNN    = hp.HParam('CNN_5_RNN',      hp.IntInterval(1, 1024))
        self.DNN_1        = hp.HParam('DNN_1',          hp.IntInterval(1, 1024))
        self.DNN_2        = hp.HParam('DNN_2',          hp.IntInterval(1, 1024))
        self.DNN_rate     = hp.HParam('DNN_rate',       hp.IntInterval(1, 1024))
        self.CNN_dropout  = hp.HParam('CNN_dropout',    hp.RealInterval(0.001, 0.5))
        self.DNN_dropout  = hp.HParam('DNN_dropout',    hp.RealInterval(0.001, 0.5))
        self.learning_rate  = hp.HParam('learning_rate',hp.RealInterval(1e-6, 1.0))


        """
        self.CNN_1_kernal = HP_dict['CNN_1_kernal']
        self.CNN_1_RNN    = HP_dict['CNN_1_RNN']
        self.CNN_2_kernal = HP_dict['CNN_2_kernal']
        self.CNN_2_RNN    = HP_dict['CNN_2_RNN']
        self.CNN_3_kernal = HP_dict['CNN_3_kernal']
        self.CNN_3_RNN    = HP_dict['CNN_3_RNN']
        self.CNN_4_kernal = HP_dict['CNN_4_kernal']
        self.CNN_4_RNN    = HP_dict['CNN_4_RNN']
        self.CNN_5_kernal = HP_dict['CNN_5_kernal']
        self.CNN_5_RNN    = HP_dict['CNN_5_RNN']
        self.DNN_1        = HP_dict['DNN_1']
        self.DNN_2        = HP_dict['DNN_2']
        self.DNN_rate     = HP_dict['DNN_rate']
        self.CNN_dropout  = HP_dict['CNN_dropout' ]
        self.DNN_dropout  = HP_dict['DNN_dropout' ]
        self.learning_rate  =  HP_dict['learning_rate']
        """

        self.Metric_name_loss = 'rate_loss_test'
        self.Metric_name_test_pearson = 'pearson_correlation_test'
        self.Metric_name_test_spearman = 'spearman_correlation_test'
        self.Metric_name_bench_pearson = 'pearson_correlation_bench'
        self.Metric_name_bench_spearman = 'spearman_correlation_bench'

        with tf.summary.create_file_writer(self.log_dir + '/hp_tunning').as_default():
            hp.hparams_config(
                hparams=[self.CNN_1_kernal, self.CNN_1_RNN,
                         self.CNN_2_kernal, self.CNN_2_RNN,
                         self.CNN_3_kernal, self.CNN_3_RNN,
                         self.CNN_4_kernal, self.CNN_4_RNN,
                         self.CNN_5_kernal, self.CNN_5_RNN,
                         self.DNN_1, self.DNN_2, self.DNN_rate,
                         self.CNN_dropout, self.DNN_dropout,
                         self.learning_rate],
                metrics=[hp.Metric(self.Metric_name_loss, display_name='Loss'),
                         hp.Metric(self.Metric_name_test_pearson, display_name='Pearson Correlation - Test data'),
                         hp.Metric(self.Metric_name_test_spearman, display_name='Spearman Correlation - Test data'),
                         hp.Metric(self.Metric_name_bench_pearson, display_name='Pearson Correlation - Bench data'),
                         hp.Metric(self.Metric_name_bench_spearman, display_name='Spearman Correlation - Bench data')]
            )

        

        # Model
        self.model = CNNLSTM_SpCas9(
            HP= HP_dict,
            random_seed = random_seed
        )

        self.data = self.model.SpCas9_data
        self.bench_data = self.model.bench_data_endo


        # Callbacks
        test_input = self.data['test']['seq']
        test_indel_true = self.data['test']['indel_rate']
        bench_input = self.bench_data['total']['seq']
        bench_indel_true = self.bench_data['total']['indel_rate']
        
        self.callbacks = HP_tunning_tensorboard_callback(log_dir=self.log_dir)
        
        test_data_correlation_plot = self.callbacks.callback_log_plot(
            model = self.model.CNNLSTM_regression_model,
            input_data = test_input,
            true_data = test_indel_true,
            plot = plot_scatter,
            plot_name = 'Test data correlation'
        )

        bench_data_correlation_plot = self.callbacks.callback_log_plot(
            model = self.model.CNNLSTM_regression_model,
            input_data = bench_input,
            true_data = bench_indel_true,
            plot = plot_scatter,
            plot_name = 'Endo data correlation'
        )

        callbacks_plot = [test_data_correlation_plot, bench_data_correlation_plot]
        self.callbacks_list = self.callbacks.TB_callbacks(plot_callbacks = callbacks_plot)




    def training(self,):
        self.model.model_train(
            callback= self.callbacks_list,
            verbose=self.verbose
        )
        
        pass

    def evaluate(self,):
        model_evaluate = self.model.model_evaluate(verbose=0)
        
        self.evaluate_total_loss        = model_evaluate['total_loss']
        self.evaluate_test_loss         = model_evaluate['test_rate_loss']
        self.evaluate_test_pearson      = model_evaluate['test_data_pearson']
        self.evaluate_test_spearman     = model_evaluate['test_data_spearman']
        self.evaluate_bench_pearson     = model_evaluate['bench_data_pearson']
        self.evaluate_bench_spearman    = model_evaluate['bench_data_spearman']
        #print(model_evaluate)
        self.H_param(
            loss= self.evaluate_test_loss,
            pearson= self.evaluate_test_pearson,
            spearman= self.evaluate_test_spearman,
            bench_pearson= self.evaluate_bench_pearson,
            bench_spearman= self.evaluate_bench_spearman,
            step=1
        )
        pass


    
    def H_param(self, loss, pearson, spearman, bench_pearson=None, bench_spearman=None,step=3):
        hparams = {
            'CNN_1_kernal'  : self.HP_dict['CNN_1_kernal'] ,
            'CNN_1_RNN'     : self.HP_dict['CNN_1_RNN']    ,
            'CNN_2_kernal'  : self.HP_dict['CNN_2_kernal'] ,    
            'CNN_2_RNN'     : self.HP_dict['CNN_2_RNN']    ,
            'CNN_3_kernal'  : self.HP_dict['CNN_3_kernal'] ,    
            'CNN_3_RNN'     : self.HP_dict['CNN_3_RNN']    ,
            'CNN_4_kernal'  : self.HP_dict['CNN_4_kernal'] ,    
            'CNN_4_RNN'     : self.HP_dict['CNN_4_RNN']    ,
            'CNN_5_kernal'  : self.HP_dict['CNN_5_kernal'] ,    
            'CNN_5_RNN'     : self.HP_dict['CNN_5_RNN']    ,
            'DNN_1'         : self.HP_dict['DNN_1']        ,
            'DNN_2'         : self.HP_dict['DNN_2']        ,
            'DNN_rate'      : self.HP_dict['DNN_rate']     ,
            'CNN_dropout'   : self.HP_dict['CNN_dropout']  ,    
            'DNN_dropout'   : self.HP_dict['DNN_dropout']  ,
            'learning_rate' : self.HP_dict['learning_rate'],
        }
        
        with tf.summary.create_file_writer(self.log_dir + '/hp_tunning').as_default():
            hp.hparams(hparams=hparams)
            #print(loss, pearson, spearman)
            tf.summary.scalar(self.Metric_name_loss, loss, step=step)
            tf.summary.scalar(self.Metric_name_test_pearson, pearson, step=step)
            tf.summary.scalar(self.Metric_name_test_spearman, spearman, step=step)
            if bench_pearson is not None:
                tf.summary.scalar(self.Metric_name_bench_pearson, bench_pearson, step=step)
                tf.summary.scalar(self.Metric_name_bench_spearman, bench_spearman, step=step)

    

class CNNLSTM_short_tensorboard:
    def __init__(self, HP_dict, log_dir = 'tensorboard/', GPU=1, random_seed = 1234, verbose=1):
        time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        GPU_stamp = '_GPU_{}'.format(GPU)
        self.log_dir =  log_dir + time_stamp + GPU_stamp
        self.verbose = verbose

        # HP
        self.HP_dict = HP_dict


        self.CNN_1_kernal = hp.HParam('CNN_1_kernal',   hp.IntInterval(1, 1024))
        self.CNN_1_RNN    = hp.HParam('CNN_1_RNN',      hp.IntInterval(1, 1024))
        self.CNN_2_kernal = hp.HParam('CNN_2_kernal',   hp.IntInterval(1, 1024))
        self.CNN_2_RNN    = hp.HParam('CNN_2_RNN',      hp.IntInterval(1, 1024))
        #self.CNN_3_kernal = hp.HParam('CNN_3_kernal',   hp.IntInterval(1, 1024))
        #self.CNN_3_RNN    = hp.HParam('CNN_3_RNN',      hp.IntInterval(1, 1024))
        #self.CNN_4_kernal = hp.HParam('CNN_4_kernal',   hp.IntInterval(1, 1024))
        #self.CNN_4_RNN    = hp.HParam('CNN_4_RNN',      hp.IntInterval(1, 1024))
        #self.CNN_5_kernal = hp.HParam('CNN_5_kernal',   hp.IntInterval(1, 1024))
        #self.CNN_5_RNN    = hp.HParam('CNN_5_RNN',      hp.IntInterval(1, 1024))
        self.DNN_1        = hp.HParam('DNN_1',          hp.IntInterval(1, 1024))
        self.DNN_2        = hp.HParam('DNN_2',          hp.IntInterval(1, 1024))
        self.DNN_rate     = hp.HParam('DNN_rate',       hp.IntInterval(1, 1024))
        self.CNN_dropout  = hp.HParam('CNN_dropout',    hp.RealInterval(0.001, 0.5))
        self.DNN_dropout  = hp.HParam('DNN_dropout',    hp.RealInterval(0.001, 0.5))
        self.learning_rate  = hp.HParam('learning_rate',hp.RealInterval(1e-6, 1.0))


        """
        self.CNN_1_kernal = HP_dict['CNN_1_kernal']
        self.CNN_1_RNN    = HP_dict['CNN_1_RNN']
        self.CNN_2_kernal = HP_dict['CNN_2_kernal']
        self.CNN_2_RNN    = HP_dict['CNN_2_RNN']
        self.CNN_3_kernal = HP_dict['CNN_3_kernal']
        self.CNN_3_RNN    = HP_dict['CNN_3_RNN']
        self.CNN_4_kernal = HP_dict['CNN_4_kernal']
        self.CNN_4_RNN    = HP_dict['CNN_4_RNN']
        self.CNN_5_kernal = HP_dict['CNN_5_kernal']
        self.CNN_5_RNN    = HP_dict['CNN_5_RNN']
        self.DNN_1        = HP_dict['DNN_1']
        self.DNN_2        = HP_dict['DNN_2']
        self.DNN_rate     = HP_dict['DNN_rate']
        self.CNN_dropout  = HP_dict['CNN_dropout' ]
        self.DNN_dropout  = HP_dict['DNN_dropout' ]
        self.learning_rate  =  HP_dict['learning_rate']
        """

        self.Metric_name_loss = 'loss'
        self.Metric_name_test_pearson = 'pearson_correlation_test'
        self.Metric_name_test_spearman = 'spearman_correlation_test'
        self.Metric_name_bench_pearson = 'pearson_correlation_bench'
        self.Metric_name_bench_spearman = 'spearman_correlation_bench'

        with tf.summary.create_file_writer(self.log_dir + '/hp_tunning').as_default():
            hp.hparams_config(
                hparams=[self.CNN_1_kernal, self.CNN_1_RNN,
                         self.CNN_2_kernal, self.CNN_2_RNN,
                         #self.CNN_3_kernal, self.CNN_3_RNN,
                         #self.CNN_4_kernal, self.CNN_4_RNN,
                         #self.CNN_5_kernal, self.CNN_5_RNN,
                         self.DNN_1, self.DNN_2, self.DNN_rate,
                         self.CNN_dropout, self.DNN_dropout,
                         self.learning_rate],
                metrics=[hp.Metric(self.Metric_name_loss, display_name='Loss'),
                         hp.Metric(self.Metric_name_test_pearson, display_name='Pearson Correlation - Test data'),
                         hp.Metric(self.Metric_name_test_spearman, display_name='Spearman Correlation - Test data'),
                         hp.Metric(self.Metric_name_bench_pearson, display_name='Pearson Correlation - Bench data'),
                         hp.Metric(self.Metric_name_bench_spearman, display_name='Spearman Correlation - Bench data')]
            )

        

        # Model
        self.model = CNNLSTM_short_SpCas9(
            HP= HP_dict,
            random_seed = random_seed
        )

        self.data = self.model.SpCas9_data
        self.bench_data = self.model.bench_data_endo


        # Callbacks
        test_input = self.data['test']['seq']
        test_indel_true = self.data['test']['indel_rate']
        bench_input = self.bench_data['total']['seq']
        bench_indel_true = self.bench_data['total']['indel_rate']
        
        self.callbacks = HP_tunning_tensorboard_callback(log_dir=self.log_dir)
        
        test_data_correlation_plot = self.callbacks.callback_log_plot(
            model = self.model.CNNLSTM_regression_model,
            input_data = test_input,
            true_data = test_indel_true,
            plot = plot_scatter,
            plot_name = 'Test data correlation'
        )

        bench_data_correlation_plot = self.callbacks.callback_log_plot(
            model = self.model.CNNLSTM_regression_model,
            input_data = bench_input,
            true_data = bench_indel_true,
            plot = plot_scatter,
            plot_name = 'Endo data correlation'
        )

        callbacks_plot = [test_data_correlation_plot, bench_data_correlation_plot]
        self.callbacks_list = self.callbacks.TB_callbacks(plot_callbacks = callbacks_plot)




    def training(self,):
        self.model.model_train(
            callback= self.callbacks_list,
            verbose=self.verbose
        )
        
        pass

    def evaluate(self,):
        model_evaluate = self.model.model_evaluate(verbose=0)
        
        self.evaluate_total_loss        = model_evaluate['total_loss']
        self.evaluate_val_loss          = model_evaluate['test_rate_loss']
        self.evaluate_test_pearson      = model_evaluate['test_data_pearson']
        self.evaluate_test_spearman     = model_evaluate['test_data_spearman']
        self.evaluate_bench_pearson     = model_evaluate['bench_data_pearson']
        self.evaluate_bench_spearman    = model_evaluate['bench_data_spearman']
        #print(model_evaluate)
        self.H_param(
            loss= self.evaluate_val_loss,
            pearson= self.evaluate_test_pearson,
            spearman= self.evaluate_test_spearman,
            bench_pearson= self.evaluate_bench_pearson,
            bench_spearman= self.evaluate_bench_spearman,
            step=1
        )
        pass


    
    def H_param(self, loss, pearson, spearman, bench_pearson=None, bench_spearman=None,step=3):
        hparams = {
            'CNN_1_kernal'  : self.HP_dict['CNN_1_kernal'] ,
            'CNN_1_RNN'     : self.HP_dict['CNN_1_RNN']    ,
            'CNN_2_kernal'  : self.HP_dict['CNN_2_kernal'] ,    
            'CNN_2_RNN'     : self.HP_dict['CNN_2_RNN']    ,
            #'CNN_3_kernal'  : self.HP_dict['CNN_3_kernal'] ,    
            #'CNN_3_RNN'     : self.HP_dict['CNN_3_RNN']    ,
            #'CNN_4_kernal'  : self.HP_dict['CNN_4_kernal'] ,    
            #'CNN_4_RNN'     : self.HP_dict['CNN_4_RNN']    ,
            #'CNN_5_kernal'  : self.HP_dict['CNN_5_kernal'] ,    
            #'CNN_5_RNN'     : self.HP_dict['CNN_5_RNN']    ,
            'DNN_1'         : self.HP_dict['DNN_1']        ,
            'DNN_2'         : self.HP_dict['DNN_2']        ,
            'DNN_rate'      : self.HP_dict['DNN_rate']     ,
            'CNN_dropout'   : self.HP_dict['CNN_dropout']  ,    
            'DNN_dropout'   : self.HP_dict['DNN_dropout']  ,
            'learning_rate' : self.HP_dict['learning_rate'],
        }
        
        with tf.summary.create_file_writer(self.log_dir + '/hp_tunning').as_default():
            hp.hparams(hparams=hparams)
            #print(loss, pearson, spearman)
            tf.summary.scalar(self.Metric_name_loss, loss, step=step)
            tf.summary.scalar(self.Metric_name_test_pearson, pearson, step=step)
            tf.summary.scalar(self.Metric_name_test_spearman, spearman, step=step)
            if bench_pearson is not None:
                tf.summary.scalar(self.Metric_name_bench_pearson, bench_pearson, step=step)
                tf.summary.scalar(self.Metric_name_bench_spearman, bench_spearman, step=step)

    


class SpCas9_tensorboard:
    def __init__(self, HP_dict, log_dir = 'tensorboard/', GPU=1, random_seed = 1234, verbose=1):
        time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        GPU_stamp = '_GPU_{}'.format(GPU)
        self.log_dir =  log_dir + time_stamp + GPU_stamp
        self.verbose = verbose

        self.__HP_dict = HP_dict
        self.__random_seed = random_seed


        self.TB_HParam_define()
        self.Model_define()
        self.callbacks()




    def TB_HParam_define(self):

        # HP
        self.__CNN_1_kernal = hp.HParam('CNN_1_kernal',   hp.IntInterval(1, 1024))
        self.__CNN_1_RNN    = hp.HParam('CNN_1_RNN',      hp.IntInterval(1, 1024))
        self.__CNN_2_kernal = hp.HParam('CNN_2_kernal',   hp.IntInterval(1, 1024))
        self.__CNN_2_RNN    = hp.HParam('CNN_2_RNN',      hp.IntInterval(1, 1024))
        self.__CNN_3_kernal = hp.HParam('CNN_3_kernal',   hp.IntInterval(1, 1024))
        self.__CNN_3_RNN    = hp.HParam('CNN_3_RNN',      hp.IntInterval(1, 1024))
        self.__CNN_4_kernal = hp.HParam('CNN_4_kernal',   hp.IntInterval(1, 1024))
        self.__CNN_4_RNN    = hp.HParam('CNN_4_RNN',      hp.IntInterval(1, 1024))
        self.__CNN_5_kernal = hp.HParam('CNN_5_kernal',   hp.IntInterval(1, 1024))
        self.__CNN_5_RNN    = hp.HParam('CNN_5_RNN',      hp.IntInterval(1, 1024))
        self.__DNN_1        = hp.HParam('DNN_1',          hp.IntInterval(1, 1024))
        self.__DNN_2        = hp.HParam('DNN_2',          hp.IntInterval(1, 1024))
        self.__DNN_rate     = hp.HParam('DNN_rate',       hp.IntInterval(1, 1024))
        self.__CNN_dropout  = hp.HParam('CNN_dropout',    hp.RealInterval(0.001, 0.5))
        self.__DNN_dropout  = hp.HParam('DNN_dropout',    hp.RealInterval(0.001, 0.5))
        self.__learning_rate  = hp.HParam('learning_rate',hp.RealInterval(1e-6, 1.0))


        """
        self.CNN_1_kernal = HP_dict['CNN_1_kernal']
        self.CNN_1_RNN    = HP_dict['CNN_1_RNN']
        self.CNN_2_kernal = HP_dict['CNN_2_kernal']
        self.CNN_2_RNN    = HP_dict['CNN_2_RNN']
        self.CNN_3_kernal = HP_dict['CNN_3_kernal']
        self.CNN_3_RNN    = HP_dict['CNN_3_RNN']
        self.CNN_4_kernal = HP_dict['CNN_4_kernal']
        self.CNN_4_RNN    = HP_dict['CNN_4_RNN']
        self.CNN_5_kernal = HP_dict['CNN_5_kernal']
        self.CNN_5_RNN    = HP_dict['CNN_5_RNN']
        self.DNN_1        = HP_dict['DNN_1']
        self.DNN_2        = HP_dict['DNN_2']
        self.DNN_rate     = HP_dict['DNN_rate']
        self.CNN_dropout  = HP_dict['CNN_dropout' ]
        self.DNN_dropout  = HP_dict['DNN_dropout' ]
        self.learning_rate  =  HP_dict['learning_rate']
        """

        self.__Metric_name_loss = 'rate_loss_test'
        self.__Metric_name_test_pearson = 'pearson_correlation_test'
        self.__Metric_name_test_spearman = 'spearman_correlation_test'
        self.__Metric_name_bench_pearson = 'pearson_correlation_bench'
        self.__Metric_name_bench_spearman = 'spearman_correlation_bench'

        with tf.summary.create_file_writer(self.log_dir + '/hp_tunning').as_default():
            hp.hparams_config(
                hparams=[self.__CNN_1_kernal, self.__CNN_1_RNN,
                         self.__CNN_2_kernal, self.__CNN_2_RNN,
                         self.__CNN_3_kernal, self.__CNN_3_RNN,
                         self.__CNN_4_kernal, self.__CNN_4_RNN,
                         self.__CNN_5_kernal, self.__CNN_5_RNN,
                         self.__DNN_1, self.__DNN_2, self.__DNN_rate,
                         self.__CNN_dropout, self.__DNN_dropout,
                         self.__learning_rate],
                metrics=[hp.Metric(self.__Metric_name_loss, display_name='Loss'),
                         hp.Metric(self.__Metric_name_test_pearson, display_name='Pearson Correlation - Test data'),
                         hp.Metric(self.__Metric_name_test_spearman, display_name='Spearman Correlation - Test data'),
                         hp.Metric(self.__Metric_name_bench_pearson, display_name='Pearson Correlation - Bench data'),
                         hp.Metric(self.__Metric_name_bench_spearman, display_name='Spearman Correlation - Bench data')]
            )

        
    def Model_define(self):
        # Model
        self._model_define = SpCas9_MODEL(
            HP= self.__HP_dict,
            random_seed = self.__random_seed
        )
        self.model = self._model_define.model
        self.model_regression = self._model_define.model_regression()

        # Data
        self.data = self._model_define.SpCas9_data
        self.bench_data = self._model_define.bench_data_endo


    def callbacks(self):
        # Callbacks
        test_input = self.data['test']['seq']
        test_indel_true = self.data['test']['indel_rate']
        bench_input = self.bench_data['total']['seq']
        bench_indel_true = self.bench_data['total']['indel_rate']
        
        self.__callbacks = HP_tunning_tensorboard_callback(log_dir=self.log_dir)
        
        test_data_correlation_plot = self.__callbacks.callback_log_plot(
            model = self.model_regression,
            input_data = test_input,
            true_data = test_indel_true,
            plot = plot_scatter,
            plot_name = 'Test data correlation'
        )

        bench_data_correlation_plot = self.__callbacks.callback_log_plot(
            model = self.model_regression,
            input_data = bench_input,
            true_data = bench_indel_true,
            plot = plot_scatter,
            plot_name = 'Endo data correlation'
        )

        callbacks_plot = [test_data_correlation_plot, bench_data_correlation_plot]
        self.__callbacks_list = self.__callbacks.TB_callbacks(plot_callbacks = callbacks_plot)




    def training(self,):
        self._model_define.model_train(
            callback= self.__callbacks_list,
            verbose=self.verbose
        )
        
        pass

    def evaluate(self,):
        model_evaluate = self._model_define.model_evaluate(verbose=0)
        
        self.evaluate_total_loss        = model_evaluate['total_loss']
        self.evaluate_test_loss         = model_evaluate['test_rate_loss']
        self.evaluate_test_pearson      = model_evaluate['test_data_pearson']
        self.evaluate_test_spearman     = model_evaluate['test_data_spearman']
        self.evaluate_bench_pearson     = model_evaluate['bench_data_pearson']
        self.evaluate_bench_spearman    = model_evaluate['bench_data_spearman']
        #print(model_evaluate)
        self.__H_param(
            loss= self.evaluate_test_loss,
            pearson= self.evaluate_test_pearson,
            spearman= self.evaluate_test_spearman,
            bench_pearson= self.evaluate_bench_pearson,
            bench_spearman= self.evaluate_bench_spearman,
            step=1
        )
        pass


    
    def __H_param(self, loss, pearson, spearman, bench_pearson=None, bench_spearman=None,step=3):
        hparams = {
            'CNN_1_kernal'  : self.__HP_dict['CNN_1_kernal'] ,
            'CNN_1_RNN'     : self.__HP_dict['CNN_1_RNN']    ,
            'CNN_2_kernal'  : self.__HP_dict['CNN_2_kernal'] ,    
            'CNN_2_RNN'     : self.__HP_dict['CNN_2_RNN']    ,
            'CNN_3_kernal'  : self.__HP_dict['CNN_3_kernal'] ,    
            'CNN_3_RNN'     : self.__HP_dict['CNN_3_RNN']    ,
            'CNN_4_kernal'  : self.__HP_dict['CNN_4_kernal'] ,    
            'CNN_4_RNN'     : self.__HP_dict['CNN_4_RNN']    ,
            'CNN_5_kernal'  : self.__HP_dict['CNN_5_kernal'] ,    
            'CNN_5_RNN'     : self.__HP_dict['CNN_5_RNN']    ,
            'DNN_1'         : self.__HP_dict['DNN_1']        ,
            'DNN_2'         : self.__HP_dict['DNN_2']        ,
            'DNN_rate'      : self.__HP_dict['DNN_rate']     ,
            'CNN_dropout'   : self.__HP_dict['CNN_dropout']  ,    
            'DNN_dropout'   : self.__HP_dict['DNN_dropout']  ,
            'learning_rate' : self.__HP_dict['learning_rate'],
        }
        
        with tf.summary.create_file_writer(self.log_dir + '/hp_tunning').as_default():
            hp.hparams(hparams=hparams)
            #print(loss, pearson, spearman)
            tf.summary.scalar(self.__Metric_name_loss, loss, step=step)
            tf.summary.scalar(self.__Metric_name_test_pearson, pearson, step=step)
            tf.summary.scalar(self.__Metric_name_test_spearman, spearman, step=step)
            if bench_pearson is not None:
                tf.summary.scalar(self.__Metric_name_bench_pearson, bench_pearson, step=step)
                tf.summary.scalar(self.__Metric_name_bench_spearman, bench_spearman, step=step)

    