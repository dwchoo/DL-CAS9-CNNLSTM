import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import skopt
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence, plot_objective, plot_evaluations
from skopt.utils import use_named_args

from multiprocessing import Pool, Process
import multiprocessing

from Model import CNNLSTM_SpCas9
from model_tensorboard import CNNLSTM_tensorboard



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


class CNNLSTM_HP_tunning:
    def __init__(self, default_HP_dict, log_dir='tensorboard_log/'):
        self.log_dir = log_dir
        
        self.CNN_1_kernal   = Integer(low=8, high=1024, name= 'CNN_1_kernal_node')
        self.CNN_1_RNN      = Integer(low=8, high=1024, name= 'CNN_1_RNN_node')
        self.CNN_2_kernal   = Integer(low=8, high=1024, name= 'CNN_2_kernal_node')
        self.CNN_2_RNN      = Integer(low=8, high=1024, name= 'CNN_2_RNN_node')
        self.CNN_3_kernal   = Integer(low=8, high=1024, name= 'CNN_3_kernal_node')
        self.CNN_3_RNN      = Integer(low=8, high=1024, name= 'CNN_3_RNN_node')
        self.CNN_4_kernal   = Integer(low=8, high=1024, name= 'CNN_4_kernal_node')
        self.CNN_4_RNN      = Integer(low=8, high=1024, name= 'CNN_4_RNN_node')
        self.CNN_5_kernal   = Integer(low=8, high=1024, name= 'CNN_5_kernal_node')
        self.CNN_5_RNN      = Integer(low=8, high=1024, name= 'CNN_5_RNN_node')
        self.DNN_1          = Integer(low=8, high=1024, name= 'DNN_1_node')
        self.DNN_2          = Integer(low=8, high=1024, name= 'DNN_2_node')
        self.DNN_rate       = Integer(low=8, high=1024, name= 'DNN_rate_node')
        self.CNN_dropout    = Real(low=0.001, high= 0.6, name= 'CNN_dropout_node')
        self.DNN_dropout    = Real(low=0.001, high= 0.6, name= 'DNN_dropout_node')
        self.learning_rate  = Real(low=1e-7, high=0.1, prior='log-uniform', name='Learning_rate_node')
        
        self.dimension_HP = [
                            self.CNN_1_kernal ,
                            self.CNN_1_RNN    ,
                            self.CNN_2_kernal ,
                            self.CNN_2_RNN    ,
                            self.CNN_3_kernal ,
                            self.CNN_3_RNN    ,
                            self.CNN_4_kernal ,
                            self.CNN_4_RNN    ,
                            self.CNN_5_kernal ,
                            self.CNN_5_RNN    ,
                            self.DNN_1        ,
                            self.DNN_2        ,
                            self.DNN_rate     ,
                            self.CNN_dropout  ,
                            self.DNN_dropout  ,
                            self.learning_rate,
                        ]
        
        self.default_HP_list = list(default_HP_dict.values())
    
    
    def tunning_start(self, n_random_start = 50, n_cell=100, x0=None):
        if x0 is None:
            x0 = self.default_HP_list
        elif type(x0) is dict:
            x0 = list(x0.values())

        self.gp_fitting = gp_minimize(
            func=self.tunning_func,
            dimensions=self.dimension_HP,
            n_calls=n_cell,
            n_random_starts=n_random_start,
            acq_func='EI',
            x0=x0
        )
        
    def tunning_result_plot(self,log_dir = None, file_format='pdf'):
        if log_dir is None:
            log_dir = self.log_dir
        if file_format is not 'pdf' and file_format is not 'png':
            print('file formate error')
            print("Choose 'pdf' or 'png'")
            return None
        evaluation_plot = plot_evaluations(self.gp_fitting)
        objective_plot = plot_objective(self.gp_fitting)

        evaluation_plot.flatten()[0].figure.savefig(
            '{}evaluation_plot.{}'.format(log_dir, file_format),
            bbox_inches='tight'
        )
        plt.close()
        objective_plot.flatten()[0].figure.savefig(
            '{}objective_plot.{}'.format(log_dir, file_format),
            bbox_inches='tight'
        )
        plt.close()


    def tunning_func(self,HP_list):
        _tunning_model = lambda _HP_list : self.tunning_model(
                                                            HP_list=_HP_list,
                                                            GPU=1,
                                                            random_seed=1234,
                                                            log_dir=self.log_dir
                                                            )

        return _tunning_model(HP_list)

    def tunning_model(self, HP_list, GPU = 1, random_seed =1234 ,log_dir='tensorboard_log/'):
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            gpu_num = GPU
            try:
                tf.config.experimental.set_visible_devices(gpus[gpu_num], 'GPU')
                #tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[gpu_num],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])
            except RuntimeError as e:
                print(e)

        HP_list2dict = {
            'CNN_1_kernal'  : int(HP_list[0]) ,
            'CNN_1_RNN'     : int(HP_list[1]) ,
            'CNN_2_kernal'  : int(HP_list[2]) ,
            'CNN_2_RNN'     : int(HP_list[3]) ,
            'CNN_3_kernal'  : int(HP_list[4]) ,
            'CNN_3_RNN'     : int(HP_list[5]) ,
            'CNN_4_kernal'  : int(HP_list[6]) ,
            'CNN_4_RNN'     : int(HP_list[7]) ,
            'CNN_5_kernal'  : int(HP_list[8]) ,
            'CNN_5_RNN'     : int(HP_list[9]) ,

            'DNN_1'         : int(HP_list[10]),
            'DNN_2'         : int(HP_list[11]),
            'DNN_rate'      : int(HP_list[12]),

            'CNN_dropout'   : float(HP_list[13]),
            'DNN_dropout'   : float(HP_list[14]),

            'learning_rate' : float(HP_list[15])
        }

        print("======================start trainning=======================")
        print(HP_list2dict.items())

        self.model_tensorboard = CNNLSTM_tensorboard(
            HP_dict=HP_list2dict,
            log_dir=log_dir,
            random_seed=random_seed
        )
        self.model_tensorboard.training()
        self.model_tensorboard.evaluate()
        val_loss = self.model_tensorboard.evaluate_val_loss

        return val_loss

