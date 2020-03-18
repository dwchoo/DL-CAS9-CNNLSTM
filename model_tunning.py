import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime

import skopt
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence, plot_objective, plot_evaluations
from skopt.utils import use_named_args

from multiprocessing import Pool, Process, Lock
import multiprocessing

from Model import *
from model_tensorboard import *
from variable_init import *



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

        self.change_HP_scale_bool = True        
        self.CNN_1_kernal   = Integer(low=0, high=10, name= 'CNN_1_kernal_node')
        self.CNN_1_RNN      = Integer(low=0, high=10, name= 'CNN_1_RNN_node')
        self.CNN_2_kernal   = Integer(low=0, high=10, name= 'CNN_2_kernal_node')
        self.CNN_2_RNN      = Integer(low=0, high=10, name= 'CNN_2_RNN_node')
        self.CNN_3_kernal   = Integer(low=0, high=10, name= 'CNN_3_kernal_node')
        self.CNN_3_RNN      = Integer(low=0, high=10, name= 'CNN_3_RNN_node')
        self.CNN_4_kernal   = Integer(low=0, high=10, name= 'CNN_4_kernal_node')
        self.CNN_4_RNN      = Integer(low=0, high=10, name= 'CNN_4_RNN_node')
        self.CNN_5_kernal   = Integer(low=0, high=10, name= 'CNN_5_kernal_node')
        self.CNN_5_RNN      = Integer(low=0, high=10, name= 'CNN_5_RNN_node')
        self.DNN_1          = Integer(low=0, high=10, name= 'DNN_1_node')
        self.DNN_2          = Integer(low=0, high=10, name= 'DNN_2_node')
        self.DNN_rate       = Integer(low=0, high=10, name= 'DNN_rate_node')
        self.CNN_dropout    = Real(low=0.001, high= 0.5, name= 'CNN_dropout_node')
        self.DNN_dropout    = Real(low=0.001, high= 0.5, name= 'DNN_dropout_node')
        self.learning_rate  = Real(low=1e-6, high=0.1, prior='log-uniform', name='Learning_rate_node')
        
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
    
    def default_HP_result(self,HP_dict):
        _default_HP = list(HP_dict.values())
        _default_HP_loss = self.tunning_func_multiprocs(_default_HP)
        
    
    
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

        tunning_result_HP = self.gp_fitting.x
        tunning_result_loss = self.gp_fitting.fun
        self.tunning_result_single_case = {'HP' : tunning_result_HP, 'loss' : tunning_result_loss}

    def tunning_start_multiprocs(self, n_random_start=50, n_cell=100, x0=None,
                                case_num=3, log_dir='tensorboard_log/', GPU_list = [0,1,2]
                                ):

        start_time_stamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        print("Start time : {}".format(start_time_stamp))
        
        self.now_cell = 1
        self.totoal_cell = n_cell
        
        if x0 is None:
            x0 = self.default_HP_list
        elif type(x0) is dict:
            x0 = list(x0.values())
        
        self.tunning_func_initialize_multiprocs(
            case_num = case_num,
            log_dir=log_dir,
            GPU_list = GPU_list
        )

        self.gp_fitting_multiprocs = gp_minimize(
            func= self.tunning_func_multiprocs,
            dimensions=self.dimension_HP,
            n_calls=n_cell,
            n_random_starts=n_random_start,
            acq_func='EI',
            x0=x0
        )

        end_time_stamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        print("Finish time : {}".format(end_time_stamp))


        tunning_result_HP = self.gp_fitting_multiprocs.x
        tunning_result_loss = self.gp_fitting_multiprocs.fun
        self.tunning_result_multiprocs = {'HP' : tunning_result_HP, 'loss' : tunning_result_loss}


    def multiprocessing_process(self, HP_list, case_num=3, log_dir='tensorboard_log/', GPU_list = [0,1,2]):
        
        random_seed_list = [random.randrange(1000,9999) for i in range(case_num)]
        GPU_num = len(GPU_list)
        GPU_work_list = [GPU_list[i%GPU_num] for i in range(case_num)]

        CASE_zip = zip(random_seed_list, GPU_work_list)

        #Multi-Process
        lock = Lock()
        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        procs = []
        for index, (random_seed, GPU) in enumerate(CASE_zip):
            proc = Process(target=self.tunning_model,
                            args=(HP_list, GPU, random_seed, log_dir,
                                  lock, index, return_dict),
                            name='{}_case_{}_GPU'.format(index, GPU)
                            )
            procs.append(proc)
            proc.start()

            if index % GPU_num == (GPU_num-1):
                proc.join()
        
        for proc in procs:
            proc.join()

        results = return_dict.values()
        result_val_loss_list        = [results[i]['val_loss'        ] for i in range(case_num)]
        result_test_pearson_list    = [results[i]['test_pearson'    ] for i in range(case_num)]
        result_test_spearman_list   = [results[i]['test_spearman'   ] for i in range(case_num)]
        result_bench_pearson_list   = [results[i]['bench_pearson'   ] for i in range(case_num)]
        result_bench_spearman_list  = [results[i]['bench_spearman'  ] for i in range(case_num)]

        result_dict = {
            'val_loss'      : result_val_loss_list       ,
            'test_pearson'  : result_test_pearson_list   ,
            'test_spearman' : result_test_spearman_list  ,
            'bench_pearson' : result_bench_pearson_list  ,
            'bench_spearman': result_bench_spearman_list ,

        }

        return result_dict

    def evaluate_result_mean(self, result_list, calc_max = True,weight = 2):
        def maxORmin(_list):
            if calc_max is True:
                return max(_list)
            else:
                return min(_list)
        _value = maxORmin(result_list)
        total_count = len(result_list) + weight - 1
        if weight == 1:
            result_mean = sum(result_list)/total_count
        else:
            result_mean = sum(result_list + (int(weight) - 1) * [_value])/total_count
        return result_mean

    def tunning_func_initialize_multiprocs(self,
            case_num = 3,
            log_dir='tensorboard_log/',
            GPU_list = [0,1,2],
            max_weight = 2
        ):
        # initialize paramters
        self.now_cell = 1
        self._case_num = case_num
        self._log_dir = log_dir
        self._GPU_list = GPU_list
        self._max_weight = max_weight


    def tunning_func_multiprocs(self, HP_list):
        print("====================== start trainning =======================")
        print("progress : {}/{}".format(self.now_cell, self.totoal_cell))

        HP_list2dict = self.change_HP_scale(
                HP_list=HP_list,
                changed_scale=self.change_HP_scale_bool,
                show_items=True
                )
       
        try:
            multiprocessing_result = self.multiprocessing_process(
                HP_list= HP_list,
                case_num= self._case_num,
                log_dir= self._log_dir,
                GPU_list= self._GPU_list
            )
            evaluate_val_loss_mean = self.evaluate_result_mean(
                result_list= multiprocessing_result['val_loss'],
                calc_max=True,
                weight= self._max_weight
            )

            evaluate_test_pearson_mean = self.evaluate_result_mean(
                result_list= multiprocessing_result['test_pearson'],
                calc_max=False,
                weight= self._max_weight
            )
            evaluate_test_spearman_mean = self.evaluate_result_mean(
                result_list= multiprocessing_result['test_spearman'],
                calc_max=False,
                weight= self._max_weight
            )
            evaluate_bench_pearson_mean = self.evaluate_result_mean(
                result_list= multiprocessing_result['bench_pearson'],
                calc_max=False,
                weight= self._max_weight
            )
            evaluate_bench_spearman_mean = self.evaluate_result_mean(
                result_list= multiprocessing_result['bench_spearman'],
                calc_max=False,
                weight= self._max_weight
            )
            self.now_cell += 1
            print("### Mean of results ###")
            self.display_model_result(
                val_loss        = evaluate_val_loss_mean,
                test_pearson    = evaluate_test_pearson_mean,
                test_spearman   = evaluate_test_spearman_mean,
                bench_pearson   = evaluate_bench_pearson_mean,
                bench_spearman  = evaluate_bench_spearman_mean,
            )
            print("##############################################################")


            return evaluate_val_loss_mean
        except:
            print("'multiprocessing_process' parameters are not initialized")
            print("run 'tunning_func_initialize_multiprocs'")
        


    def tunning_func(self,HP_list):
        _tunning_model = lambda _HP_list : self.tunning_model(
                                                            HP_list=_HP_list,
                                                            GPU=1,
                                                            random_seed=1234,
                                                            log_dir=self.log_dir
                                                            )

        return _tunning_model(HP_list)

    def tunning_model(self, HP_list, GPU = 1, random_seed =1234 ,log_dir='tensorboard_log/',
                      lock=None, proc_num = None, return_dict=None):
        
        #if lock is not None:
            #lock.acquire()
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                gpu_num = GPU
                #print('GPU number : {}'.format(GPU))
                try:
                    tf.config.experimental.set_visible_devices(gpus[gpu_num], 'GPU')
                    #tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
                    tf.config.experimental.set_virtual_device_configuration(
                        gpus[gpu_num],
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])
                except RuntimeError as e:
                    print(e)

            HP_list2dict = self.change_HP_scale(
                HP_list=HP_list,
                changed_scale=self.change_HP_scale_bool,
                show_items=False
                )

            #print("====================== start trainning =======================")
            #print(HP_list2dict.items())

            self.model_tensorboard = CNNLSTM_tensorboard(
                HP_dict=HP_list2dict,
                log_dir=log_dir,
                GPU=GPU,
                random_seed=random_seed,
                verbose=0
            )
            self.model_tensorboard.training()
            self.model_tensorboard.evaluate()
            val_loss = self.model_tensorboard.evaluate_test_loss
            test_pearson    = self.model_tensorboard.evaluate_test_pearson
            test_spearman   = self.model_tensorboard.evaluate_test_spearman
            bench_pearson   = self.model_tensorboard.evaluate_bench_pearson
            bench_spearman  = self.model_tensorboard.evaluate_bench_spearman

            result_dict = {
                'val_loss'      : val_loss,
                'test_pearson'  : test_pearson,
                'test_spearman' : test_spearman,
                'bench_pearson' : bench_pearson,
                'bench_spearman': bench_spearman,
            }
            self.display_model_result(
                val_loss        = result_dict['val_loss'],
                test_pearson    = result_dict['test_pearson'],
                test_spearman   = result_dict['test_spearman'],
                bench_pearson   = result_dict['bench_pearson'],
                bench_spearman  = result_dict['bench_spearman'],
            )

            if proc_num is not None and return_dict is not None:
                return_dict[proc_num] = result_dict
            return val_loss
        
        finally:
            pass
            #if lock is not None:
                #lock.release()

        
    def display_model_result(self,
            val_loss,
            test_pearson,
            test_spearman,
            bench_pearson,
            bench_spearman
        ):
        print('{:8} : {:>6.5f}    {:12} : {:>6.5f}    {:12} : {:>6.5f}    {:12} : {:>6.5f}    {:12} : {:>6.5f}'.format(
                'val_loss', val_loss,
                'Test Pearson', test_pearson,
                'Test Spearman', test_spearman,
                'Bench Pearson', bench_pearson,
                'Bench Spearman', bench_spearman,
        ))

    def show_best_point(self,):
        #HP_list = self.tunning_result_multiprocs['HP']
        try:
            HP_list = self.gp_fitting.x
        except AttributeError:
            HP_list = self.gp_fitting_multiprocs.x

        HP_list2dict = self.change_HP_scale(
            HP_list=HP_list,
            changed_scale=self.change_HP_scale_bool,
            show_items=False)

        for name, HP in HP_list2dict.items():
            if type(HP) == float:
                print("{:13} : {:>4.7f}".format(name, HP))
            else:
                print("{:13} : {:>4}".format(name, HP))



    def tunning_result_plot(self,log_dir = None, file_format='pdf'):
        if log_dir is None:
            log_dir = self.log_dir
        if file_format is not 'pdf' and file_format is not 'png':
            print('file formate error')
            print("Choose 'pdf' or 'png'")
            return None
        
        try:
            gp_fitting = self.gp_fitting
        except AttributeError:
            gp_fitting = self.gp_fitting_multiprocs
            
        evaluation_plot = plot_evaluations(gp_fitting)
        objective_plot = plot_objective(gp_fitting)

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
        print("Saved result plot at {}".format(log_dir))


    def change_HP_scale(self, HP_list, changed_scale = False, show_items = False):
        if changed_scale is False:
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
        else:
            HP_list2dict = {
                'CNN_1_kernal'  : int(2**HP_list[0]) ,
                'CNN_1_RNN'     : int(2**HP_list[1]) ,
                'CNN_2_kernal'  : int(2**HP_list[2]) ,
                'CNN_2_RNN'     : int(2**HP_list[3]) ,
                'CNN_3_kernal'  : int(2**HP_list[4]) ,
                'CNN_3_RNN'     : int(2**HP_list[5]) ,
                'CNN_4_kernal'  : int(2**HP_list[6]) ,
                'CNN_4_RNN'     : int(2**HP_list[7]) ,
                'CNN_5_kernal'  : int(2**HP_list[8]) ,
                'CNN_5_RNN'     : int(2**HP_list[9]) ,

                'DNN_1'         : int(2**HP_list[10]),
                'DNN_2'         : int(2**HP_list[11]),
                'DNN_rate'      : int(2**HP_list[12]),

                'CNN_dropout'   : float(HP_list[13]),
                'DNN_dropout'   : float(HP_list[14]),

                'learning_rate' : float(HP_list[15])
            }


        if show_items is True:
            print(HP_list2dict.items())

        return HP_list2dict








class SpCas9_HP_tunning:
    def __init__(self,
        default_HP_dict,
        log_dir='tensorboard_log/',
        multi_case_tunning = True
    ):
        self._default_HP = default_HP_dict
        self.log_dir = log_dir
        self._multi_case_tunning = multi_case_tunning
        self._model_epoch_control = 200
        self._model_verbose = 0

        self.HP_dimension_define()



    def HP_dimension_define(self,):

        self._change_HP_scale_bool = True        
        self._CNN_1_kernal   = Integer(low=0, high=10, name= 'CNN_1_kernal_node')
        self._CNN_1_RNN      = Integer(low=0, high=10, name= 'CNN_1_RNN_node')
        self._CNN_2_kernal   = Integer(low=0, high=10, name= 'CNN_2_kernal_node')
        self._CNN_2_RNN      = Integer(low=0, high=10, name= 'CNN_2_RNN_node')
        self._CNN_3_kernal   = Integer(low=0, high=10, name= 'CNN_3_kernal_node')
        self._CNN_3_RNN      = Integer(low=0, high=10, name= 'CNN_3_RNN_node')
        self._CNN_4_kernal   = Integer(low=0, high=10, name= 'CNN_4_kernal_node')
        self._CNN_4_RNN      = Integer(low=0, high=10, name= 'CNN_4_RNN_node')
        self._CNN_5_kernal   = Integer(low=0, high=10, name= 'CNN_5_kernal_node')
        self._CNN_5_RNN      = Integer(low=0, high=10, name= 'CNN_5_RNN_node')
        self._DNN_1          = Integer(low=0, high=10, name= 'DNN_1_node')
        self._DNN_2          = Integer(low=0, high=10, name= 'DNN_2_node')
        self._DNN_rate       = Integer(low=0, high=10, name= 'DNN_rate_node')
        self._CNN_dropout    = Real(low=0.001, high= 0.5, name= 'CNN_dropout_node')
        self._DNN_dropout    = Real(low=0.001, high= 0.5, name= 'DNN_dropout_node')
        self._learning_rate  = Real(low=1e-6, high=0.1, prior='log-uniform', name='Learning_rate_node')
        
        self.dimension_HP = [
                            self._CNN_1_kernal ,
                            self._CNN_1_RNN    ,
                            self._CNN_2_kernal ,
                            self._CNN_2_RNN    ,
                            self._CNN_3_kernal ,
                            self._CNN_3_RNN    ,
                            self._CNN_4_kernal ,
                            self._CNN_4_RNN    ,
                            self._CNN_5_kernal ,
                            self._CNN_5_RNN    ,
                            self._DNN_1        ,
                            self._DNN_2        ,
                            self._DNN_rate     ,
                            self._CNN_dropout  ,
                            self._DNN_dropout  ,
                            self._learning_rate,
                        ]

        self._default_HP_list = list(self._default_HP.values())



    def model_tensorboard_define(self, HP_list2dict, log_dir, GPU, random_seed, verobse):
        _tensorboard_model = SpCas9_tensorboard(
            HP_dict=HP_list2dict,
            log_dir=log_dir,
            GPU=GPU,
            random_seed=random_seed,
            verbose=verobse
        )
        ###
        _tensorboard_model._model_define.epochs = self._model_epoch_control

        _model = _tensorboard_model.model

        _tensorboard_model.training()
        _tensorboard_model.evaluate()

        model_result = _tensorboard_model.model_evaluation

        return _model, model_result


    
    def tunning_model(self, HP_list, GPU = 1, random_seed =1234 ,log_dir='tensorboard_log/',
                      lock=None, proc_num = None, return_dict=None):
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            gpu_num = GPU
            #print('GPU number : {}'.format(GPU))
            try:
                tf.config.experimental.set_visible_devices(gpus[gpu_num], 'GPU')
                #tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[gpu_num],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])
            except RuntimeError as e:
                print(e)

        HP_list2dict = self.change_HP_scale(
            HP_list=HP_list,
            changed_scale=self._change_HP_scale_bool,
            show_items=False
            )

        self.model, self.results = self.model_tensorboard_define(
            HP_list2dict = HP_list2dict,
            log_dir      = log_dir,
            GPU          = GPU,
            random_seed  = random_seed,
            verobse      = self._model_verbose,
        )

        self.display_model_result(
            test_loss       = self.results.test_rate_loss,
            test_pearson    = self.results.test_pearson  ,
            test_spearman   = self.results.test_spearman ,
            bench_pearson   = self.results.bench_pearson ,
            bench_spearman  = self.results.bench_spearman,
        )

        if proc_num is not None and return_dict is not None:
            return_dict[proc_num] = self.results

        test_rate_loss = self.results.test_rate_loss

        return test_rate_loss


    def default_HP_result(self,HP_dict):
        _default_HP = list(HP_dict.values())
        _default_HP_loss = self.tunning_func_multiprocs(_default_HP)
        return _default_HP_loss
        
    
    
    def tunning_start(self, n_random_start=50, n_cell=100, x0=None,
                      case_num=3, log_dir='tensorboard_log/', GPU_list = [0,1,2]
                     ):
        if x0 is None:
            x0 = self._default_HP_list
        elif type(x0) is dict:
            x0 = list(x0.values())
        
        start_time_stamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        print("Start time : {}".format(start_time_stamp))

        self._now_cell = 1
        self._totoal_cell = n_cell

        if self._multi_case_tunning is False:
            self.gp_fitting = gp_minimize(
                func=self.tunning_func,
                dimensions=self.dimension_HP,
                n_calls=n_cell,
                n_random_starts=n_random_start,
                acq_func='EI',
                x0=x0
            )
            tunning_result_HP = self.gp_fitting.x
            tunning_result_loss = self.gp_fitting.fun
            self.tunning_result = {'HP' : tunning_result_HP, 'loss' : tunning_result_loss}
        else:
            self.tunning_func_initialize_multiprocs(
                case_num = case_num,
                log_dir=log_dir,
                GPU_list = GPU_list
            )
            self.gp_fitting = gp_minimize(
                func= self.tunning_func_multiprocs,
                dimensions=self.dimension_HP,
                n_calls=n_cell,
                n_random_starts=n_random_start,
                acq_func='EI',
                x0=x0
            )
            tunning_result_HP = self.gp_fitting.x
            tunning_result_loss = self.gp_fitting.fun
        

        end_time_stamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        print("Finish time : {}".format(end_time_stamp))


        self.tunning_result = {'HP' : tunning_result_HP, 'loss' : tunning_result_loss}
    


    def multiprocessing_process(self, HP_list, case_num=3, log_dir='tensorboard_log/', GPU_list = [0,1,2]):
        
        random_seed_list = [random.randrange(1000,9999) for i in range(case_num)]
        GPU_num = len(GPU_list)
        GPU_work_list = [GPU_list[i%GPU_num] for i in range(case_num)]

        CASE_zip = zip(random_seed_list, GPU_work_list)

        #Multi-Process
        lock = Lock()
        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        procs = []
        for index, (random_seed, GPU) in enumerate(CASE_zip):
            proc = Process(target=self.tunning_model,
                            args=(HP_list, GPU, random_seed, log_dir,
                                  lock, index, return_dict),
                            name='{}_case_{}_GPU'.format(index, GPU)
                            )
            procs.append(proc)
            proc.start()

            if index % GPU_num == (GPU_num-1):
                proc.join()
        
        for proc in procs:
            proc.join()

        results = return_dict.values()
        result_test_rate_loss_list  = [results[i].test_rate_loss  for i in range(case_num)]
        result_test_pearson_list    = [results[i].test_pearson    for i in range(case_num)]
        result_test_spearman_list   = [results[i].test_spearman   for i in range(case_num)]
        result_bench_pearson_list   = [results[i].bench_pearson   for i in range(case_num)]
        result_bench_spearman_list  = [results[i].bench_spearman  for i in range(case_num)]

        results_list = model_results()
        results_list.test_rate_loss = result_test_rate_loss_list
        results_list.correlation_init(
            test_pearson    = result_test_pearson_list  ,
            test_spearman   = result_test_spearman_list ,
            bench_pearson   = result_bench_pearson_list ,
            bench_spearman  = result_bench_spearman_list,
        )
    
        return results_list


    def evaluate_result_mean(self, result_list, calc_max = True,weight = 2):
        def maxORmin(_list):
            if calc_max is True:
                return max(_list)
            else:
                return min(_list)
        _value = maxORmin(result_list)
        total_count = len(result_list) + weight - 1
        if weight == 1:
            result_mean = sum(result_list)/total_count
        else:
            result_mean = sum(result_list + (int(weight) - 1) * [_value])/total_count
        return result_mean

    def tunning_func_initialize_multiprocs(self,
            case_num = 3,
            log_dir='tensorboard_log/',
            GPU_list = [0,1,2],
            max_weight = 2
        ):
        # initialize paramters
        self._now_cell = 1
        self._case_num = case_num
        self._log_dir = log_dir
        self._GPU_list = GPU_list
        self._max_weight = max_weight


    def tunning_func_multiprocs(self, HP_list):
        print("====================== start trainning =======================")
        print("progress : {}/{}".format(self._now_cell, self._totoal_cell))
        
        #print(HP_list)

        HP_list2dict = self.change_HP_scale(
                HP_list=HP_list,
                changed_scale=self._change_HP_scale_bool,
                show_items=True
                )
       
        try:
            multiprocessing_result = self.multiprocessing_process(
                HP_list= HP_list,
                case_num= self._case_num,
                log_dir= self._log_dir,
                GPU_list= self._GPU_list
            )
            evaluate_test_loss_mean = self.evaluate_result_mean(
                result_list= multiprocessing_result.test_rate_loss,
                calc_max=True,
                weight= self._max_weight
            )

            evaluate_test_pearson_mean = self.evaluate_result_mean(
                result_list= multiprocessing_result.test_pearson,
                calc_max=False,
                weight= self._max_weight
            )
            evaluate_test_spearman_mean = self.evaluate_result_mean(
                result_list= multiprocessing_result.test_spearman,
                calc_max=False,
                weight= self._max_weight
            )
            evaluate_bench_pearson_mean = self.evaluate_result_mean(
                result_list= multiprocessing_result.bench_pearson,
                calc_max=False,
                weight= self._max_weight
            )
            evaluate_bench_spearman_mean = self.evaluate_result_mean(
                result_list= multiprocessing_result.bench_spearman,
                calc_max=False,
                weight= self._max_weight
            )

            self._now_cell += 1

            print("### Mean of results ###")
            self.display_model_result(
                test_loss       = evaluate_test_loss_mean,
                test_pearson    = evaluate_test_pearson_mean,
                test_spearman   = evaluate_test_spearman_mean,
                bench_pearson   = evaluate_bench_pearson_mean,
                bench_spearman  = evaluate_bench_spearman_mean,
            )
            print("##############################################################")


            return evaluate_test_loss_mean
        except:
            print("'multiprocessing_process' parameters are not initialized")
            print("run 'tunning_func_initialize_multiprocs'")
        


    def tunning_func(self,HP_list):
        print("====================== start trainning =======================")
        print("progress : {}/{}".format(self._now_cell, self._totoal_cell))

        _tunning_model = lambda _HP_list : self.tunning_model(
                                                            HP_list=_HP_list,
                                                            GPU=1,
                                                            random_seed=1234,
                                                            log_dir=self.log_dir
                                                            )
        
        self._now_cell += 1
        print("##############################################################")

        return _tunning_model(HP_list)

    
    

        
    def display_model_result(self,
            test_loss,
            test_pearson,
            test_spearman,
            bench_pearson,
            bench_spearman
        ):
        print('{:8} : {:>6.5f}    {:12} : {:>6.5f}    {:12} : {:>6.5f}    {:12} : {:>6.5f}    {:12} : {:>6.5f}'.format(
                'test_rate_loss', test_loss,
                'Test Pearson', test_pearson,
                'Test Spearman', test_spearman,
                'Bench Pearson', bench_pearson,
                'Bench Spearman', bench_spearman,
        ))

    def show_best_point(self,):
        #HP_list = self.tunning_result_multiprocs['HP']
        HP_list = self.gp_fitting.x

        HP_list2dict = self.change_HP_scale(
            HP_list=HP_list,
            changed_scale=self._change_HP_scale_bool,
            show_items=False)

        for name, HP in HP_list2dict.items():
            if type(HP) == float:
                print("{:13} : {:>4.7f}".format(name, HP))
            else:
                print("{:13} : {:>4}".format(name, HP))



    def tunning_result_plot(self,log_dir = None, file_format='pdf'):
        if log_dir is None:
            log_dir = self.log_dir
        if file_format is not 'pdf' and file_format is not 'png':
            print('file formate error')
            print("Choose 'pdf' or 'png'")
            return None

        gp_fitting = self.gp_fitting
            
        evaluation_plot = plot_evaluations(gp_fitting)
        objective_plot = plot_objective(gp_fitting)

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
        print("Saved result plot at {}".format(log_dir))


    def change_HP_scale(self, HP_list, changed_scale = False, show_items = False):
        if changed_scale is False:
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
        else:
            HP_list2dict = {
                'CNN_1_kernal'  : int(2**HP_list[0]) ,
                'CNN_1_RNN'     : int(2**HP_list[1]) ,
                'CNN_2_kernal'  : int(2**HP_list[2]) ,
                'CNN_2_RNN'     : int(2**HP_list[3]) ,
                'CNN_3_kernal'  : int(2**HP_list[4]) ,
                'CNN_3_RNN'     : int(2**HP_list[5]) ,
                'CNN_4_kernal'  : int(2**HP_list[6]) ,
                'CNN_4_RNN'     : int(2**HP_list[7]) ,
                'CNN_5_kernal'  : int(2**HP_list[8]) ,
                'CNN_5_RNN'     : int(2**HP_list[9]) ,

                'DNN_1'         : int(2**HP_list[10]),
                'DNN_2'         : int(2**HP_list[11]),
                'DNN_rate'      : int(2**HP_list[12]),

                'CNN_dropout'   : float(HP_list[13]),
                'DNN_dropout'   : float(HP_list[14]),

                'learning_rate' : float(HP_list[15])
            }


        if show_items is True:
            print(HP_list2dict.items())

        return HP_list2dict



class CNNLSTM_3_layer_HP_tunning(SpCas9_HP_tunning):
    def __init__(self, default_HP_dict, log_dir='tensorboard_log/', multi_case_tunning=True):
        super().__init__(default_HP_dict, log_dir=log_dir, multi_case_tunning=multi_case_tunning)

    def HP_dimension_define(self):

        self._change_HP_scale_bool = True

        self._CNN_1_kernal   = Integer(low=0, high=10, name= 'CNN_1_kernal_node')
        self._CNN_1_RNN      = Integer(low=0, high=10, name= 'CNN_1_RNN_node')
        self._CNN_2_kernal   = Integer(low=0, high=10, name= 'CNN_2_kernal_node')
        self._CNN_2_RNN      = Integer(low=0, high=10, name= 'CNN_2_RNN_node')
        self._CNN_3_kernal   = Integer(low=0, high=10, name= 'CNN_3_kernal_node')
        self._CNN_3_RNN      = Integer(low=0, high=10, name= 'CNN_3_RNN_node')
        self._DNN_1          = Integer(low=0, high=10, name= 'DNN_1_node')
        self._DNN_2          = Integer(low=0, high=10, name= 'DNN_2_node')
        self._DNN_rate       = Integer(low=0, high=10, name= 'DNN_rate_node')
        self._CNN_dropout    = Real(low=0.001, high= 0.5, name= 'CNN_dropout_node')
        self._DNN_dropout    = Real(low=0.001, high= 0.5, name= 'DNN_dropout_node')
        self._learning_rate  = Real(low=1e-6, high=0.1, prior='log-uniform', name='Learning_rate_node')
        
        self.dimension_HP = [
                            self._CNN_1_kernal ,
                            self._CNN_1_RNN    ,
                            self._CNN_2_kernal ,
                            self._CNN_2_RNN    ,
                            self._CNN_3_kernal ,
                            self._CNN_3_RNN    ,
                            self._DNN_1        ,
                            self._DNN_2        ,
                            self._DNN_rate     ,
                            self._CNN_dropout  ,
                            self._DNN_dropout  ,
                            self._learning_rate,
                        ]

        self._default_HP_list = list(self._default_HP.values())


    def model_tensorboard_define(self, HP_list2dict, log_dir, GPU, random_seed, verobse):
        _tensorboard_model = CNNLSTM_3_layer_tensorboard(
            HP_dict=HP_list2dict,
            log_dir=log_dir,
            GPU=GPU,
            random_seed=random_seed,
            verbose=verobse
        )
        ###
        _tensorboard_model._model_define.epochs = self._model_epoch_control

        _model = _tensorboard_model.model

        _tensorboard_model.training()
        _tensorboard_model.evaluate()

        model_result = _tensorboard_model.model_evaluation

        return _model, model_result

    def tunning_model(self, HP_list, GPU=1, random_seed=1234, log_dir='tensorboard_log/', lock=None, proc_num=None, return_dict=None):
        return super().tunning_model(HP_list, GPU=GPU, random_seed=random_seed, log_dir=log_dir, lock=lock, proc_num=proc_num, return_dict=return_dict)


    def default_HP_result(self, HP_dict):
        return super().default_HP_result(HP_dict)

    def tunning_start(self, n_random_start=50, n_cell=100, x0=None, case_num=3, log_dir='tensorboard_log/', GPU_list=[0,1,2]):
        return super().tunning_start(n_random_start=n_random_start, n_cell=n_cell, x0=x0, case_num=case_num, log_dir=log_dir, GPU_list=GPU_list)

    def change_HP_scale(self, HP_list, changed_scale=False, show_items=False):
        if changed_scale is False:
            HP_list2dict = {
                    'CNN_1_kernal'  : int(HP_list[0]) ,
                    'CNN_1_RNN'     : int(HP_list[1]) ,
                    'CNN_2_kernal'  : int(HP_list[2]) ,
                    'CNN_2_RNN'     : int(HP_list[3]) ,
                    'CNN_3_kernal'  : int(HP_list[4]) ,
                    'CNN_3_RNN'     : int(HP_list[5]) ,

                    'DNN_1'         : int(HP_list[6]),
                    'DNN_2'         : int(HP_list[7]),
                    'DNN_rate'      : int(HP_list[8]),

                    'CNN_dropout'   : float(HP_list[9]),
                    'DNN_dropout'   : float(HP_list[10]),

                    'learning_rate' : float(HP_list[11])
                }
        else:
            HP_list2dict = {
                'CNN_1_kernal'  : int(2**HP_list[0]) ,
                'CNN_1_RNN'     : int(2**HP_list[1]) ,
                'CNN_2_kernal'  : int(2**HP_list[2]) ,
                'CNN_2_RNN'     : int(2**HP_list[3]) ,
                'CNN_3_kernal'  : int(2**HP_list[4]) ,
                'CNN_3_RNN'     : int(2**HP_list[5]) ,

                'DNN_1'         : int(2**HP_list[6]),
                'DNN_2'         : int(2**HP_list[7]),
                'DNN_rate'      : int(2**HP_list[8]),

                'CNN_dropout'   : float(HP_list[9]),
                'DNN_dropout'   : float(HP_list[10]),

                'learning_rate' : float(HP_list[11])
            }


        if show_items is True:
            print(HP_list2dict.items())

        return HP_list2dict



class CNNLSTM_regression_3_layer_HP_tunning(CNNLSTM_3_layer_HP_tunning):
    def __init__(self, default_HP_dict, log_dir='tensorboard_log/', multi_case_tunning=True):
        super().__init__(default_HP_dict, log_dir=log_dir, multi_case_tunning=multi_case_tunning)

    def HP_dimension_define(self):
        return super().HP_dimension_define()

    def model_tensorboard_define(self, HP_list2dict, log_dir, GPU, random_seed, verobse):
        _tensorboard_model = CNNLSTM_regression_3_layer_tensorboard(
            HP_dict=HP_list2dict,
            log_dir=log_dir,
            GPU=GPU,
            random_seed=random_seed,
            verbose=verobse
        )
        ###
        _tensorboard_model._model_define.epochs = self._model_epoch_control

        _model = _tensorboard_model.model

        _tensorboard_model.training()
        _tensorboard_model.evaluate()

        model_result = _tensorboard_model.model_evaluation

        return _model, model_result