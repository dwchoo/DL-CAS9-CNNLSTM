import sys

from hyperparameters import *

from model_tunning import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'




def main(type = 1, log_dir = 'tensorboard_log/', HP_file_name='hyperparameters.py'):
    if type == '1':
        print("CNNLSTM_model\n")
        CNNLSMT_model(log_dir=log_dir, file_name=HP_file_name)
    elif type == '2':
        print("CNNLSMT_test_model\n")
        CNNLSMT_test_model(log_dir=log_dir, file_name=HP_file_name)
    elif type == '3':
        print("CNNLSTM_3_layer_model\n")
        CNNLSTM_3_layer_model(log_dir=log_dir, file_name=HP_file_name)
    elif type == '4':
        print("CNNLSTM_regression_3_layer_model\n")
        CNNLSTM_regression_3_layer_model(log_dir=log_dir, file_name=HP_file_name)
        


    else:
        print("type {} is not exist".format(type))



def CNNLSMT_model(log_dir='tensorboard_log/', file_name='hyperparameters.py'):
    if file_name[-3:] == '.py':
         HP_file_name = file_name[:-3]
    else:
        HP_file_name = file_name
    HP_dict = __import__(HP_file_name).HP_tunning_scale
    
    model = CNNLSTM_short_HP_tunning(
        default_HP_dict=HP_dict,
        log_dir=log_dir
    )

    model.tunning_start_multiprocs(
        n_random_start=30,
        n_cell=60,
        x0=None,
        case_num=3,
        log_dir=log_dir,
        GPU_list=[1,2,0]
    )
    model.tunning_result_plot()
    model.show_best_point()

    
    #print(HP_dict)
    #print(log_dir)



def CNNLSMT_test_model(log_dir='tensorboard_log/', file_name='hyperparameters.py'):
    if file_name[-3:] == '.py':
         HP_file_name = file_name[:-3]
    else:
        HP_file_name = file_name
    HP_dict = __import__(HP_file_name).HP_tunning_short_scale
    
    model = SpCas9_HP_tunning(
        default_HP_dict=HP_dict,
        log_dir=log_dir,
        multi_case_tunning=True
    )

    model.tunning_start(
        n_random_start=30,
        n_cell=80,
        x0=None,
        case_num=3,
        log_dir=log_dir,
        GPU_list=[1,2,0]
    )
    model.tunning_result_plot()
    model.show_best_point()


def CNNLSTM_3_layer_model(log_dir='tensorboard_log/', file_name='hyperparameters.py'):
    if file_name[-3:] == '.py':
         HP_file_name = file_name[:-3]
    else:
        HP_file_name = file_name
    HP_dict = __import__(HP_file_name).HP_tunning_3_layer_scale
    
    model = CNNLSTM_3_layer_HP_tunning(
        default_HP_dict=HP_dict,
        log_dir=log_dir,
        multi_case_tunning=True
    )

    model._model_epoch_control = 300
    model._model_verbose = 0

    model.tunning_start(
        n_random_start=30,
        n_cell=80,
        x0=None,
        case_num=3,
        log_dir=log_dir,
        GPU_list=[1,2,0]
    )
    model.tunning_result_plot()
    model.show_best_point()




def CNNLSTM_regression_3_layer_model(log_dir='tensorboard_log/', file_name='hyperparameters.py'):
    if file_name[-3:] == '.py':
         HP_file_name = file_name[:-3]
    else:
        HP_file_name = file_name
    HP_dict = __import__(HP_file_name).HP_tunning_3_layer_scale
    
    model = CNNLSTM_regression_3_layer_HP_tunning(
        default_HP_dict=HP_dict,
        log_dir=log_dir,
        multi_case_tunning=True
    )

    model._model_epoch_control = 300
    model._model_verbose = 0

    model.tunning_start(
        n_random_start=30,
        n_cell=80,
        x0=None,
        case_num=3,
        log_dir=log_dir,
        GPU_list=[1,2,0]
    )
    model.tunning_result_plot()
    model.show_best_point()





if __name__ == "__main__":

    input_index_length = len(sys.argv)
    if input_index_length == 4:
        main(type=sys.argv[1], log_dir=sys.argv[2], HP_file_name=sys.argv[3])
    elif input_index_length == 3:
        main(type=sys.argv[1], log_dir=sys.argv[2])
    elif input_index_length == 2:
        main(type=sys.argv[1])
    else:
        print("Choose type")