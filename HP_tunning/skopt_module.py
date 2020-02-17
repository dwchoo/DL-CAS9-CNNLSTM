import numpy as np
import skopt
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence, plot_objective, plot_evaluations
from skopt.utils import use_named_args

#from mnist import mnist

hp_dict = {
    'cnn1' : 50,
    'cnn2' : 50,
    'dropout' : 0.3,
    'learning_rate' : 0.001
    }

default_HP = list(hp_dict.values())

def mnist_model_tunning(hp_list):
    from mnist import mnist
    HP_list2dict = {
    'cnn1' : int(hp_list[0]),
    'cnn2' : int(hp_list[1]),
    'dropout' : float(hp_list[2]),
    'learning_rate' : float(hp_list[3])
    }
    print("======================start trainning=======================")
    print(HP_list2dict.items())
    mnist_model = mnist(hp_dict=HP_list2dict)
    mnist_model.model()
    mnist_model.training()
    mnist_model.evaluate()
    loss = mnist_model.loss
    return loss

dim_CNN_1_nodes = Integer(low=10, high=500, name='CNN_1_node')
dim_CNN_2_nodes = Integer(low=10, high=500, name='CNN_2_node')
dim_dropout_nodes = Real(low=0.001, high=0.5, name='Dropout_node')
dim_learning_rate_nodes = Real(low=1e-6, high=1e-1, prior='log-uniform',name='Learning_rate_node')

dimension_HP = [
                dim_CNN_1_nodes  ,
                dim_CNN_2_nodes  ,
                dim_dropout_nodes ,
                dim_learning_rate_nodes
                ]



n_cell = 30
n_random_starts = 10

gp_fitting = gp_minimize(func=mnist_model_tunning,
                        dimensions=dimension_HP,
                        n_calls=n_cell,
                        n_random_starts=n_random_starts,
                        acq_func='EI',
                        #x0=default_HP
                        )

plot1 = plot_evaluations(gp_fitting)
plot2 = plot_objective(gp_fitting)