import numpy as np
import skopt
from skopt import gp_minimize, Optimizer
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence, plot_objective, plot_evaluations
from skopt.utils import use_named_args


hp_dict = {
    'cnn1' : 50,
    'cnn2' : 50,
    'dropout' : 0.3,
    'learning_rate' : 0.001
    }

default_HP = list(hp_dict.values())

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

tunning = Optimizer(dimensions=dimension_HP,
                    base_estimator='GP',
                    acq_func='EI',
                    
                    )