#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#import io


def plot_scatter(prediction_data, true_data, plot_name='Test'):
    figure = plt.figure(figsize=(8,8))
    plt.title(plot_name, fontsize=20)
    plt.ylabel('Prediction', fontsize=15)
    plt.xlabel('True', fontsize=15)
    plt.scatter(x=true_data, y=prediction_data)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.grid(True)

    return figure

