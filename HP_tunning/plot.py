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

'''
def plot_to_image(figure):
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

def log_plot(epoch, model, input_data, true_data, plot, plot_name='Test scatter'):
    prediction = model.predict(input_data)
    if prediction.shape[1] >= 2:
        prediction = np.argmax(prediction, axis=1)
    else:
        prediction = prediction.reshape(-1,)

    figure = plot(prediction_data=prediction, true_data=true_data, plot_name=plot_name)
    fig_image = plot_to_image(figure)
    return fig_image
'''