import tensorflow as tf
import tensorboard
from tensorboard.plugins.hparams import api as hp

from datetime import datetime
import matplotlib.pyplot as plt
import io

from HP_tunning.plot import *


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


class HP_tunning_tensorboard_callback:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.histogram_freq = 10

        # tensorboard plot image log dir
        self.file_writer_plot = tf.summary.create_file_writer(self.log_dir + '/validation')


    def TB_callbacks(self, plot_callbacks=[]):
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=self.histogram_freq)#, profile_batch=0)
        
        if type(plot_callbacks) is not list:
            plot_callbacks = [plot_callbacks]
        
        callback_list = [tensorboard_callback] + plot_callbacks
        return callback_list

    def callback_log_plot(self, model, input_data, true_data, plot, plot_name='test scatter'):
        _callbacks_log_plot = lambda _epoch, _logs : self.log_regression_plot(
                                                                            epoch= _epoch,
                                                                            logs= _logs,
                                                                            model= model,
                                                                            input_data = input_data,
                                                                            true_data = true_data,
                                                                            plot = plot,
                                                                            plot_name = plot_name,
                                                                        )
        return tf.keras.callbacks.LambdaCallback(on_epoch_end=_callbacks_log_plot)

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

    def log_regression_plot(self, epoch, logs, model, input_data, true_data, plot, plot_name='Test scatter'):
        if epoch%self.histogram_freq == 0:
            prediction = model.predict(input_data)
            #if prediction.shape[1] >= 2:
            #    prediction = np.argmax(prediction, axis=1)
            #else:
            #    prediction = prediction.reshape(-1,)
            prediction = prediction.reshape(-1,)

            figure = plot(prediction_data=prediction, true_data=true_data, plot_name=plot_name)
            fig_image = self.plot_to_image(figure)
            
            with self.file_writer_plot.as_default():
                tf.summary.image(plot_name, fig_image, step=epoch)
            #return fig_image
