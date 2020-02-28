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


class tmp_tensorboard:
    def __init__(self, HP_dict, log_dir = 'tensorboard/'):
        time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir =  log_dir + time_stamp

        # HP
        self.CNN_1_kernal = hp.HParam('CNN_1_kernal',hp.IntInterval(1, 1024))
        self.CNN_1_RNN    = hp.HParam('CNN_1_RNN',hp.IntInterval(1, 1024))
        self.CNN_2_kernal = hp.HParam('CNN_2_kernal',hp.IntInterval(1, 1024))
        self.CNN_2_RNN    = hp.HParam('CNN_2_RNN',hp.IntInterval(1, 1024))
        self.CNN_3_kernal = hp.HParam('CNN_3_kernal',hp.IntInterval(1, 1024))
        self.CNN_3_RNN    = hp.HParam('CNN_3_RNN',hp.IntInterval(1, 1024))
        self.CNN_4_kernal = hp.HParam('CNN_4_kernal',hp.IntInterval(1, 1024))
        self.CNN_4_RNN    = hp.HParam('CNN_4_RNN',hp.IntInterval(1, 1024))
        self.CNN_5_kernal = hp.HParam('CNN_5_kernal',hp.IntInterval(1, 1024))
        self.CNN_5_RNN    = hp.HParam('CNN_5_RNN',hp.IntInterval(1, 1024))
        self.DNN_1        = hp.HParam('DNN_1',hp.IntInterval(1, 1024))
        self.DNN_2        = hp.HParam('DNN_2',hp.IntInterval(1, 1024))
        self.DNN_rate     = hp.HParam('DNN_rate',hp.IntInterval(1, 1024))
        self.CNN_dropout  = hp.HParam('CNN_dropout', hp.RealInterval(0.001, 0.6))
        self.DNN_dropout  = hp.HParam('DNN_dropout', hp.RealInterval(0.001, 0.6))
        self.learning_rate  = hp.HParam('learning_rate', hp.RealInterval(1e-6, 1.0))


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
                metrics=[hp.Metric('loss', display_name='Loss'),
                         hp.Metric('pearson_correlation', display_name='Pearson Correlation'),
                         hp.Metric('spearman_correlation', display_name='Spearman Correlation')]
            )

        
        def H_param(self, loss, pearson, spearman, step=1):
            hparams = {
                'CNN_1_kernal'  : HP_dict['CNN_1_kernal'] ,
                'CNN_1_RNN'     : HP_dict['CNN_1_RNN']    ,
                'CNN_2_kernal'  : HP_dict['CNN_2_kernal'] ,    
                'CNN_2_RNN'     : HP_dict['CNN_2_RNN']    ,
                'CNN_3_kernal'  : HP_dict['CNN_3_kernal'] ,    
                'CNN_3_RNN'     : HP_dict['CNN_3_RNN']    ,
                'CNN_4_kernal'  : HP_dict['CNN_4_kernal'] ,    
                'CNN_4_RNN'     : HP_dict['CNN_4_RNN']    ,
                'CNN_5_kernal'  : HP_dict['CNN_5_kernal'] ,    
                'CNN_5_RNN'     : HP_dict['CNN_5_RNN']    ,
                'DNN_1'         : HP_dict['DNN_1']        ,
                'DNN_2'         : HP_dict['DNN_2']        ,
                'DNN_rate'      : HP_dict['DNN_rate']     ,
                'CNN_dropout'   : HP_dict['CNN_dropout' ] ,    
                'DNN_dropout'   : HP_dict['DNN_dropout' ] ,
                'learning_rate' : HP_dict['learning_rate'],
            }
            with tf.summary.create_file_writer(self.log_dir + '/hp_tunning').as_default():
                hp.hparams(hparams=hparams)
                tf.summary.scalar('Loss', loss, step=step)
                tf.summary.scalar('Pearson_correlation', pearson, step=step)
                tf.summary.scalar('Spearman_correlation', spearman, step=step)

        



class HP_tunning_tensorboard_callback:
    def __init__(self, log_dir, model,
                data = {},#{'input' : None, 'True' : None},
                plot = plot_scatter,
                plot_name = 'scatter'):
        
        self.log_dir = log_dir
        self.input_data = data['input']
        self.true_data = data['True']

        self.callbacks_log_plot = lambda _epoch, _logs : self.log_regression_plot(
                                                                            epoch= _epoch,
                                                                            logs= _logs,
                                                                            model= model,
                                                                            input_data = self.input_data,
                                                                            true_data = self.true_data,
                                                                            plot = plot,
                                                                            plot_name = plot_name,
                                                                        )



    def TB_callbacks(self):
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)#, profile_batch=0)

        self.file_writer_plot = tf.summary.create_file_writer(self.log_dir + '/train')

        plot_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=self.callbacks_log_plot)
        #pass
        return [tensorboard_callback, plot_callback]


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
