import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import sklearn.metrics
import itertools
import io

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras import Input, Model

from tensorboard.plugins.hparams import api as hp

#from keras.layers import *


class mnist:
    def __init__(self, hp_dict):
        self.data_set()

        time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = 'tensorboard/' + time_stamp

        #hp
        #self.cnn1 = hp_dict['cnn1']
        #self.cnn2 = hp_dict['cnn2']
        #self.dropout = hp_dict['dropout']

        #hp-param
        self.HP_cnn1 = hp.HParam('cnn1',hp.IntInterval(10, 500))#hp_dict['cnn1']]))
        self.HP_cnn2 = hp.HParam('cnn2',hp.IntInterval(10, 500))#hp_dict['cnn2']]))
        self.HP_dropout = hp.HParam('dropout', hp.RealInterval(0.001, 0.6))
        #self.HP_cnn1 = hp.HParam('cnn1',hp. hp_dict['cnn1'])
        #self.HP_cnn2 = hp.HParam('cnn2',hp_dict['cnn2'])
        #self.HP_dropout = hp.HParam('dropout', hp_dict['dropout'])
        
        with tf.summary.create_file_writer(self.log_dir + '/hp_tunning').as_default():
            hp.hparams_config(
                hparams=[self.HP_cnn1, self.HP_cnn2, self.HP_dropout],
                metrics=[hp.Metric('accuracy', display_name='Accuracy')],
            )
        


        self.cnn1 = hp_dict['cnn1']#self.HP_cnn1.domain.values[0]
        self.cnn2 = hp_dict['cnn2']#self.HP_cnn2.domain.values[0]
        self.dropout = hp_dict['dropout']

        

        #self.model()
        #self.training()
        #self.evaludate()

        
        
        
        
        
        pass



    

    def data_set(self):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.x_train, self.x_test = x_train / 255.0, x_test / 255.0
        self.y_train = y_train
        self.y_test = y_test

        self.train_img = self.x_train.reshape((-1,28,28,1))
        self.test_img = self.x_test.reshape((-1,28,28,1))
        pass

    def model(self):
        CNN_input = Input(shape = (28,28,1))
        CNN = Conv2D(self.cnn1,(3,3),activation='relu')(CNN_input)
        CNN = MaxPool2D((2,2))(CNN)
        CNN = Dropout(self.dropout)(CNN)
        CNN = Conv2D(self.cnn2,(3,3),activation='relu')(CNN)
        CNN = MaxPool2D((2,2))(CNN)
        CNN = Dropout(self.dropout)(CNN)
        CNN = Flatten()(CNN)
        CNN = Dense(10,activation='softmax')(CNN)
        self.CNN_model = Model(CNN_input,CNN)
        #CNN_model.summary()
        pass

    def training(self, callback_list=[]):

        call_list = self.TB_callbacks() + callback_list

        self.CNN_model.compile(optimizer='adam',
                                loss = 'sparse_categorical_crossentropy',
                                metrics=['accuracy'])
        self.CNN_model.fit(self.train_img,self.y_train, epochs=5,
                            callbacks = call_list)
        pass

    def TB_callbacks(self,):
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)#, profile_batch=0)

        self.file_writer_cm = tf.summary.create_file_writer(self.log_dir + '/test')

        self.cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=self.log_confusion_matrix)
        #pass
        return [tensorboard_callback, self.cm_callback]

    
    def evaluate(self):
        loss, acc = self.CNN_model.evaluate(self.test_img, self.y_test, verbose=2)
        hparams = {
            'cnn1' : self.cnn1,
            'cnn2' : self.cnn2,
            'dropout' : self.dropout,
        }
        with tf.summary.create_file_writer(self.log_dir+'/hp_tunning').as_default():
            hp.hparams(hparams)
            accuracy = acc
            tf.summary.scalar('accuracy', accuracy, step=1)
        pass

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

    def plot_confusion_matrix(self, cm, class_names):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.

        Args:
            cm (array, shape = [n, n]): a confusion matrix of integer classes
            class_names (array, shape = [n]): String names of the integer classes
        """
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=0)
        plt.yticks(tick_marks, class_names)

        # Normalize the confusion matrix.
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return figure

    

    def log_confusion_matrix(self, epoch, logs):

        class_names = ['{}'.format(i) for i in range(10)]

        # Use the model to predict the values from the validation dataset.
        test_pred_raw = self.CNN_model.predict(self.test_img)
        test_pred = np.argmax(test_pred_raw, axis=1)

        # Calculate the confusion matrix.
        cm = sklearn.metrics.confusion_matrix(self.y_test, test_pred)
        # Log the confusion matrix as an image summary.
        figure = self.plot_confusion_matrix(cm, class_names=class_names)
        cm_image = self.plot_to_image(figure)

        # Log the confusion matrix as an image summary.
        with self.file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)
    
    