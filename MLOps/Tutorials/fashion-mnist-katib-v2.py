#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import os
import argparse
from tensorflow.python.keras.callbacks import Callback

from datetime import datetime, timezone
import logging


# In[2]:


class MyFashionMnist(object):
  def train(self):
    
    # 입력 값을 받게 추가합니다.
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', required=False, type=float, default=0.001)
    parser.add_argument('--dropout_rate', required=False, type=float, default=0.2)
    # 0 : SGD, 1 : Adam
    parser.add_argument('--opt', required=False, type=int, default=0)
    # epoch 5 ~ 15
    parser.add_argument('--epoch', required=False, type=int, default=5)    
    # relu, sigmoid, softmax, tanh
    parser.add_argument('--act', required=False, type=str, default='relu')        
    # layer 1 ~ 5
    parser.add_argument('--layer', required=False, type=int, default=1)        
        
    args = parser.parse_args()    
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    for i in range(int(args.layer)):    
        model.add(tf.keras.layers.Dense(128, activation=args.act))
        model.add(tf.keras.layers.Dropout(args.dropout_rate))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.summary()
    
    sgd = tf.keras.optimizers.SGD(learning_rate=args.learning_rate)
    adam = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    
    optimizers = [sgd, adam]
    model.compile(optimizer=optimizers[args.opt],
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    model.fit(x_train, y_train,
              verbose=0,
              validation_data=(x_test, y_test),
              epochs=args.epoch,
              callbacks=[KatibMetricLog()])

    model.evaluate(x_test,  y_test, verbose=0)

class KatibMetricLog(Callback):
    def on_batch_end(self, batch, logs={}):
        print("batch=" + str(batch),
              "accuracy=" + str(logs.get('acc')),
              "loss=" + str(logs.get('loss')))
    def on_epoch_begin(self, epoch, logs={}):
        print("epoch " + str(epoch) + ":")
    
    def on_epoch_end(self, epoch, logs={}):
        # RFC 3339
        local_time = datetime.now(timezone.utc).astimezone().isoformat()
        logging.info(( "\n{} accuracy={:.4f} loss={:.4f} Validation-accuracy={:.4f} Validation-loss={:.4f}"
                       .format( local_time, logs.get('acc'), logs.get('loss'), 
                                logs.get('val_acc'), logs.get('val_loss') )
                      ))
        print("Validation-accuracy=" + str(logs.get('val_acc')),
              "Validation-loss=" + str(logs.get('val_loss')))
        return

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")
    
if __name__ == '__main__':
    
    # For metricsCollector
    file_path = '/var/log/katib'
    createDirectory(file_path)
    logging.basicConfig(filename=file_path+'/'+'metrics.log', level=logging.DEBUG)
    
    if os.getenv('FAIRING_RUNTIME', None) is None:
        from kubeflow import fairing
        from kubeflow.fairing.kubernetes import utils as k8s_utils

        DOCKER_REGISTRY = '192.168.0.180:5000' #'kubeflow-registry.default.svc.cluster.local:30000'
        fairing.config.set_builder(
            'append',
            image_name='fairing-job-v2',
            base_image='brightfly/kubeflow-jupyter-lab:tf2.0-cpu',
            registry=DOCKER_REGISTRY, 
            push=True)
        # cpu 1, memory 2GiB
        fairing.config.set_deployer('job',
                                    namespace='space-openess', #'dudaji',
                                    pod_spec_mutators=[
                                        k8s_utils.get_resource_mutator(cpu=1,
                                                                       memory=2)]
         
                                   )
        fairing.config.run()
    else:
        remote_train = MyFashionMnist()
        remote_train.train()


# In[ ]:




