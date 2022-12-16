#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import os


# In[ ]:


class MyFashionMnist(object):
  def train(self):

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.summary()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10)

    model.evaluate(x_test,  y_test, verbose=2)


if __name__ == '__main__':
    if os.getenv('FAIRING_RUNTIME', None) is None:
        from kubeflow.fairing.builders.append.append import AppendBuilder
        from kubeflow.fairing.preprocessors.converted_notebook import             ConvertNotebookPreprocessor

        DOCKER_REGISTRY = '192.168.0.180:5000' #'kubeflow-registry.default.svc.cluster.local:30000'
        builder = AppendBuilder(
            registry=DOCKER_REGISTRY,
            image_name='fairing-job',
            base_image='brightfly/kubeflow-jupyter-lab:tf2.0-cpu', #'brightfly/kubeflow-jupyter-lab:tf2.0-gpu'
            push=True,
            preprocessor=ConvertNotebookPreprocessor(
                notebook_file="fashion-mnist-fairing-onlyb.ipynb"
            )
        )
        builder.build()
    else:
        myModel = MyFashionMnist()
        myModel.train()

