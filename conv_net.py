import msgpack
import tensorflow as tf
import numpy as np
import zmq
import pprint


tf.config.experimental_run_functions_eagerly(True)


class Pusher(tf.keras.layers.Layer):
    def __init__(self, context, port):
        self.sock = context.socket(zmq.PUSH)
        self.sock.connect(f'tcp://localhost:{port}')
        super(Pusher, self).__init__()
        self.a = []
        self.cnt = 0

    def build(self, input_shape):
        super(Pusher, self).build(input_shape)

    def call(self, inputs, **kwargs):
        try:
            self.a = inputs
            if self.cnt == 0:
                print(inputs.numpy())
            serialized = msgpack.packb(inputs.numpy().tolist(), use_bin_type=True)
            self.sock.send(serialized)
        except AttributeError:
            print("no numpy")
        print(inputs.shape)
        self.cnt += 1
        return inputs


if __name__ == '__main__':
    ctx = zmq.Context()
    port = 5558

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(6, kernel_size=(5,5), padding='same', activation='tanh', input_shape=x_train[0].shape, strides=(1,1)),
        tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid'),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation='tanh'),
        tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
        tf.keras.layers.Flatten(),
        Pusher(ctx, port)
        # tf.keras.layers.Dense(120, activation='tanh'),
        # tf.keras.layers.Dense(84, activation='tanh'),
        # tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # model.fit(x_train, y_train, epochs=100)
    # model.evaluate(x_test, y_test)
    model.predict(x_test, verbose=2, batch_size=100)

    print('predict end')
    ctx.destroy()
    # print(len(model.layers))
    layer = model.layers[5]
    print(layer.cnt)