import msgpack
import numpy as np
import tensorflow as tf
import zmq

tf.config.experimental_run_functions_eagerly(True)


class Puller(tf.keras.layers.Layer):
    def __init__(self, context, port):
        self.sock = context.socket(zmq.PULL)
        self.sock.bind(f'tcp://*:{port}')
        super(Puller, self).__init__()
        self.cnt = 0

    def __del__(self):
        self.sock.close()

    def build(self, input_shape):
        super(Puller, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if self.cnt == 0:
            self.cnt += 1
            return tf.convert_to_tensor(np.zeros((100,400)), dtype=tf.float32)
        msg = self.sock.recv()
        unpacked_data = msgpack.unpackb(msg, use_list=True, raw=False)
        output = np.asarray(unpacked_data)
        print(output.shape)
        converted = tf.convert_to_tensor(output, dtype=tf.float32)
        self.cnt += 1
        return converted


if __name__ == '__main__':
    ctx = zmq.Context()
    port = 5558

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train, x_test = x_train / 255.0, x_test / 255.0
    # x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    # x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    model = tf.keras.models.Sequential([
        Puller(ctx, port),
        # tf.keras.layers.Input((400,)),
        tf.keras.layers.Dense(120, activation='tanh'),
        tf.keras.layers.Dense(84, activation='tanh'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.predict(np.zeros(10000), verbose=2, batch_size=100))
    print(model.layers[0].cnt)

    ctx.destroy()