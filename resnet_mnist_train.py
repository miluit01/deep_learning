import tensorflow as tf
import datetime

from tensorflow.keras import callbacks, datasets, optimizers, losses
from tensorflow.python.ops.gen_math_ops import arg_max
from resnet import resnet18
from custom_callbacks import LearningRateScheduler

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def preprocess(x, y):
    # [0, 255] => [0.0, 1.0] => [-0.5, 0.5]
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 0.5 # cast is ok; but convert_to_tensor is not working-> uint8 can't be converted to float32 tensor
    # due to the self customized model the x input image need to be reshaped from 2x1 [28,28] => 1x1 [28*28]
    y = tf.cast(y, dtype=tf.int64)
    # the output need to one-hot as well
    y = tf.one_hot(y, depth=10)

    return x, y

LR_SCHEDULE = [
    # (epoch to start, learning rate) tuples
    (5, 0.05),
    (10, 0.01),
    (15, 0.005),
    (20, 0.001),
]

def lr_schedule(epoch, lr):
    """Helper function to retrieve the scheduled learning rate based on epoch."""
    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
        return lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1]
    return lr

def main():
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/' + current_time

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='./mnist.pnz')

    x_train = tf.reshape(x_train, (-1, 28, 28, 1))
    x_test = tf.reshape(x_test, (-1, 28, 28, 1))

    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(preprocess).shuffle(10000).batch(200)
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(preprocess).batch(100)

    network = resnet18(class_numbers=10)
    network.compile(optimizer=optimizers.Adam(lr=0.001),
                    loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    network.build((None, 28, 28, 1))
    network.summary()
    tf.summary.trace_on(graph=True)

    network.fit(train_db,
                # batch_size=32 !!! Don't needed bcz using the keras.dataset instance
                epochs=30,
                validation_data=test_db,
                validation_freq=1,
                callbacks = [tensorboard_callback]
               )

    # bullshit evaluate should use custom data sets
    network.evaluate(test_db)

    # [b, 28, 28]
    clip = next(iter(test_db))
    # [b, 28, 28] => [b, 10]
    result = network.predict(clip[0])
    print(result.shape)
    # [b, 10] => [b, 10, 10]
    result = tf.one_hot(result, depth=result.shape[1])
    # print the index of max value in the one_hot result, the index is the predicted number of images
    # print(tf.math.argmax(result, axis=2))
    network.save_weights('checkpoints/resnet18_mnist.h5')

if __name__ == '__main__':
    main()
