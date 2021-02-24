import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers, regularizers, Sequential

initializer = tf.random_normal_initializer(stddev=0.001)

class ResBlock(layers.Layer):

    def __init__(self, filters, strides=1):
        super(ResBlock, self).__init__()

        self.conv1 = layers.Conv2D(filters, (3, 3), strides=strides, padding='same', kernel_initializer=initializer)
        self.bn1   = layers.BatchNormalization()
        self.relu  = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filters, (3, 3), strides=1, padding='same', kernel_initializer=initializer)
        self.bn2   = layers.BatchNormalization()

        if strides == 1:
            self.identity = lambda x:x
        else:
            self.identity = Sequential([
                                        layers.Conv2D(filters, (1, 1), strides)
                                        ])

    def call(self, inputs, training=None):
        conv1_out = self.relu(self.bn1(self.conv1(inputs), training=training))
        conv2_out = self.bn2(self.conv2(conv1_out), training=training)

        identity = self.identity(inputs)
        output = tf.nn.relu(layers.add([conv2_out, identity]))

        return output


class ResNet(models.Model):

    def __init__(self, dims, class_numbers):
        super(ResNet, self).__init__()

        self.stem = Sequential([
                                layers.Conv2D(64, (7, 7), strides=2, padding='same', kernel_initializer=initializer), #
                                layers.BatchNormalization(),
                                layers.Activation('relu'),
                                layers.MaxPool2D(pool_size=(2,2), strides=1, padding='same')
                                ])

        self.layer1 = self.build_resblocks(64, dims[0])
        self.layer2 = self.build_resblocks(128, dims[1], strides=2)
        self.layer3 = self.build_resblocks(256, dims[2], strides=2)
        self.layer4 = self.build_resblocks(512, dims[3], strides=2)

        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(class_numbers)

    def call(self, inputs, training=None):
        x = self.stem(inputs)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x

    def build_resblocks(self, filters, numbers, strides=1):
        res_blocks = Sequential([ResBlock(filters, strides)])

        for _ in range(1, numbers):
            res_blocks.add(ResBlock(filters, strides=1))

        return res_blocks


def resnet18(class_numbers):
    return ResNet(dims=[2, 2, 2, 2], class_numbers=class_numbers)


def resnet34(class_numbers):
    return ResNet(dims=[3, 4, 6, 3], class_numbers=class_numbers)
