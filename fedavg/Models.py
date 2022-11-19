import os
import tensorflow as tf
import numpy as np
from dataSets import DataSet
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # BatchNorm에 bias가 포함되어 있으므로, conv2d는 bias=False로 설정합니다.
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
        )

        # identity mapping, input과 output의 feature map size, filter 수가 동일한 경우 사용.
        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        # projection mapping using 1x1conv
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=10, init_weights=True):
        super().__init__()

        self.in_channels=64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # weights inittialization
        if init_weights:
            self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self,x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        x = self.conv3_x(output)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    # define weight initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def resnet18():
    return ResNet(BasicBlock, [2,2,2,2])

def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    return ResNet(BottleNeck, [3,4,6,3])

def resnet101():
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    return ResNet(BottleNeck, [3, 8, 36, 3])

class Models(object):
    def __init__(self, modelName, inputs):
        self.inputs = inputs
        self.model_name = modelName
        if self.model_name == 'mnist_2nn':
            self.mnist_2nn_construct(inputs)
        elif self.model_name == 'mnist_cnn':
            self.mnist_cnn_construct(inputs)
        elif self.model_name == 'cifar10_cnn':
            self.cifar10_cnn_construct(inputs)


    def mnist_2nn_construct(self, inputs):
        self.fc1 = self.full_connect(inputs, 784, 200, 'h1')
        self.fc2 = self.full_connect(self.fc1, 200, 200, 'h2')
        self.outputs = self.full_connect(self.fc2, 200, 10, 'last_layer', relu=False)

    def mnist_cnn_construct(self, inputs):
        self.trans_inputs  = tf.reshape(inputs, [-1, 28, 28, 1])
        self.cov1 = self.convolve(self.trans_inputs, 1, 5, 1, 1, 32, 'cov1', True, 'SAME')
        self.pool1 = self.max_pool_nxn(self.cov1, 2, 2, 'pool1')
        self.cov2 = self.convolve(self.pool1, 32, 5, 1, 1, 64, 'cov2', True, 'SAME')
        self.pool2 = self.max_pool_nxn(self.cov2, 2, 2, 'pool2')
        with tf.variable_scope('transform') as scope:
            self.trans_pool2 = tf.reshape(self.pool2, [-1, 7 * 7 * 64])
        self.fc1 = self.full_connect(self.trans_pool2, 7 * 7 * 64, 512, 'fc1')
        self.outputs = self.full_connect(self.fc1, 512, 10, 'last_layer', relu=False)

    def cifar10_cnn_construct(self, inputs):
        self.cov1 = self.convolve(inputs, 3, 5, 1, 1, 64, 'cov1', True, 'SAME')
        self.pool1 = self.max_pool_nxn(self.cov1, 3, 2, 'pool1')
        self.cov2 = self.convolve(self.pool1, 64, 5, 1, 1, 64, 'cov2', True, 'SAME')
        self.pool2 = self.max_pool_nxn(self.cov2, 3, 2, 'pool2')
        with tf.variable_scope('transform') as scope:
            self.trans_pool2 = tf.reshape(self.pool2, [-1, 6 * 6 * 64])
        self.fc1 = self.full_connect(self.trans_pool2, 6 * 6 * 64, 384, 'fc1')
        self.fc2 = self.full_connect(self.fc1, 384, 192, 'fc2')
        self.outputs = self.full_connect(self.fc2, 192, 10, 'last_layer', relu=False)


    def full_connect(self, inputs, num_in, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('weights', shape=[num_in, num_out], dtype=tf.float32, trainable=True)
            biases = tf.get_variable('biases', shape=[num_out], dtype=tf.float32, trainable=True)
            ws_plus_bs = tf.nn.xw_plus_b(inputs, weights, biases)

            if relu == True:
                outputs = tf.nn.relu(ws_plus_bs)
                return outputs
            else:
                return ws_plus_bs


    def convolve(self, inputs, inputs_channels, kernel_size, stride_y, stride_x, num_features, name, relu=True, padding='SAME'):
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('weights', shape=[kernel_size, kernel_size, inputs_channels, num_features],
                                      dtype=tf.float32, trainable=True)
            biases = tf.get_variable('baises', shape=[num_features], dtype=tf.float32, trainable=True)
            conv = tf.nn.conv2d(inputs, weights, [1, stride_y, stride_x, 1], padding=padding)
            cov_puls_bs = tf.nn.bias_add(conv, biases)

            if relu == True:
                outputs = tf.nn.relu(cov_puls_bs)
                return outputs
            else:
                return cov_puls_bs


    def max_pool_nxn(self, inputs, ksize, ssize, name):
        with tf.variable_scope(name) as scope:
            return tf.nn.max_pool(inputs, ksize=[1, ksize, ksize, 1], strides=[1, ssize, ssize, 1], padding='SAME')





if __name__=='__main__':

    # GPU preparation
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'

    mnist = DataSet('mnist', is_IID=1)

    with tf.variable_scope('inputs') as scope:
        input_images = tf.placeholder(shape=[None, 784], dtype=tf.float32, name='input_images')
        true_label = tf.placeholder(shape=[None, 10], dtype=tf.float32, name='true_label')

    mnist_2nn = Models('mnist_2nn', input_images)
    predict_label = tf.nn.softmax(mnist_2nn.outputs)

    with tf.variable_scope('loss') as scope:
        Cross_entropy = -tf.reduce_mean(true_label*tf.log(predict_label), axis=1)

    with tf.variable_scope('train') as scope:
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train = optimizer.minimize(Cross_entropy)

    with tf.variable_scope('validation') as scope:
        correct_prediction = tf.equal(tf.argmax(predict_label, axis=1), tf.argmax(true_label, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    # ---------------------------------------- train --------------------------------------------- #
    with tf.Session(config=tf.ConfigProto(
            log_device_placement=False, \
            allow_soft_placement=True, \
            gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        sess.run(tf.initialize_all_variables())

        for i in range(1000):
            batch_images, batch_labels = mnist.next_batch(100)
            sess.run(train, feed_dict={input_images: batch_images, true_label: batch_labels})
            if i%20 == 0:
                batch_images = mnist.test_data
                batch_labels = mnist.test_label
                print(sess.run(accuracy, feed_dict={input_images: batch_images, true_label: batch_labels}))
