
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
# 加载数据
minst=input_data.read_data_sets("MNIST_data",one_hot=True)

# input
imageInput=tf.placeholder(tf.float32,[None,784]) #28*28
lableInput=tf.placeholder(tf.float32,[None,10])

#维度调整 N*784->M*28*28*1 1通道
imageInputReshape=tf.reshape(imageInput,[-1,28,28,1])

#卷积 内核5x5 in:1 out:32
w0=tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1))
b0=tf.Variable(tf.constant(0.1,shape=[32]))

# layer1激励函数+卷积运算
layer1=tf.nn.relu(tf.nn.conv2d(imageInputReshape,w0,strides=[1,1,1,1],padding='SAME')+b0)

# pool采样 -> 数据量减小
layer1_pool=tf.nn.max_pool(layer1,ksize=[1,4,4,1],strides=[1,4,4,1],padding='SAME')
