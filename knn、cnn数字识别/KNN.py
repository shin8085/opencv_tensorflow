#本质 knn 从k个相似图片中选取出出现频率最高的数字
import tensorflow as tf
import numpy as np
import random
from tensorflow.examples.tutorials.mnist import input_data


#加载数据 one_hot一个数组中一个元素为1则其他都为0
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

#属性设置
trainNum=55000
testNum=10000
trainSize=500
testSize=5
k=4

#数据分解 从0-trainNum中选trainSize个  不可重复
trainIndex=np.random.choice(trainNum,trainSize,replace=False)
testIndex=np.random.choice(testNum,testSize,replace=False)
trainData=mnist.train.images[trainIndex] #获取训练图片  500x784  500个图片 每个图片784个像素点(28*28)
trainLabel=mnist.train.labels[trainIndex] #获取训练标签  500x10
testData=mnist.test.images[testIndex] #获取测试图片  5x784
testLabel=mnist.test.labels[testIndex] #获取测试标签 5x10  10用来表示10个数

#tf input
trainDataInput=tf.placeholder(shape=[None,784],dtype=tf.float32)
trainLabelInput=tf.placeholder(shape=[None,10],dtype=tf.float32)
testDataInput=tf.placeholder(shape=[None,784],dtype=tf.float32)
testLabelInput=tf.placeholder(shape=[None,10],dtype=tf.float32)

#knn distance
f1=tf.expand_dims(testDataInput,1) #维度扩展 5x784 -> 5x1x784
f2=trainDataInput-f1
f3=tf.reduce_sum(tf.abs(f2),reduction_indices=2) #数据累加(第2个维度)
f4=tf.negative(f3) #取反
f5,f6=tf.nn.top_k(f4,k=4) #选取f4中最大的四个元素即f3中最小的四个元素
f7=tf.gather(trainLabelInput,f6) #f6为索引下标
f8=tf.reduce_sum(f7,reduction_indices=1)
f9=tf.argmax(f8,dimension=1) #选取第一的维度上最大的值，并记录下标

with tf.Session() as sess:
    p1=sess.run(f1,feed_dict={testDataInput:testData})
    #print(p1.shape) #(5,1,784)
    p2=sess.run(f2,feed_dict={trainDataInput:trainData,testDataInput:testData})
    #print(p2.shape) #(5,500,784)  (1,100)->testData[1]-trainData[100]
    p3=sess.run(f2,feed_dict={trainDataInput:trainData,testDataInput:testData})
    #print(p3.shape) #(5,500)
    p4 = sess.run(f2, feed_dict={trainDataInput: trainData, testDataInput: testData})
    p5,p6=sess.run((f5,f6),feed_dict={trainDataInput:trainData,testDataInput:testData})
    #print("p5=",p5.shape) #(5,4) 每一张测试图片（5张） 分别对应4张训练图片的距离
    #print("p6=",p6.shape) #(5,4) 下标
    #print("p5[0,0]",p5[0,0])
    #print("p6[0,0]",p6[0,0])
    p7=sess.run(f7,feed_dict={trainDataInput:trainData,testDataInput:testData,trainLabelInput:trainLabel})
    #print(p7.shape) #(5,4,10)
    p8=sess.run(f8,feed_dict={trainDataInput:trainData,testDataInput:testData,trainLabelInput:trainLabel})
    #print(p8.shape) #(5,10)
    p9 = sess.run(f9, feed_dict={trainDataInput: trainData, testDataInput: testData, trainLabelInput: trainLabel})
    print(p9)
    p10=np.argmax(testLabel,axis=1)
    print(p10)
j=0
for i in range(5):
    if p10[i]==p9[i]:
        j+=1
print("概率:",j/5.0)