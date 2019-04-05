import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

date=np.linspace(1,15,15)
startPrice=np.array([97.214,108.554,117.654,124.654,124.654,112.254,122.654,119.354,111.254,127.554,134.354,146.254,157.154,158.654,159.664])
endPrice=np.array([90.414,115.854,126.054,124.154,113.154,121.254,121.645,110.154,129.654,133.554,144.854,158.554,153.714,157.144,160.114])

#画图
for i in range(0,15):
    tdate=np.zeros([2])
    tdate[0]=i #时间
    tdate[1]=i
    price=np.zeros([2])
    price[0]=startPrice[i] #开盘价格
    price[1]=endPrice[i] #收盘价格
    if startPrice[i] < endPrice[i] :
        plt.plot(tdate,price,'r',lw=8)
    else:
        plt.plot(tdate,price,'g',lw=8)
#plt.show()

#预测股票收盘价格
#A输入层 B隐藏层 C输出层
#A(15x1)*w1(1x10)+b1(1*10)=B(15x10)   w权重，b偏移矩阵
#B(15x10)*w2(10x1)+b2(15x1)=C(15x1)
dateNormal=np.zeros([15,1])
priceNormal=np.zeros([15,1])
#归一化
for i in range(0,15):
    dateNormal[i,0]=i/14.0
    priceNormal[i,0]=endPrice[i]/200.0

x=tf.placeholder(tf.float32,[None,1]) #N行1列
y=tf.placeholder(tf.float32,[None,1])
#B
w1=tf.Variable(tf.random_uniform([1,10],0,1))
b1=tf.Variable(tf.zeros([1,10]))
wb1=tf.matmul(x,w1)+b1
layer1=tf.nn.relu(wb1)#激励函数（映射）得到隐藏层
#C
w2=tf.Variable(tf.random_uniform([10,1],0,1))
b2=tf.Variable(tf.zeros([15,1]))
wb2=tf.matmul(layer1,w2)+b2
layer2=tf.nn.relu(wb2)
loss=tf.reduce_mean(tf.square(y-layer2)) #计算值和真实值之间的差值 y真值 layer2计算值
#步长0.1  梯度下降法  减小loss
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 训练一万次
    for i in range(0,10000):
        sess.run(train_step,feed_dict={x:dateNormal,y:priceNormal})
    #经训练，得到w1,w2,b1,b2
    pred=sess.run(layer2,feed_dict={x:dateNormal})
    predPrice=np.zeros([15,1]) #预期收盘价格
    for i in range(0,15):
        predPrice[i,0]=(pred*200)[i,0] #数据还原
    plt.plot(date,predPrice,'b',lw=1)
plt.show()