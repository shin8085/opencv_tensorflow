import tensorflow as tf

data1=tf.constant(25) #常量
data2=tf.Variable(10,name='var') #变量
print(data1)
print(data2)
print('--------------')
#加减乘除
dataAdd=tf.add(data1,data2)
dataMul=tf.multiply(data1,data2)
#dataCopy=tf.assign(data2,dataAdd) #data2=dataAdd
dataSub=tf.subtract(data1,data2)
dataDiv=tf.divide(data1,data2)
with tf.Session() as sess:
    # 初始化变量
    init=tf.global_variables_initializer()
    sess.run(init)

    print('data1',sess.run(data1),'data2',sess.run(data2))
    print('data1+data2=',sess.run(dataAdd))
    print('data1*data2',dataMul.eval()) #==sess.run(dataMul)
    print(sess.run([dataSub,dataDiv])) #输出多个内容
    print('--------------')


data3=tf.placeholder(tf.float32)
data4=tf.placeholder(tf.float32)
dataAdd34=tf.add(data3,data4)
with tf.Session() as sess:
    print("data3*data4:",sess.run(dataAdd34,feed_dict={data3:6,data4:2}))
    print('--------------')


#矩阵
mat1=tf.constant([[1,2,3]])
mat2=tf.constant([[1,2],
                  [2,3],
                  [3,4]])
mat3=tf.constant([4,5,6])
mat4=tf.zeros([2,3]) #全0矩阵
mat4=tf.ones([3,2]) #全1矩阵
mat5=tf.fill([3,3],15) #填充
mat6=tf.zeros_like(mat5) #与mat5维度相同
mat7=tf.linspace(1.0,2.0,10) #1.0到2.0 10等分的矩阵
mat8=tf.random_uniform([2,3],-1,2) #元素为-1到2的随机矩阵
matMul=tf.matmul(mat1,mat2) #矩阵相乘
matAdd=tf.add(mat1,mat3) #矩阵相加
print('shape',mat2.shape)
with tf.Session() as sess:
    print('row0',sess.run(mat2[0])) #行
    print('line0',sess.run(mat2[:,0])) #列
    print('mat1*mat2',sess.run(matMul))
    print('mat1+mat3',sess.run(matAdd))