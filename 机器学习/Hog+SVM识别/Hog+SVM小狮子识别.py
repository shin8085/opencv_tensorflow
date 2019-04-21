# Hog 特征
# 1模块划分 2梯度 方向 模板 3bin 投影 4每个模块hog
# imagge win block cell
# win 64*124  block 16*16  block_step 8*8  block_count 105  cell_size 8*8
# cell bin 梯度：运算    每个像素->梯度:大小 方向
# block=4cell
# 360/40=9bin
# 1bin=40 cell
# haar 值   hog 向量（维度）
# 维度=105*4*9=3780


# hog+svm识别
# 1准备样本 2训练 3测试
# pos 正样本 包含所检测目标  neg 负样本   64X128
# 正样本 尽可能的多样  环境 干扰
# 1准备参数 2创建hog 3svm 4计算hog 5label 6训练 7预测
import cv2
import numpy as np
import matplotlib.pyplot as plt

#1准备参数
posNum=1464
negNum=2592
winSize=(64,128)
blockSize=(16,16) #105
blockStride=(8,8) #4 cell
cellSize=(8,8)
nBin=9 #9 bin  3780

#2创建hog
hog=cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nBin)

#3svm
svm=cv2.ml.SVM_create();

#计算hog
featureNum=int(((128-16)/8+1)*((64-16)/8+1)*4*9) #3780
featureArray=np.zeros(((posNum+negNum),featureNum),np.float32)
featureLabel=np.zeros(((posNum+negNum),1),np.int32)
#svm 监督学习 样本(hog)  标签
#处理正样本
for i in range(0,1):
    fileName="pos/pos"+str(i)+".jpg"
    img=cv2.imread(fileName)
    hist=hog.compute(img,(8,8)) #hog特征的计算  3780维
    for j in range(0,featureNum):
        featureArray[i,j]=hist[j]
        #featureArray hog1[1,:] hog2[2,:] .....
    featureLabel[i,0]=1 #正样本 label为1
print("正样本处理完成")

#处理负样本
for i in range(0,negNum):
    fileName="neg/neg"+str(i)+".jpg"
    img=cv2.imread(fileName)
    hist=hog.compute(img,(8,8))
    for j in range(featureNum):
        featureArray[i+posNum,j]=hist[j]
    featureLabel[i+posNum,0]=-1 #负样本 label为-1

svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setC(0.01)
print("负样本处理完成")

# 6训练
result=svm.train(featureArray,cv2.ml.ROW_SAMPLE,featureLabel)
print("训练完成")

#7检测
alpha=np.zeros((1),np.float32)
rho=svm.getDecisionFunction(0,alpha)
alphaArray=np.zeros((1,1),np.float32)
supportVArray=np.zeros((1,featureNum),np.float32)
resultArray=np.zeros((1,featureNum),np.float32)
alphaArray[0,0]=alpha
resultArray=-1*alphaArray*supportVArray
print("数据准备完成")

#detect
myDetect=np.zeros((3781),np.float32)
for i in range(0,378):
    myDetect[i]=resultArray[0,i]
myDetect[3780]=rho[0]
print("detect准备完成")

#构建hog
myHog=cv2.HOGDescriptor()
myHog.setSVMDetector(myDetect)
print("hog构建完成")

#加载检测图片
print("加载检测图片")
imageSrc=cv2.imread('Test.jpg')
print("开始检测")
objs=myHog.detectMultiScale(imageSrc,0,(8,8),(32,32),1.05,2) #检测
print("检测完成")
x=int(objs[0][0][0])
y=int(objs[0][0][1])
w=int(objs[0][0][2])
h=int(objs[0][0][3])
#图片绘制
cv2.rectangle(imageSrc,(x,y),(x+w,y+h),(255,0,0),2)
cv2.imshow("imgSrc",imageSrc)
cv2.waitKey(0)