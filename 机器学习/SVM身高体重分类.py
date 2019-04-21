#SVM 寻求一个最优的超平面进行分类
#SVN 核：line

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1 准备data
woman=np.array([[156,47],[162,50],[166,52],[170,56],[176,60]])
man=np.array([[156,53],[164,55],[172,60],[176,62],[184,69]])

# 2 lable
lable=np.array([[0],[0],[0],[0],[0],[1],[1],[1],[1],[1]])

# 3 data
data=np.vstack((woman,man)) #合并数组
data=np.array(data,dtype=np.float32)

# SVM 所有数据都要有标签
# [156,47] -> 0 woman
# 有标签的训练 -- 监督学习
# 0负样本 1正样本

# 4 训练
svm=cv2.ml.SVM_create() # ml 机器学习模块  SVM_create() 创建一个支持向量机
# 属性设置
svm.setType(cv2.ml.SVM_C_SVC) # svm type
svm.setKernel(cv2.ml.SVM_LINEAR) #线性
svm.setC(0.1)
# 训练
result=svm.train(data,cv2.ml.ROW_SAMPLE,lable)

#预测
pt_data=np.array([[167,55],[162,57]],dtype=np.float32) #预测数据
print(pt_data)
(par1,par2)=svm.predict(pt_data)  #预测结果
print(par2)
