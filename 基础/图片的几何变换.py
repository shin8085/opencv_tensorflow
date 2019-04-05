import cv2
import numpy as np
#--------------------------图片的缩放--------------------------
img1=cv2.imread('images_imageChanges/img1.jpg',1)
#print(type(img1))
#cv2.imshow('img1',img1)
img1Info=img1.shape
print(img1Info)
height=img1Info[0]
width=img1Info[1]
mode=img1Info[2]

# 放大  缩小  等比例  非等比例
dstHeight=int(height*0.5) #目标高度
dstWidth=int(width*0.5)
#最近临域插值 双线性插值 像素关系重采样 立方插值
dstimg1=cv2.resize(img1,(dstWidth,dstHeight))
#cv2.imshow('dstimg1',dstimg1)
#cv2.waitKey(0)

#---------------------------代码实现图片缩放--------------------------------
dstimg2=np.zeros((dstHeight,dstWidth,3),np.uint8) #0~255
#计算坐标
for i in range(0,dstHeight):
    for j in range(0,dstWidth):
        iNew=int(i*(height*1.0/dstHeight))
        jNew=int(j*(width*1.0/dstWidth))
        dstimg2[i,j]=img1[iNew,jNew]
#cv2.imshow('dstimg2',dstimg2)
#cv2.waitKey(0)

#----------------------------图片的剪切-----------------------------
dstimg3=img1[100:200,100:300]
#cv2.imshow('dstimg3',dstimg3)
#cv2.waitKey(0)

#--------------------------图片的位移--------------------------------
#[[1,0],[0,1]] 2*2 A
#[[100],[200]] 2*1 B
#[[x],[y]] C
#A*C+B=[[x*1+y*0],[x*0+y*1]]+[[100],[200]]=[[x+100],[y+200]]
matMove=np.float32([[1,0,100],
                   [0,1,200]])
dstimg4=cv2.warpAffine(img1,matMove,(width,height))
#cv2.imshow('dstimg4',dstimg4)
#cv2.waitKey(0)

#---------代码实现图片位移-----------
dstimg5=np.zeros(img1.shape,dtype=np.uint8)
for i in range(0,height):
    for j in range(0,width-100):  #右移100个像素
        dstimg5[i,j+100]=img1[i,j]
#cv2.imshow("dstimg5",dstimg5)
#cv2.waitKey(0)

#-------------图片镜像---------------
dstimg6=np.zeros((height,width*2,mode),dtype=np.uint8) #水平镜像
for i in range(dstimg6.shape[0]):
    for j in range(dstimg6.shape[1]):
        if j>=width:
            dstimg6[i,j]=img1[i,width-(j-width)-1]
        else:
            dstimg6[i,j]=img1[i,j]
#画一条线
for i in range(0,height):
    dstimg6[i,width]=[0,0,255] #bgr
#cv2.imshow("dstimg6",dstimg6)
#cv2.waitKey(0)

#-----------------通过转换矩阵图片缩放-----------------
matZoom=np.float32([[0.5,0,0],   #见图片位移
                  [0,0.5,0]])
dstimg7=cv2.warpAffine(img1,matZoom,(int(width/2),int(height/2)))
#cv2.imshow("dstimg7",dstimg7)
#cv2.waitKey(0)

#---------------矩阵的仿射变换-------------
#src -> dst 3 (左上角，左下角，右上角）
matSrc=np.float32([[0,0],[0,height-1],[width-1,0]])
matDst=np.float32([[50,50],[300,height-200],[width-300,100]])
#组合
matAffine=cv2.getAffineTransform(matSrc,matDst)
dstimg8=cv2.warpAffine(img1,matAffine,(width,height))
#cv2.imshow("dstimg8",dstimg8)
#cv2.waitKey(0)

#--------------------------图片旋转-------------------------
matRotate=cv2.getRotationMatrix2D((width*0.5,height*0.5),45,1) #中心点 旋转角度 缩放系数
dstimg9=cv2.warpAffine(img1,matRotate,(width,height))
cv2.imshow("dstimg9",dstimg9)
cv2.waitKey(0)