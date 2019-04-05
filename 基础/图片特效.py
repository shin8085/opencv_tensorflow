import cv2
import numpy as np


#-----------图像的灰度处理-----------------
img0=cv2.imread("images_imageChanges/img1.jpg",1) #彩色
height=img0.shape[0]
width=img0.shape[1]
mode=img0.shape[2]
#print(img0.shape)
#cv2.imshow("img0",img0)

#直接读取为灰度图片
dstimg1=cv2.imread("images_imageChanges/img1.jpg",0) #灰度图片
#print(dstimg1.shape)
#cv2.imshow("dstimg1",dstimg1)

#读取后转为灰度图片
dstimg2=cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
#cv2.imshow("dstimg2",dstimg2)

#-----------代码实现灰度处理------------
# gray = (R+G+B)/3
dstimg3=np.zeros(img0.shape,np.uint8)
for i in range(0,height):
    for j in range(0,width):
        (b,g,r)=img0[i,j] #读取像素点
        gray=(int(b)+int(g)+int(r))/3
        dstimg3[i,j]=gray
#cv2.imshow("dstimg3",dstimg3)

#gray=R*0.299+g*0.587+b*0.114
dstimg4=np.zeros((height,width),np.uint8)
for i in range(0,height):
    for j in range(0,width):
        (b,g,r)=img0[i,j]
        dstimg4[i,j]=np.uint(r*0.299+g*0.587+b*0.114)
#cv2.imshow("dstimg4",dstimg4)

#---------------代码优化------------
#定点>浮点 >><< >+->*/
dstimg5=np.zeros((height,width),np.uint8)
for i in range(0,height):
    for j in range(0,width):
        (b,g,r)=img0[i,j]
        b=int(b)
        g=int(g)
        r=int(r)
        #gray=np.uint(r*1+g*2+b*1)/4
        gray = (r + (g<<1) + b)>>2
        dstimg5[i,j]=np.uint8(gray)
#cv2.imshow("dstimg5",dstimg5)

#-------------马赛克----------------
dstimg6=np.copy(img0);
for i in range(500,600):
    for j in range(400,500):
        if i%10==0 and j%10==0:
            for x in range(i,i+10):
                for y in range(j,j+10):
                    dstimg6[x,y]=img0[i,j]
#cv2.imshow("dstimg6",dstimg6)

#-----------------边缘检测-----------------
#转为灰度 高斯滤波 canny
#gray=cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
#imgG=cv2.GaussianBlur(gray,(3,3),0)
dstimg7=cv2.Canny(img0,50,50)
#cv2.imshow("dstimg7",dstimg7)
cv2.waitKey(0)