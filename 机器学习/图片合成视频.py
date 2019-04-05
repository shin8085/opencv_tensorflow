import cv2

img=cv2.imread("images/image0.jpg")
imgInfo=img.shape
size=(imgInfo[0],imgInfo[1])
# 名称 解码器 帧率 size
videoWrite=cv2.VideoWriter('images/video2.mp4',-1,5,size) #写入对象
for i in range(0,10):
    fileName="images/image"+str(i)+".jpg"
    img=cv2.imread(fileName)
    videoWrite.write(img) #写入
