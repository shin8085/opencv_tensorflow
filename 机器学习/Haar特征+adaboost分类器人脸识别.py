import cv2

# 特征->像素经过运算后得到的结果（具体的值 向量 矩阵 多维数组）
# 如何利用特征区分目标？ 阈值判决
# 如何得到特征

# 1 load xml 2 load jpg 3 haar(opencv已完成) gray 4detect 5 draw

#load xml
xmlFace=cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')
#load jpg
imgFace=cv2.imread("images/somePeople.jpg")
#haar(opencv已完成) gray
imgGray=cv2.cvtColor(imgFace,cv2.COLOR_BGR2GRAY)
#detect 检测人脸 1灰度图片 2缩放系数 3人脸最小的像素值
faces=xmlFace.detectMultiScale(imgGray,1.3,5)
print('人脸个数=',len(faces))
#drow
for (x,y,w,h) in faces:
    cv2.rectangle(imgFace,(x,y),(x+w,y+h),(0,0,255),2)
cv2.imshow('faces',imgFace)
cv2.imwrite("images/findFaces.jpg",imgFace)
cv2.waitKey(0)