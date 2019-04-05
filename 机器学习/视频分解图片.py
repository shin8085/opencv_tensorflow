import cv2

cap=cv2.VideoCapture("video1.mp4") #打开视频
isOpen=cap.isOpened() #判断视频是否打开
fps=cap.get(cv2.CAP_PROP_FPS)
width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if isOpen==True:
    for i in range(0,10):
        (flag,frame)=cap.read() #读取每一帧 flag是否读取成功 frame每一帧的图片
        fileName="images/image"+str(i)+'.jpg'
        if flag==True:
            cv2.imwrite(fileName,frame,[cv2.IMWRITE_JPEG_QUALITY,100])