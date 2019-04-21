import cv2

cap=cv2.VideoCapture("neg.mp4")
isOpen=cap.isOpened()
fps=cap.get(cv2.CAP_PROP_FPS)
width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
nums=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
(flag,frame)=cap.read()
if isOpen==True:
    for i in range(0,nums):
        (flag,frame)=cap.read()
        fileName="neg/neg"+str(i)+".jpg"
        if flag==True:
            cv2.imwrite(fileName,frame)
