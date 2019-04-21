import cv2
import numpy as np


for i in range(0,2592):
    fileName = "neg/neg" + str(i) + ".jpg"
    img = cv2.imread(fileName)
    imginfo = img.shape
    height = imginfo[0]
    width = imginfo[1]

    trains = cv2.transpose(img)  # 旋转
    dst = cv2.flip(trains, 1)  # 镜像

    dst = cv2.resize(dst, (int(height * 0.1), int(width * 0.1)))[0:128, 4:68]

    cv2.imwrite(fileName, dst,[cv2.IMWRITE_JPEG_QUALITY,100])
    print(i)
'''fileName = "Test.jpg"
img = cv2.imread(fileName)
imginfo = img.shape
height = imginfo[0]
width = imginfo[1]

dst = cv2.resize(img, (int(width * 0.25), int(height * 0.25)))

cv2.imwrite(fileName, dst, [cv2.IMWRITE_JPEG_QUALITY, 100])'''


