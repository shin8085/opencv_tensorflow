import cv2

# 图片读写
img=cv2.imread('images/Image0.png',1)  # 读取图片 0灰度图片 1彩色图片

print(img[20][20])  # 输出像素值bgr

cv2.imwrite('images/Image1.jpg',img) # 写入图片
cv2.imwrite('images/Image2.jpg',img,[cv2.IMWRITE_JPEG_QUALITY, 50])  # jpg图片 0-100 有损压缩
cv2.imwrite('images/Image.png',img,[cv2.IMWRITE_PNG_COMPRESSION, 0])  #无 损 0-9 可设置透明度

# 画一条线
for i in range(100):
    img[10+i, 100]=(255, 0, 0)

cv2.imshow("images/Image",img) # 展示图片
cv2.waitKey(0)  # 暂停



