# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = plt.imread(r"d:\myjpg\44.jpg")
plt.subplot(231),plt.imshow(img),plt.title('picture')  
#根据公式转成灰度图
img = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2] 
#显示灰度图
plt.subplot(232),plt.imshow(img,'gray'),plt.title('original')  
#进行傅立叶变换，并显示结果
fft2 = np.fft.fft2(img)
plt.subplot(233),plt.imshow(np.abs(fft2),'gray'),plt.title('fft2')  
#将图像变换的原点移动到频域矩形的中心，并显示效果
shift2center = np.fft.fftshift(fft2)
plt.subplot(234),plt.imshow(np.abs(shift2center),'gray'),plt.title('shift2center')  
#对傅立叶变换的结果进行对数变换，并显示效果
log_fft2 = np.log(1 + np.abs(fft2))
plt.subplot(235),plt.imshow(log_fft2,'gray'),plt.title('log_fft2')  
#对中心化后的结果进行对数变换，并显示结果
log_shift2center = np.log(1 + np.abs(shift2center))
plt.subplot(236),plt.imshow(log_shift2center,'gray'),plt.title('log_shift2center')
plt.show()

# import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt

# img = cv2.imread(r"d:\myjpg\44.jpg", cv2.IMREAD_COLOR)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# fft2=np.fft.fft2(gray)
# fig_amp = np.abs(fft2)
# fig_pha = np.angle(fft2)
# print("amp:",fig_amp)
# print("pha:",fig_pha)

# abs_fft2 = abs(np.fft.fftshift(fft2))

# log_abs_fft2=np.log(1+abs_fft2)
# plt.subplot(1,2,1),plt.imshow(log_abs_fft2,cmap='gray'),plt.title('numpy.fft2')

# fft2_rev=np.fft.ifft2(fft2)
# plt.subplot(1,2,2),plt.imshow(np.abs(fft2_rev),cmap='gray'),plt.title('numpy.fft2.reverse')

# plt.show()

# #傅里叶变换
# import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt

# #读取图像
# img = cv.imread(r"d:\myjpg\44.jpg", 0)

# #快速傅里叶变换算法得到频率分布
# f = np.fft.fft2(img)

# #默认结果中心点位置是在左上角,
# #调用fftshift()函数转移到中间位置
# fshift = np.fft.fftshift(f)       

# #fft结果是复数, 其绝对值结果是振幅
# fimg = np.log(np.abs(fshift))

# #展示结果
# plt.subplot(121), plt.imshow(img, 'gray'), plt.title('Original Fourier')
# plt.axis('off')
# plt.subplot(122), plt.imshow(fimg, 'gray'), plt.title('Fourier Fourier')
# plt.axis('off')
# plt.show()



# -*- coding: utf-8 -*-
# 傅里叶变换和逆变换
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import math

#读取图像
img = cv.imread(r"d:\myjpg\44.jpg", 0)

img=np.array(Image.open(r"d:\myjpg\44.jpg"))
img=img[:,:,0]

print(img.shape)

#傅里叶变换
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
res = np.log(np.abs(fshift))
print(res.shape)

#低通滤波
height=img.shape[0]
width=img.shape[1]
a0=round(height/2)
b0=round(width/2)
d=min(a0,b0)/12

low_filter=np.zeros((height,width),np.int)
for i in range(height):
	for j in range(width):
		dis=math.sqrt((i-a0)**2+(j-b0)**2)
		if(dis<=d):
			low_filter[i,j]=1

low_shift_org=np.multiply(fshift,low_filter)
low_shift = np.log(np.abs(low_shift_org))

#傅里叶逆变换
low_ishift = np.fft.ifftshift(low_shift_org)
low_iimg = np.fft.ifft2(low_ishift)
low_iimg = np.abs(low_iimg)
print(low_iimg.shape)

#高通滤波
height=img.shape[0]
width=img.shape[1]
high_filter=np.ones((height,width),np.int)
for i in range(height):
	for j in range(width):
		dis=math.sqrt((i-a0)**2+(j-b0)**2)
		if(dis<=d):
			high_filter[i,j]=0

high_shift_org=np.multiply(fshift,high_filter)
high_shift = np.log(np.abs(high_shift_org))

#傅里叶逆变换
high_ishift = np.fft.ifftshift(high_shift_org)
high_iimg = np.fft.ifft2(high_ishift)
high_iimg = np.abs(high_iimg)
print(high_iimg.shape)

#傅里叶逆变换
ishift = np.fft.ifftshift(fshift)
iimg = np.fft.ifft2(ishift)
iimg = np.abs(iimg)
print(iimg.shape)

#展示结果
plt.subplot(331), plt.imshow(img, 'gray'), plt.title('Original Image')
plt.axis('off')
plt.subplot(332), plt.imshow(res, 'gray'), plt.title('Fourier Image')
plt.axis('off')
plt.subplot(3,3,3), plt.imshow(iimg, 'gray'), plt.title('Inverse Fourier Image')
plt.axis('off')
plt.subplot(334), plt.imshow(low_filter, 'gray'), plt.title('low_filter Fourier Image')
plt.axis('off')
plt.subplot(335), plt.imshow(low_shift, 'gray'), plt.title('low_shift Fourier Image')
plt.axis('off')
plt.subplot(336), plt.imshow(low_iimg, 'gray'), plt.title('low_iimg Inverse Fourier Image')
plt.axis('off')
plt.subplot(337), plt.imshow(high_filter, 'gray'), plt.title('high_filter Fourier Image')
plt.axis('off')
plt.subplot(338), plt.imshow(high_shift, 'gray'), plt.title('high_shift Fourier Image')
plt.axis('off')
plt.subplot(339), plt.imshow(high_iimg, 'gray'), plt.title('high_iimg Inverse Fourier Image')
plt.axis('off')
plt.show()


plt.figure()
plt.imshow(img,'gray'), plt.title('Original Image')
plt.figure()
plt.imshow(low_iimg+1.5*high_iimg,'gray'), plt.title('high_iimg  Image')
plt.show()