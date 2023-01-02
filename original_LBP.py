from PIL import Image
import numpy as np

def rgb2gray(img):
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]

    gray = 0.3 * r + 0.59 * g + 0.11 * b
    return gray

# 原始LBP
def LBP(img): # 传入参数的图片已经转为灰度图了
    h, w = img.shape[:2]
    img_LBP = np.zeros(img.shape, dtype = img.dtype)
    for row in range(1, h-1):
        for col in range(1, w-1):
            center = img[row, col]
            LBPtemp = 0

            # 比中心像素大的点赋值为1，比中心像素小的点赋值为0，然后将这8位二进制数转换成十进制数
            LBPtemp |= (img[row-1,col-1] >= center) << 7
            LBPtemp |= (img[row-1,col  ] >= center) << 6
            LBPtemp |= (img[row-1,col+1] >= center) << 5
            LBPtemp |= (img[row  ,col+1] >= center) << 4
            LBPtemp |= (img[row+1,col+1] >= center) << 3
            LBPtemp |= (img[row+1,col  ] >= center) << 2
            LBPtemp |= (img[row+1,col-1] >= center) << 1
            LBPtemp |= (img[row  ,col-1] >= center) << 0

            img_LBP[row, col] = LBPtemp

    return img_LBP

if __name__ == "__main__":
    img = np.array(Image.open("./pics/1.JPG"))
    print(img.shape)
    img = rgb2gray(img)
    img = LBP(img, 3, 8)
    Image.fromarray(np.uint8(img)).save('./pics/3.JPG')