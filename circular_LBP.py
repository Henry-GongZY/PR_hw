from PIL import Image
import numpy as np

def rgb2gray(img):
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]

    gray = 0.3 * r + 0.59 * g + 0.11 * b
    return gray

# 旋转不变的圆形LBP算法
def circular_LBP(img, R, P): # 参数: 原始图像img，半径R，采样点个数P
    h, w = img.shape
    img_LBP = np.zeros(img.shape, dtype = img.dtype)
    for row in range(R, h-R):
        for col in range(R, w-R):
            LBP_str = []
            for p in range(P): # 遍历全部采样点
                # 计算采样点的坐标(浮点数)
                x_p = row + R * np.cos(2 * np.pi * p / P)
                y_p = col + R * np.sin(2 * np.pi * p / P)
                # print(x_p, y_p)
                
                x_1 = int(np.floor(x_p))
                y_1 = int(np.floor(y_p)) # floor是向下取整
                x_2 = min(x_1 + 1, h - 1)
                y_2 = min(y_1 + 1, w - 1) # 防止超出原图的尺寸
                
                # 双线性插值求出这个采样点的像素值
                value0 = (x_2 - x_p) * img[x_1, y_1] + (x_p - x_1) * img[x_2, y_1]
                value1 = (x_2 - x_p) * img[x_1, y_2] + (x_p - x_1) * img[x_2, y_2,]
                temp = int((y_2 - y_p) * value0 + (y_p - y_1) * value1)
                
                # 与窗口中心坐标的像素值进行比较
                if temp >= img[row, col]:
                    LBP_str.append(1)
                else:
                    LBP_str.append(0)
            
            # print(LBP_str)
            LBP_str = ''.join('%s' %id for id in LBP_str)

            # 旋转不变性
            Min = int(LBP_str, 2) # 转换为十进制数，作为Min的初始值
            for i in range(len(LBP_str)):
                t = LBP_str[i::] + LBP_str[:i]
                # print(t, int(t, 2))
                Min = min(int(t, 2), Min)
            
            img_LBP[row, col] = Min
    
    return img_LBP

if __name__ == "__main__":
    img = np.array(Image.open("./pics/1.JPG"))
    print(img.shape)
    img = rgb2gray(img)
    img = circular_LBP(img, 3, 8)
    Image.fromarray(np.uint8(img)).save('./pics/3.JPG')