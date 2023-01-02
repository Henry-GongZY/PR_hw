import torch
import torch.nn as nn
import numpy as np
from torchvision import models
import os
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Vgg19_out(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19_out, self).__init__()
        vgg = models.vgg19(pretrained=False).to(device)  # .cuda()
        vgg.load_state_dict(torch.load(r'./vgg19-dcbb9e9d.pth'))
        vgg.eval()
        vgg_pretrained_features = vgg.features
        # print('vgg_pretrained:',vgg_pretrained_features)

        self.requires_grad = requires_grad
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        print('0-3:', self.slice1)
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        print('4-8:', self.slice2)
        for x in range(9, 14):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        print('9-13:', self.slice3)
        for x in range(14, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        print('14-22:', self.slice4)
        for x in range(23, 32):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        print('23-31:', self.slice5)
        if not self.requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class Perceptual_loss(nn.Module):
    def __init__(self):
        super(Perceptual_loss, self).__init__()
        self.vgg = Vgg19_out().to(device)
        self.criterion = nn.MSELoss()

    def forward(self, x, dir):  # 显示x_vgg[0]的64个特征图
        x_vgg = self.vgg(x)
        '''x - vgg[0]->(1, 64, 256, 256)'''
        cluster = 4
        for i in range(x_vgg[cluster].shape[1]):
            fea = x_vgg[cluster][:, i, :, :]
            fea = fea.view(fea.shape[1], fea.shape[2])
            fea = fea.data.cpu().numpy()
            # use sigmoid to [0,1]
            fea = 1.0 / (1 + np.exp(-1 * fea))
            # to [0,255]
            fea = np.clip(fea, 0, 1)
            fea = np.round(255 * fea)
            cv2.imwrite(dir + str(i) + '.jpg', fea)


if __name__ == "__main__":
    fea_save_path = "./feature_save5_4/"  # 特征图保存地址
    if not os.path.exists(fea_save_path):
        os.mkdir(fea_save_path)
    img = np.array(cv2.imread(r"./pics/1.JPG")) / 255.0
    img = img.transpose((2, 0, 1))
    # print(img.shape)
    img_torch = torch.unsqueeze(torch.from_numpy(img), 0)
    img_torch = torch.as_tensor(img_torch, dtype=torch.float32).to(device)
    perceptual_loss = Perceptual_loss()
    fea_img = perceptual_loss(img_torch, fea_save_path)




