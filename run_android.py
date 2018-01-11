# coding: utf-8
import os
import sys
import subprocess
import time
import random
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable

SCALE = 1.02

class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer5 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer6 = nn.Linear(1600,1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0),-1)
        out = self.layer6(out)

        return out # 64

def pull_screenshot(name="autojump.png"):
    process = subprocess.Popen('adb shell screencap -p', shell=True, stdout=subprocess.PIPE)
    screenshot = process.stdout.read()
    f = open(name, 'wb')
    f.write(screenshot)
    f.close()

def preprocess(image):
    w, h = image.size
    top =  (h - w)/2

    image = image.crop((0,top,w,w+top))
    image = image.convert('RGB')
    image = image.resize((224,224), resample=Image.LANCZOS)

    normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
    transform = transforms.Compose([transforms.ToTensor(),normalize])

    image = transform(image)

    return image

def set_button_position(im):
    global swipe_x1, swipe_y1, swipe_x2, swipe_y2
    w, h = im.size
    print(w,h)
    left = int(w / 2)
    top = int(1584 * (h / 1920.0))
    left = int(random.uniform(left-50, left+50))
    top = int(random.uniform(top-10, top+10))
    swipe_x1, swipe_y1, swipe_x2, swipe_y2 = left, top, left, top

def jump(press_time):
    press_time = int(press_time*SCALE)
    cmd = 'adb shell input swipe {x1} {y1} {x2} {y2} {duration}'.format(
        x1=swipe_x1,
        y1=swipe_y1,
        x2=swipe_x2,
        y2=swipe_y2,
        duration=press_time
    )
    os.system(cmd)


def main():

    # init conv net

    net = CNNEncoder()
    if os.path.exists("./model.pkl"):
        net.load_state_dict(torch.load("./model.pkl",map_location=lambda storage, loc: storage))
        print("load model")
    net.eval()

    print("load ok")

    while True:
        pull_screenshot("autojump.png") # obtain screen and save it to autojump.png
        image = Image.open('./autojump.png')
        set_button_position(image)
        image = preprocess(image)

        image = Variable(image.unsqueeze(0))
        press_time = net(image).data[0].numpy()
        print(press_time)
        jump(press_time)

        time.sleep(random.uniform(1.5, 2))






if __name__ == '__main__':
    main()
