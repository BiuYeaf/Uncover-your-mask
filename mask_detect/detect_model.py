import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
import torch
from PIL import Image
from utils import Config
import os.path as osp
import os
from tqdm import tqdm
import json
import numpy as np

def build_model(pth):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2  
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 192,256),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)
    model = FasterRCNN(backbone,
                    num_classes=2,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.load_state_dict(torch.load(pth, map_location=device))

    return model

def detect(img):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = build_model('resnet50+mobilev2.pth')
    model.to(device)
    model.eval()
    transforms2=T.Compose([
                    T.Resize((256,256),Image.BICUBIC),
                    T.ToTensor()
                ])
    img =transforms2(img)
    img = [img.to(device)]
    output = model(img)
    boxes = output[0]['boxes'].cpu().detach().numpy()
    scores = output[0]['scores'].cpu().detach().numpy()
    if len(scores) == 0:
        return False, None
    for i in range(len(scores)):
        if scores[i] > 0.9:
            return True, boxes[i]
    return False, None

def regular(x):
    if x >= 192:
        return 192
    if x < 64:
        return 64
    return x

class Find():
    def __init__(self):
        self.root_dir = Config['root_path']
        self.train_dir = osp.join(self.root_dir, 'train')
        self.y_train_dir =osp.join(self.root_dir, 'y_train')
        self.test_dir = osp.join(self.root_dir,'test')
        self.y_test_dir = osp.join(self.root_dir, 'y_test')
        self.centorid_dict = {}

    def creat(self):
        print('Load train data')
        dir_list = os.listdir(self.y_train_dir)
        for dir in tqdm(dir_list):
            for image in os.listdir(osp.join(self.y_train_dir, dir)):
                image = '01_'+ image
                file_name = osp.join(self.train_dir, dir, image)
                mask, boxes = detect(Image.open(file_name).convert("RGB"))
                if mask:
                    x = round((boxes[0] + boxes[2]) / 2)
                    y = round((boxes[1] + boxes[3]) / 2)
                    x = regular(x)
                    y = regular(y)
                    self.centorid_dict[osp.join(dir,image[3:])] = {'state':True, 'centroid':(x,y)}
                else:
                    self.centorid_dict[osp.join(dir,image[3:])] = {'state':False,'centroid':(0,0)}

        print('Load test data')
        dir_list = os.listdir(self.y_test_dir)
        for dir in tqdm(dir_list):
            for image in os.listdir(osp.join(self.y_test_dir, dir)):
                image = '01_'+ image
                file_name = osp.join(self.test_dir, dir, image)
                mask, boxes = detect(Image.open(file_name).convert("RGB"))
                if mask:
                    x = round((boxes[0] + boxes[2]) / 2)
                    y = round((boxes[1] + boxes[3]) / 2)
                    x = regular(x)
                    y = regular(y)
                    self.centorid_dict[osp.join(dir,image[3:])] = {'state': 1, 'centroid':[x,y]}
                else:
                    self.centorid_dict[osp.join(dir,image[3:])] = {'state': 0,'centroid':[0,0]}

    def dump_json(self):
        with open('centroid.json', 'w') as f:
            json.dump(self.centorid_dict, f)



if __name__ == '__main__':
    find = Find()
    find.creat()
    find.dump_json()

    #
    # ###
    # file = r'G:\pycharmproject\599\599final\y_train0\n000002\0002_01.jpg'
    # out = detect(Image.open(file).convert("RGB"))
