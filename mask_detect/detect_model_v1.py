import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.autograd import Variable
from torchvision import transforms as T
import torch
from PIL import Image
from utils import Config
import os.path as osp
import os
from tqdm import tqdm
import json
from data import get_dataloader
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

def detect(model, device, dataloader):
    model.to(device)
    model.eval()
    centorid_dict = {}

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    for X, Y in tqdm(dataloader):
        X = Variable(X.type(Tensor), requires_grad=False)
        output = model(X)
        for i in range(len(Y)):
            boxes = output[i]['boxes'].cpu().detach().numpy()
            scores = output[i]['scores'].cpu().detach().numpy()
            if len(scores) == 0:
                centorid_dict[Y[i]] = {'state': 0,'centroid':[0,0],'box':[0,0,0,0]}
            for idx in range(len(scores)):
                if scores[idx] > 0.9:
                    x = round((boxes[idx][0] + boxes[idx][2]) / 2)
                    y = round((boxes[idx][1] + boxes[idx][3]) / 2)
                    x = regular(x)
                    y = regular(y)
                    centorid_dict[Y[i]] = {'state': 1, 'centroid':[x,y],'box':boxes[idx].tolist()}
            if Y[i] not in centorid_dict.keys():
                centorid_dict[Y[i]] = {'state': 0, 'centroid': [0, 0],'box':[0,0,0,0]}
    with open('centroid.json', 'w') as f:
        json.dump(centorid_dict, f)


def regular(x):
    if x >= 192:
        return 192
    if x < 64:
        return 64
    return x





if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = build_model('resnet50+mobilev2.pth')

    dataloader = get_dataloader(debug=Config['debug'],
                                               batch_size=Config['batch_size'],
                                               num_workers=Config['num_workers'])

    detect(model, device, dataloader)

