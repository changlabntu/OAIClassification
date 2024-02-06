import time, os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from utils.make_config import *

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from engine.lightning_siamese import LitClassification
from pytorch_lightning.callbacks import ModelCheckpoint

import torchvision.models as models


def model_from_checkpoint(checkpoint_name):
    torch_checkpoint = torch.load(checkpoint_name)
    args = torch_checkpoint['hyper_parameters']

    # datasets
    from loaders.OAI_pain_loader import OAIUnilateralPain
    train_set = OAIUnilateralPain(mode='train')
    val_set = OAIUnilateralPain(mode='eval')
    train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=16, drop_last=False)
    eval_loader = DataLoader(val_set, batch_size=args['batch_size'], shuffle=False, num_workers=16, drop_last=False)

    # Imports
    from models.PainClassify import PainNetSingle
    net = PainNetSingle(args_m=args)
    net.par_freeze = []

    from utils.metrics_classification import ClassificationLoss, GetAUC
    loss_function = ClassificationLoss()
    metrics = GetAUC()

    ln_classification = LitClassification.load_from_checkpoint(checkpoint_name,
                                                               args=args,
                                                               train_loader=train_loader,
                                                               eval_loader=eval_loader,
                                                               net=net,
                                                               loss_function=loss_function,
                                                               metrics=metrics)
    return ln_classification, train_set, val_set


def get_cam_images(cam_dict):
    images = []
    for k in cam_dict.keys():
        print(k)
        gradcam, gradcam_pp = cam_dict[k]
        mask, _ = gradcam(normed_torch_img)
        heatmap, result = visualize_cam(mask, torch_img)

        mask_pp, _ = gradcam_pp(normed_torch_img)
        heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)

        images.append(torch.stack([torch_img.squeeze().cpu(), heatmap, heatmap_pp, result, result_pp], 0))

    imagesgrid = make_grid(torch.cat(images, 0), nrow=5)
    return imagesgrid, images


from utils.GradCAM.utils import visualize_cam, Normalize
from utils.GradCAM.gradcam import GradCAM, GradCAMpp
import os
import PIL
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import make_grid, save_image
from utils.data_utils import imagesc

ln_classification, train_set, val_set = model_from_checkpoint('checkpoints/alex_maxpool.ckpt')

# images
if 0:
    pil_img = PIL.Image.open('utils/GradCAM/bird.png')
    normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    torch_img = torch.from_numpy(np.asarray(pil_img)).permute(2, 0, 1).unsqueeze(0).float().div(255).cuda()
    torch_img = F.upsample(torch_img, size=(224, 224), mode='bilinear', align_corners=False)
    normed_torch_img = normalizer(torch_img)

x = val_set.__getitem__(100)
torch_img = torch.from_numpy(x[1][0][:, :, :, 14]).unsqueeze(0).cuda()
normed_torch_img = torch_img


pain = ln_classification.net
pain.eval(), pain.cuda()
alexnet = models.alexnet(pretrained=True)
alexnet.eval(), alexnet.cuda()

cam_dict = dict()
alexnet_model_dict = dict(type='alexnet', arch=pain, layer_name='features_11', input_size=(224, 224))
#alexnet_model_dict = dict(type='vgg', arch=pain, layer_name='features_19', input_size=(224, 224))
alexnet_gradcam = GradCAM(alexnet_model_dict, True)
alexnet_gradcampp = GradCAMpp(alexnet_model_dict, True)
cam_dict['pain'] = [alexnet_gradcam, alexnet_gradcampp]

alexnet_model_dict = dict(type='alexnet', arch=alexnet, layer_name='features_11', input_size=(224, 224))
alexnet_gradcam = GradCAM(alexnet_model_dict, True)
alexnet_gradcampp = GradCAMpp(alexnet_model_dict, True)
cam_dict['alexnet'] = [alexnet_gradcam, alexnet_gradcampp]

imagesgrid, imagesori = get_cam_images(cam_dict)
imagesc(imagesgrid)


# CAM
def get_cam_original(torch_img, channel, num_layer):
    fmap = pain.features[:num_layer](torch_img)
    lnw = pain.classifier.weight[:, :, 0, 0]

    cam = torch.zeros((fmap.shape[2:])).cuda()
    for c in range(fmap.shape[1]):
        cam = cam + fmap[0, c, :, :] * lnw[channel, c]

    cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), (224, 224))
    cam = cam.cpu().detach()[0, 0, ::]
    return cam


def norm_to_0_1(x):
    x = x - x.min()
    return x / x.max()


def get_all_cam(x, num_layer):
    all_img0 = []
    all_img1 = []
    all_cam0 = []
    all_cam1 = []
    for s in range(x[1][0].shape[3]):
        img0 = torch.from_numpy(x[1][0][:, :, :, s]).unsqueeze(0).cuda()
        cam0 = get_cam_original(img0, 1, num_layer)
        img1 = torch.from_numpy(x[1][1][:, :, :, s]).unsqueeze(0).cuda()
        cam1 = get_cam_original(img1, 0, num_layer)
        all_img0.append(img0.cpu().detach()[0, 0, ::])
        all_img1.append(img1.cpu().detach()[0, 0, ::])
        all_cam0.append(cam0)
        all_cam1.append(cam1)

    all_img0 = norm_to_0_1(torch.cat(all_img0, 1))
    all_img1 = norm_to_0_1(torch.cat(all_img1, 1))
    all_cam0 = norm_to_0_1(torch.cat(all_cam0, 1))
    all_cam1 = norm_to_0_1(torch.cat(all_cam1, 1))

    all_cam = torch.cat([all_img0, all_cam0, all_img1, all_cam1], 0)
    return all_cam


for i in range(len(val_set)):
    x = val_set.__getitem__(i)
    all_cam = get_all_cam(x, num_layer=12)
    imagesc(all_cam, show=False, save='cams/alex_maxpool/' + str(i) + '.png')
