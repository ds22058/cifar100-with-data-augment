import torch
import torchvision
import cv2
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.autograd import Variable
import utils
import copy

transform_method = {}
transform_method['train'] = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

transform_method['test'] = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])
#baseline
train_data = torchvision.datasets.CIFAR100(root='./cifar100', train=True, transform=transform_method['train'],
                                           download=True)
baseline_set = train_data.data[:10,:,:,:]
cutmix_set = copy.deepcopy(train_data.data[:10,:,:,:])
mixup_set = copy.deepcopy(train_data.data[:10,:,:,:])
cutout_set = copy.deepcopy(train_data.data[:10,:,:,:])
for i in range(10):
    cv2.imwrite('./images/baseline{}.jpg'.format(i),baseline_set[i,:,:,:])

rand_index = 2

lam = 0.5
for i in range(10):
    mixup_set[i,:,:,:] = lam * mixup_set[i,:,:,:] + (1 - lam) * mixup_set[rand_index, :, :, :]
    cv2.imwrite('./images/mixup{}.jpg'.format(i),mixup_set[i,:,:,:])

lam = np.random.beta(1,1)
bbx1, bby1, bbx2, bby2 = utils.rand_bbox([10,3,32,32], lam)
cutmix_set[:, bbx1:bbx2, bby1:bby2,:] = cutmix_set[rand_index, bbx1:bbx2, bby1:bby2,:]
cutout_set[:, bbx1:bbx2, bby1:bby2,:] = 0
for i in range(10):
    cv2.imwrite('./images/cutmix{}.jpg'.format(i),cutmix_set[i,:,:,:])
    cv2.imwrite('./images/cutout{}.jpg'.format(i),cutout_set[i,:,:,:])






