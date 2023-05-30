import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import os
import resnet
from tqdm import tqdm
import config
from cutout import Cutout
from torch.autograd import Variable
import utils
import cv2 as cv
from torch.utils.tensorboard import SummaryWriter


def main(args, transform_method, file_path):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(os.path.join('./logs',args.method+'_'+args.model)):
        os.makedirs(os.path.join('./logs',args.method+'_'+args.model))
    writer = SummaryWriter(os.path.join('./logs',args.method+'_'+args.model))
    train_data = torchvision.datasets.CIFAR100(root='./cifar100', train=True, transform=transform_method['train'], download=True)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True, pin_memory=True)
    test_data = torchvision.datasets.CIFAR100(root='./cifar100', train=False, download=True, transform=transform_method['test'])
    testloader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True, pin_memory=True)
    #show picture after augment
    # a, b, c = train_data.data[:3,:,:,:]

    # net = torchvision.models.resnet34(weights=False)
    # net = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
    # net.load_state_dict(torch.load('resnet34cifar100.pkl'))
    net = resnet.ResNet50(num_classes=100)
    # 修改通道数
    # in_channel = net.fc.in_features
    # net.fc = nn.Sequential(
    #     nn.Linear(in_channel, 10),
    #     nn.LogSoftmax(dim=1)
    # )
    #
    # child_counter = 0
    # for child in net.children():
    #     print('child',child_counter)
    #     print(child)
    #     if child_counter<7:
    #         for param in child.parameters():
    #             param.requires_grad = False
    #     else:
    #         for param in child.parameters():
    #             param.requires_grad = True
    #     child_counter += 1


    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120, 180, 240], gamma=0.1)
    print('start training')
    with open(file_path, 'w') as f:

        for epoch in range(args.epoch):
            #train
            train_loss = 0
            correct = 0
            total = 0
            train_acc = []
            net.train()
            # progress_bar = tqdm(trainloader)
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to('cuda'), targets.to('cuda')
                optimizer.zero_grad()
                if args.method=='cutmix':
                    r = np.random.rand(1)
                    if args.beta>0 and r<args.cutmix_prob:
                        lam = np.random.beta(args.beta, args.beta)
                        rand_index = torch.randperm(inputs.size()[0]).cuda()
                        target_a = targets
                        target_b = targets[rand_index]
                        bbx1, bby1, bbx2, bby2 = utils.rand_bbox(inputs.size(), lam)
                        inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                        # adjust lambda to exactly match pixel ratio
                        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
                        # cv2.imwrite('img{}'.format(batch_idx)+'.png', (inputs[0,:,:,:]-np.min(inputs[0,:,:,:])/(np.max(inputs[0,:,:,:])-np.min(inputs[0,:,:,:]))))
                        # compute output
                        outputs = net(inputs)
                        loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
                    else:
                        outputs = net(inputs)
                        loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                elif args.method=='mixup':
                    inputs, targets_a, targets_b, lam = utils.mixup_data(inputs, targets,
                                                                   args.alpha, use_cuda)
                    inputs, targets_a, targets_b = map(Variable, (inputs,
                                                                  targets_a, targets_b))
                    outputs = net(inputs)
                    loss = utils.mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()#loss.data[0]
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                                + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
                else:
                    outputs = net(torch.squeeze(inputs, 1))
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                if (batch_idx+1)%50==0:
                    train_acc.append(100. * correct / total)
                if (batch_idx+1)%250==0:
                    print(batch_idx + 1, '/', len(trainloader), 'epoch: %d' % epoch, '| Loss: %.3f | Acc: %.3f%% (%d/%d)'
                          % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

            writer.add_scalar('trian_loss', train_loss / (batch_idx + 1), epoch)
            writer.add_scalar('train_acc', round(100. * correct / total, 6), epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], epoch)

            #test
            net.eval()
            test_loss = 0
            test_correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
                    inputs, targets = inputs.to('cuda'), targets.to('cuda')
                    outputs = net(torch.squeeze(inputs, 1))
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    test_correct += predicted.eq(targets).sum().item()
                test_acc =  100. * test_correct / total
                writer.add_scalar('test_acc', test_acc, epoch)
            print('epoch: %d' % (epoch+1), 'test_cc: %.3f%% '% (test_acc), 'Learning_rate= %.3f:' %scheduler.get_last_lr()[0])
            f.write("EPOCH=%03d,Learning_rate= %.5f ,train_accuracy= %.3f ,test_accuracy= %.3f%% (%d/%d)"
                    % (epoch + 1, scheduler.get_last_lr()[0],np.mean(train_acc) , test_acc, test_correct, total))
            f.write('\n')
            f.flush()
            scheduler.step()
        torch.save(net.state_dict(), '{}cifar100_{}.pkl'.format(args.model, args.method))



if __name__ == '__main__' :

    parser = config.config_parser()
    args = parser.parse_args()
    transform_method = {}
    # normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
    #                                  std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    transform_method['train'] = transforms.Compose([
    transforms.ToTensor(),
    # transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(brightness=(0,1), contrast=(0,1), saturation=(0,1), hue=0),
    # transforms.RandomVerticalFlip(p=0.5),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    # normalize
])
    transform_method['test'] = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    # normalize
])
    if args.method=='cutout':
        # transform_method['train'].transforms.append(cutout(n_holes=args.n_holes, length=args.length))
        transform_method['train'] = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            Cutout(n_holes=args.n_holes, length=args.length)])
        main(args, transform_method=transform_method, file_path='./{}_cifar100_{}.txt'.format(args.method, args.model))
    # elif args.method=='cutmix':
    #     main(args, transform_method=transform_method, file_path='./cutmix_cifar100_resnet50.txt')
    # elif args.method=='mixup':
    #     main(args, transform_method=transform_method, file_path='./mixup_cifar100_resney50.txt')
    else:
        main(args, transform_method=transform_method, file_path='./{}_cifar100_{}.txt'.format(args.method, args.model))





