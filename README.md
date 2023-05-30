# cifar100-with-data-augment
Using data augment in image classification
## 训练
train.py --method {baseline/cutout/cutmix/mixup} --epoch 300 
## 查看data aug结果
show_picture.py
## 训练结果
结果在logs文件夹中，使用tensorboard查看，或者找到对应的txt文件
