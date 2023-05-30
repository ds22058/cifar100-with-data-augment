# cifar100 with data augment

Using data augment in images classification

## 环境配置

python3.9

pytorch1.12.1

RTX3050ti

## 训练

具体参数见config.py

```python
python train.py --method {baseline/cutout/cutmix/mixup} --epoch300
```

## 查看data augment结果

```python
show_picture.py
```

## 训练结果

结果保存在logs文件夹中

```python
tensorboard --logdir=logs/{filename}
```

或者查看对应的txt文件