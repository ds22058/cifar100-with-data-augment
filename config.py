import argparse

def config_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.1, type= float, help="learning rate")
    parser.add_argument('--method', default=None, type=str, help='cutmix cutout mixup or baseline')
    parser.add_argument('--epoch', default=200, type=int, help='training epoches')
    parser.add_argument('--n_holes', type=int, default=1, help='number of holes to cut out from image')
    parser.add_argument('--length', type=int, default=16, help='length of the holes')
    parser.add_argument('--beta', type=float, default=0.5, help='cutmix beta')
    parser.add_argument('--cutmix_prob', type=float, default=0.5, help='prob of cutmix')
    parser.add_argument('--alpha', default=1., type=float, help='mixup interpolation coefficient (default: 1)')
    parser.add_argument('--model', default='resnet18', type=str)
    return parser
