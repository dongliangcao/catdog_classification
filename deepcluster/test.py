import argparse
import os
from pickle import TRUE
import time
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from util import AverageMeter, load_model
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

parser = argparse.ArgumentParser(description="""Train linear classifier on top
                                 of frozen convolutional layers of an AlexNet.""")

parser.add_argument('--data', type=str, help='path to dataset', default='../../nature_imgs')
parser.add_argument('--model_path', type=str, help='path to model', default='runs/checkpoint.pth.tar')
parser.add_argument('--conv', type=int, choices=[1, 2, 3, 4, 5], default=1,
                    help='on top of which convolutional layer train logistic regression')
parser.add_argument('--tencrops', action='store_true',
                    help='validation accuracy averaged over 10 crops')
parser.add_argument('--exp', type=str, default='runs/eval', help='exp folder')
parser.add_argument('--workers', default=2, type=int,
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', type=int, default=10, help='number of total epochs to run (default: 10)')
parser.add_argument('--batch_size', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--lr', default=0.0005, type=float, help='learning rate (default: 0.0005)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--weight_decay', '--wd', default=-4, type=float,
                    help='weight decay pow (default: -4)')
parser.add_argument('--seed', type=int, default=31, help='random seed')
parser.add_argument('--verbose', action='store_true', help='chatty')


def main():
    global args
    args = parser.parse_args()
    args.verbose = True
    #fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    # load model
    model = load_model(args.model_path)
    model.cuda()
    model.eval()
    

    # data loading code
    valdir = os.path.join(args.data, 'test')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transformations_val = [transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize]

    val_dataset = datasets.ImageFolder(
        valdir,
        transform=transforms.Compose(transformations_val)
    )
    print('### Prepare data ###')
    print(f'# of validation data: {len(val_dataset)}')
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=int(args.batch_size/2),
                                             shuffle=True,
                                             drop_last=True,
                                             num_workers=args.workers)

    # create logs
    exp_log = os.path.join(args.exp, 'log')
    if not os.path.isdir(exp_log):
        os.makedirs(exp_log)

    # evaluate on validation set
    nmi, ari = validate(val_loader, model)

    # write csv
    assert os.path.isfile('../result.csv')
    df = pd.read_csv('../result.csv', header=0, names=['method', 'nmi', 'ari'], dtype={'method': str, 'nmi': float, 'ari': float})
    df = df.append(pd.DataFrame({
        'method': ['deepcluster'],
        'nmi': [nmi],
        'ari': [ari]
    }, index=[len(df.index)]))
    print(df)
    df.to_csv('../result.csv')

def forward(x, model):
    if hasattr(model, 'sobel') and model.sobel is not None:
        x = model.sobel(x)
    x = model.features(x)
    x = x.view(x.size(0), -1)
    x = model.classifier(x)
    x = model.top_layer(x)
    return x

def validate(val_loader, model):
    batch_time = AverageMeter()
    nmis = AverageMeter()
    aris = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    for (input_tensor, target) in tqdm(val_loader):
        target = target.cuda()
        with torch.no_grad():
            input_var =input_tensor.cuda()

            output = forward(input_var, model)
            prob = F.softmax(output, dim=1)
            pred = prob.argmax(dim=1)

            target_np, pred_np =  target.data.cpu().numpy(), pred.data.cpu().numpy()
            nmi = normalized_mutual_info_score(target_np, pred_np)
            nmis.update(nmi)
            ari = adjusted_rand_score(target_np, pred_np)
            aris.update(ari)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return nmis.avg, aris.avg

if __name__ == '__main__':
    main()
