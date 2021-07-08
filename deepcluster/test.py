import argparse
import os
from pickle import TRUE
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from util import AverageMeter, learning_rate_decay, load_model, Logger
from sklearn.metrics import roc_auc_score, normalized_mutual_info_score

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

    best_prec1 = 0

    # load model
    model = load_model(args.model_path)
    model.cuda()
    
    # freeze the features layers
    for param in model.features.parameters():
        param.requires_grad = False
    # freeze the classifier lyaers
    for param in model.classifier.parameters():
        param.requires_grad = False
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'test')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.tencrops:
        transformations_val = [
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda(lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops])),
        ]
    else:
        transformations_val = [transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               normalize]

    transformations_train = [transforms.Resize(256),
                             transforms.CenterCrop(256),
                             transforms.RandomCrop(224),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             normalize]
    train_dataset = datasets.ImageFolder(
        traindir,
        transform=transforms.Compose(transformations_train)
    )

    val_dataset = datasets.ImageFolder(
        valdir,
        transform=transforms.Compose(transformations_val)
    )
    print('### Prepare data ###')
    print(f'# of training data: {len(train_dataset)}')
    print(f'# of validation data: {len(val_dataset)}')
    print(f'# of classes: {train_dataset.classes}')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               drop_last=True,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=int(args.batch_size/2),
                                             shuffle=True,
                                             drop_last=True,
                                             num_workers=args.workers)

    # logistic regression
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        args.lr,
        momentum=args.momentum,
        weight_decay=10**args.weight_decay
    )

    # create logs
    exp_log = os.path.join(args.exp, 'log')
    if not os.path.isdir(exp_log):
        os.makedirs(exp_log)

    loss_log = Logger(os.path.join(exp_log, 'loss_log'))
    prec1_log = Logger(os.path.join(exp_log, 'prec1'))
    # prec5_log = Logger(os.path.join(exp_log, 'prec5'))

    for epoch in range(args.epochs):
        end = time.time()

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1, loss = validate(val_loader, model, criterion)

        loss_log.log(loss)
        prec1_log.log(prec1)
        # prec5_log.log(prec5)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if is_best:
            filename = 'model_best.pth.tar'
        else:
            filename = 'checkpoint.pth.tar'
        torch.save({
            'epoch': epoch + 1,
            'arch': 'alexnet',
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, os.path.join(args.exp, filename))

def forward(x, model):
    if hasattr(model, 'sobel') and model.sobel is not None:
        x = model.sobel(x)
    x = model.features(x)
    x = x.view(x.size(0), -1)
    x = model.classifier(x)
    x = model.top_layer(x)
    return x

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    roc_auc = AverageMeter()
    nmis = AverageMeter()
    # top5 = AverageMeter()

    # freeze also batch norm layers
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        #adjust learning rate
        learning_rate_decay(optimizer, len(train_loader) * epoch + i, args.lr)

        target = target.cuda()
        input_var = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target)
        # compute output

        output = forward(input_var, model)
        prob = F.softmax(output, dim=1)
        pred = prob.argmax(dim=1)
        loss = criterion(output, target_var)
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0].item(), input.size(0))
        target_np, prob_np, pred_np =  target.data.cpu().numpy(), prob.data.cpu().numpy(), pred.data.cpu().numpy()
        roc = roc_auc_score(target_np, prob_np, multi_class='ovr', labels=np.arange(len(train_loader.dataset.classes)))
        roc_auc.update(roc)
        nmi = normalized_mutual_info_score(target_np, pred_np)
        nmis.update(nmi)
        # top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and i % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'ROC {roc_auc.val:.3f} ({roc_auc.avg:.3f})\t'
                  'NMI {nmis.val:.3f} ({nmis.avg:.3f})'
                  .format(epoch, i, len(train_loader), batch_time=batch_time, loss=losses, top1=top1, roc_auc=roc_auc, nmis=nmis))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    roc_auc = AverageMeter()
    nmis = AverageMeter()
    # top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    softmax = nn.Softmax(dim=1).cuda()
    end = time.time()
    for i, (input_tensor, target) in enumerate(val_loader):
        if args.tencrops:
            bs, ncrops, c, h, w = input_tensor.size()
            input_tensor = input_tensor.view(-1, c, h, w)
        target = target.cuda()
        with torch.no_grad():
            input_var =input_tensor.cuda()
            target_var = target

            output = forward(input_var, model)
            prob = F.softmax(output, dim=1)
            pred = prob.argmax(dim=1)
            if args.tencrops:
                output_central = output.view(bs, ncrops, -1)[: , ncrops / 2 - 1, :]
                output = softmax(output)
                output = torch.squeeze(output.view(bs, ncrops, -1).mean(1))
            else:
                output_central = output

            prec1 = accuracy(output.data, target, topk=(1,))
            top1.update(prec1[0].item(), input_tensor.size(0))
            # top5.update(prec5[0], input_tensor.size(0))
            target_np, prob_np, pred_np =  target.data.cpu().numpy(), prob.data.cpu().numpy(), pred.data.cpu().numpy()
            roc = roc_auc_score(target_np, prob_np, multi_class='ovr', labels=np.arange(len(val_loader.dataset.classes)))
            roc_auc.update(roc)
            nmi = normalized_mutual_info_score(target_np, pred_np)
            nmis.update(nmi)
            loss = criterion(output_central, target_var)
        losses.update(loss.item(), input_tensor.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and i % 100 == 0:
            print('Validation: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'ROC {roc_auc.val:.3f} ({roc_auc.avg:.3f})\t'
                  'NMI {nmis.val:.3f} ({nmis.avg:.3f})'
                  .format(i, len(val_loader), batch_time=batch_time,
                   loss=losses, top1=top1, roc_auc=roc_auc, nmis=nmis))

    return top1.avg, losses.avg

if __name__ == '__main__':
    main()
