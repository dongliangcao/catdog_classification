"""
Training pretrained VGG16 for cat dog classification
"""
import argparse, os
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from models.vgg import VGG16

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def train(args):
    # reproducibility
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # prepare data
    print('##### Prepare Data #####')
    # preprocessing
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           normalize])
    
    # prepare train and val dataset
    dataset = ImageFolder(os.path.join(args.data_dir, 'train'), transform=transform)
    train_dataset, val_dataset = random_split(dataset, [int(0.8*len(dataset)), len(dataset) - int(0.8*len(dataset))])
    print(f'Num of train data: {len(train_dataset)}, Num of val data: {len(val_dataset)}')
    print(f'Num of classes {len(train_dataset.dataset.classes)}')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    # prepare model
    print('##### Prepare Model #####')
    if args.pretrained:
        model = VGG16(num_classes=len(train_dataset.dataset.classes), pretrained_path=args.model_path).cuda()
        # freeze the features layers
        for param in model.features.parameters():
            param.requires_grad = False
    else:
        model = VGG16(num_classes=len(train_dataset.dataset.classes), pretrained_path=None).cuda()
    
    # prepare loss and optimizer
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss().cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # logger
    logdir = os.path.join(args.log_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    logger = SummaryWriter(log_dir=logdir)
    savedir = os.path.join(logdir, 'checkpoints')
    os.makedirs(savedir, exist_ok=True)
    logfile = os.path.join(logdir, 'log.txt')

    # start training
    print('##### Start Training #####')
    for epoch in range(args.start_epoch, args.epochs):
        # start training
        model.train()
        train_loss = 0.0
        val_loss = 0.0
        val_acc = 0.0
        
        step = 0
        print('Train')
        for (imgs, target) in tqdm(train_loader):
            imgs, target = imgs.cuda(), target.cuda()
            # make prediction
            pred = model(imgs)
            # compute loss
            loss = criterion(pred, target)
            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # log
            logger.add_scalar('train_loss', loss, global_step=epoch*len(train_loader)+step)
            train_loss += loss.item()
            step += 1
        # end training
        train_loss /= len(train_loader)
        logger.add_scalars('losses', {'train_loss': train_loss}, global_step=epoch)

        # start validation
        print('Val')
        model.eval()
        for (imgs, target) in tqdm(val_loader):
            imgs, target = imgs.cuda(), target.cuda()
            with torch.no_grad():
                # pred
                pred = model(imgs)
                # compute loss
                loss = criterion(pred, target)
                # compute accuracy
                pred = torch.argmax(pred, dim=1)
                target, pred = target.data.cpu().numpy(), pred.data.cpu().numpy()
                acc = accuracy_score(target, pred)
            
            val_acc += acc
            val_loss += loss.item()
        # end validation
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        logger.add_scalars('losses', {'val_loss': val_loss}, global_step=epoch)
        logger.add_scalar('val', val_acc, global_step=epoch)

        if (epoch - 1) % args.print_every_epoch == 0:
            with open(logfile, 'a+') as f:
                print(f'Epoch {epoch}/{args.epochs}, Train loss {train_loss:.4f}, Val loss {val_loss:.4f}, Val acc {val_acc:.4f}', file=f)
        if (epoch - 1) % args.save_every_epoch == 0:
            torch.save({'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()}, os.path.join(savedir, f'epoch{epoch}.pth'))

    # save model
    print('##### save model #####')
    torch.save({'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()}, os.path.join(savedir, 'final.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Supervised training of VGG16 for cat dog classification')
    parser.add_argument('--data_dir', help='folder stores the data', default='../../nature_imgs/')
    parser.add_argument('--model_path', help='file stores pretrained model', default='../pretrained_model/vgg16-397923af.pth')
    parser.add_argument('--pretrained', help='use pre-trained VGG from ImageNet', action='store_true')
    parser.add_argument('--epochs', help='number of total epochs to run (default: 10)', type=int, default=10)
    parser.add_argument('--start_epoch', help='number of epoch to start (default: 0)', type=int, default=0)
    parser.add_argument('--batch_size', help='batch size (default: 16)', type=int, default=16)
    parser.add_argument('--lr', help='learning rate (default: 0.0001)', type=float, default=0.0001)
    parser.add_argument('--log_dir', help='folder stores the training logs (default: runs/)', default='runs/')
    parser.add_argument('--resume', help='resume from a given checkpoint', default=None)
    parser.add_argument('--print_every_epoch', help='the frequency of print log (default: 1)', type=int, default=1)
    parser.add_argument('--save_every_epoch', help='the frequency of model saving (default: 5)', type=int, default=5)
    args = parser.parse_args()
    train(args)