import argparse
import os
import pickle
import time
from tqdm import tqdm

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from util import AverageMeter, load_model
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from scipy.optimize import linear_sum_assignment as linear_assignment

parser = argparse.ArgumentParser(description="""Train linear classifier on top
                                 of frozen convolutional layers of an AlexNet.""")

parser.add_argument('--data', type=str, help='path to dataset', default='../../nature_imgs')
parser.add_argument('--model_path', type=str, help='path to model', default='runs/checkpoint.pth.tar')
parser.add_argument('--exp', type=str, default='runs/eval', help='exp folder')
parser.add_argument('--workers', default=2, type=int,
                    help='number of data loading workers (default: 2)')
parser.add_argument('--batch_size', default=64, type=int,
                    help='mini-batch size (default: 64)')
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

    # initialize test dict
    test_dict = dict()
    for i in range(len(val_dataset.classes)):
        test_dict[i] = dict()
        for cls in val_dataset.classes:
            test_dict[i][cls] = 0
    
    # initialize map
    class_map = dict()
    for i, cls in enumerate(val_dataset.classes):
        class_map[i] = cls

    # prepare data
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
    cf_matrix, nmi, ari, acc, prec, recall, f1 = validate(val_loader, model, test_dict, class_map)

    print('Confusion matrix')
    print(cf_matrix)
    print(f'Accuracy score: {acc:.4f}')
    print(f'Precision score: {prec:.4f}')
    print(f'Recall score: {recall:.4f}')
    print(f'F1 score: {f1:.4f}')
    print(f'Normalized mutual information score: {nmi:.4f}')
    print(f'Adjusted random score: {ari:.4f}')

    # display and save dict
    print(test_dict)
    with open('test_dict.pkl', 'wb+') as f:
        pickle.dump(test_dict, f)

    # write csv
    assert os.path.isfile('../result.csv')
    df = pd.read_csv('../result.csv', header=0, names=['method', 'nmi', 'ari', 'acc', 'prec', 'recall', 'f1'], 
                    dtype={'method': str, 'nmi': float, 'ari': float, 'acc': float, 'prec': float, 'recall': float, 'f1': float})
    df = df.append(pd.DataFrame({
        'method': ['deepcluster'],
        'nmi': [nmi],
        'ari': [ari],
        'acc': [acc],
        'prec': [prec],
        'recall': [recall],
        'f1': [f1]
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

def validate(val_loader, model, test_dict, class_map):
    cost = np.zeros(shape=(len(val_loader.dataset.classes), len(val_loader.dataset.classes)))
    targets, preds = list(), list()
    # switch to evaluate mode
    model.eval()
    
    for (input_tensor, target) in tqdm(val_loader):
        target = target.cuda()
        with torch.no_grad():
            input_var =input_tensor.cuda()

            output = forward(input_var, model)
            prob = F.softmax(output, dim=1)
            pred = prob.argmax(dim=1)

            target_np, pred_np = target.data.cpu().numpy(), pred.data.cpu().numpy()
            targets.append(target_np)
            preds.append(pred_np)

            # update test_dict
            for i in range(target_np.shape[0]):
                cls = class_map[target_np[i]]
                test_dict[pred_np[i]][cls] += 1

            # # update metrics
            # cost += confusion_matrix(target_np, pred_np, labels=np.arange(0, len(val_loader.dataset.classes)))
    
    preds, targets = np.array(preds).reshape(-1), np.array(targets).reshape(-1)
    # update confusion matrix
    cost = confusion_matrix(targets, preds)
    _, col_ind = linear_assignment(cost, maximize=True)
    
    # update prediction according to result from linear assignment
    preds_adj = np.zeros_like(preds)
    for i in range(len(val_loader.dataset.classes)):
        preds_adj[preds == col_ind[i]] = i

    cf_matrix = confusion_matrix(targets, preds_adj)
    # plot heatmap
    sns_plot = sns.heatmap(cf_matrix, annot=True)
    sns_plot.figure.savefig('heatmap.png')

    nmi = normalized_mutual_info_score(targets, preds_adj)
    ari = adjusted_rand_score(targets, preds_adj)
    acc = accuracy_score(targets, preds_adj)
    prec = precision_score(targets, preds_adj, average='macro')
    recall = recall_score(targets, preds_adj, average='macro')
    f1 = f1_score(targets, preds_adj, average='macro')
    return cf_matrix, nmi, ari, acc, prec, recall, f1

if __name__ == '__main__':
    main()
