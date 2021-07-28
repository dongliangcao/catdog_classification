import argparse, os, pickle
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from models.vgg import VGG16

import numpy as np
import pandas as pd
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def test(args):
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
    
    # prepare test dataset
    test_dataset = ImageFolder(os.path.join(args.data_dir, 'test'), transform=transform)
    print(f'Num of test data: {len(test_dataset)}')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2, drop_last=True)

    # initialize test dict
    test_dict = dict()
    for i in range(len(test_dataset.classes)):
        test_dict[i] = dict()
        for cls in test_dataset.classes:
            test_dict[i][cls] = 0
    
    # initialize map
    class_map = dict()
    for i, cls in enumerate(test_dataset.classes):
        class_map[i] = cls

    # prepare model
    print('##### Prepare Model #####')
    if args.pretrained:
        model = VGG16(num_classes=len(test_dataset.classes), pretrained_path=args.model_path).cuda()
    else:
        model = VGG16(num_classes=len(test_dataset.classes), pretrained_path=None).cuda()
    # get features for each images
    print('##### Compute Metrics on Test Data #####')
    feats = []
    targets = []
    model.eval()
    for (imgs, target) in tqdm(test_loader):
        imgs, target = imgs.cuda(), target.cuda()
        with torch.no_grad():
            feat = model(imgs)
        feats.append(feat.data.cpu().numpy())
        targets.append(target.data.cpu().numpy())
    
    # concatenate all features and targets
    feats = np.concatenate(feats, axis=0)
    targets = np.concatenate(targets, axis=0)

    # perform PCA to reduce dimension
    if args.dim >= 4:
        tsne = TSNE(n_components=args.dim, init='pca', method='exact')
    else:
        tsne = TSNE(n_components=args.dim, init='pca')
    feats = tsne.fit_transform(feats)

    # perform KMeans to cluster features
    kmean = KMeans(n_clusters=len(test_dataset.classes))
    preds = kmean.fit_predict(feats)

    # update test dict
    for i in range(preds.shape[0]):
        cls = class_map[targets[i]]
        test_dict[preds[i]][cls] += 1

    # display and save dict
    print(test_dict)
    with open('test_dict.pkl', 'wb+') as f:
        pickle.dump(test_dict, f)

    # calculate accuracy and confusion matrix
    nmi = normalized_mutual_info_score(targets, preds)
    ari = adjusted_rand_score(targets, preds)
    print(f'Normalized mutual information score: {nmi:.4f}')
    print(f'Adjusted random score: {ari:.4f}')

    # write csv
    df = pd.read_csv('../result.csv', header=0, names=['method', 'nmi', 'ari'], dtype={'method': str, 'nmi': float, 'ari': float})
    if args.pretrained:
        df = df.append(pd.DataFrame({
            'method': ['unsupervised_pretrain'],
            'nmi': [nmi],
            'ari': [ari]
        }, index=[len(df.index)]))
    else:
        df = df.append(pd.DataFrame({
            'method': ['unsupervised'],
            'nmi': [nmi],
            'ari': [ari]
        }, index=[len(df.index)]))
    print(df)
    df.to_csv('../result.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test the pretrained model on test data')
    parser.add_argument('--data_dir', help='folder stores the data', default='../../nature_imgs/')
    parser.add_argument('--model_path', help='file stores pretrained model', default='../pretrained_model/vgg16-397923af.pth')
    parser.add_argument('--batch_size', help='batch size (default: 32)', type=int, default=32)
    parser.add_argument('--dim', type=int, default=2, help='the number of reduced dimensions (default: 2)')
    parser.add_argument('--pretrained', action='store_true', help='Use pre-trained VGG network in ImageNet')
    args = parser.parse_args()
    test(args)