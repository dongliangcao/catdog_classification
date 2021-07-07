import argparse, os
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from models.vgg import VGG16

import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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

    # prepare model
    print('##### Prepare Model #####')
    model = VGG16(num_classes=len(test_dataset.classes), pretrained_path=args.model_path).cuda()

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
    pca = PCA(n_components=args.pca)
    feats = pca.fit_transform(feats)

    # perform KMeans to cluster features
    kmean = KMeans(n_clusters=len(test_dataset.classes))
    preds = kmean.fit_predict(feats)

    # calculate accuracy and confusion matrix
    nmi = normalized_mutual_info_score(targets, preds)
    print(f'Normalized mutual information score: {nmi:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test the pretrained model on test data')
    parser.add_argument('--data_dir', help='folder stores the data', default='../../nature_imgs/')
    parser.add_argument('--model_path', help='file stores pretrained model', default='../pretrained_model/vgg16-397923af.pth')
    parser.add_argument('--batch_size', help='batch size (default: 32)', type=int, default=32)
    parser.add_argument('--pca', type=int, default=32, help='the number of reduced dimensions')
    args = parser.parse_args()
    test(args)