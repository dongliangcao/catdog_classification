import argparse, os, pickle
from tqdm import tqdm

import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from models.vgg import VGG16

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, normalized_mutual_info_score, adjusted_rand_score

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
    model = VGG16(num_classes=len(test_dataset.classes)).cuda()
    # load checkpoint
    model.load_state_dict(torch.load(args.model_path)['state_dict'])

    # logfile
    logfile = os.path.join(os.path.dirname(args.model_path), '../log.txt')
    assert os.path.isfile(logfile)

    # compute accuracy
    print('##### Compute Metrics on Test Data #####')
    test_acc = 0.0
    test_auc = 0.0
    test_confusion_mat = 0.0
    test_nmi = 0.0
    test_ari = 0.0
    model.eval()
    for (imgs, target) in tqdm(test_loader):
        imgs, target = imgs.cuda(), target.cuda()
        with torch.no_grad():
            pred = model(imgs)
            prob = F.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1)

            target_np, prob_np, pred_np = target.data.cpu().numpy(), prob.data.cpu().numpy(), pred.data.cpu().numpy()

            # update test dict
            for i in range(target_np.shape[0]):
                cls = class_map[target_np[i]]
                test_dict[pred_np[i]][cls] += 1

            # update metrics
            auc = roc_auc_score(target_np, prob_np, multi_class='ovr')
            acc = accuracy_score(target_np, pred_np)
            confusion_mat = confusion_matrix(target_np, pred_np)
            nmi = normalized_mutual_info_score(target_np, pred_np)
            ari = adjusted_rand_score(target_np, pred_np)
        test_acc += acc
        test_auc += auc
        test_confusion_mat += confusion_mat
        test_nmi += nmi
        test_ari += ari
    
    test_acc /= len(test_loader)
    test_auc /= len(test_loader)
    test_nmi /= len(test_loader)
    test_ari /= len(test_loader)

    with open(logfile, 'a') as f:
        print(f'Accuracy on test data: {test_acc:.4f}', file=f)
        print(f'ROC AUC score on test data: {test_auc:.4f}', file=f)
        print(f'NMI score on test data: {test_nmi:.4f}', file=f)
        print(f'ARI score on test data: {test_ari:.4f}', file=f)
        print('Confusion matrix on test data', file=f)
        print(f'{test_dataset.classes}', file=f)
        print(test_confusion_mat, file=f)

    # display and save dict
    print(test_dict)
    with open('test_dict.pkl', 'wb+') as f:
        pickle.dump(test_dict, f)

    # write csv
    assert os.path.isfile('../result.csv')
    df = pd.read_csv('../result.csv', header=0, names=['method', 'nmi', 'ari'], dtype={'method': str, 'nmi': float, 'ari': float})
    df = df.append(pd.DataFrame({
        'method': ['supervised'],
        'nmi': [test_nmi],
        'ari': [test_ari]
    }, index=[len(df.index)]))
    print(df)
    df.to_csv('../result.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test the trained model on test data')
    parser.add_argument('--model_path', required=True, help='the saved checkpoint for trained model')
    parser.add_argument('--data_dir', help='folder stores the data', default='../../nature_imgs/')
    parser.add_argument('--batch_size', help='batch size (default: 32)', type=int, default=32)

    args = parser.parse_args()
    test(args)