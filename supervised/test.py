import argparse, os
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from models.vgg import VGG16

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

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
    model = VGG16(num_classes=len(test_dataset.classes)).cuda()
    # load checkpoint
    model.load_state_dict(torch.load(args.model_path)['state_dict'])

    # compute accuracy
    print('##### Compute Metrics on Test Data #####')
    test_acc = 0.0
    test_auc = 0.0
    test_confusion_mat = 0.0
    model.eval()
    for (imgs, target) in tqdm(test_loader):
        imgs, target = imgs.cuda(), target.cuda()
        with torch.no_grad():
            pred = model(imgs)
            prob = F.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1)

            target, prob, pred = target.data.cpu().numpy(), prob.data.cpu().numpy(), pred.data.cpu().numpy()
            auc = roc_auc_score(target, prob, multi_class='ovr')
            acc = accuracy_score(target, pred)
            confusion_mat = confusion_matrix(target, pred)
        test_acc += acc
        test_auc += auc
        test_confusion_mat += confusion_mat
    
    test_acc /= len(test_loader)
    test_auc /= len(test_loader)

    print(f'Accuracy on test data: {test_acc:.4f}')
    print(f'ROC AUC score on test data: {test_auc:.4f}')
    print('Confusion matrix on test data')
    print(f'{test_dataset.classes}')
    print(test_confusion_mat)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test the trained model on test data')
    parser.add_argument('--model_path', help='the saved checkpoint for trained model')
    parser.add_argument('--data_dir', help='folder stores the data', default='../../nature_imgs/')
    parser.add_argument('--batch_size', help='batch size (default: 32)', type=int, default=32)

    args = parser.parse_args()
    test(args)