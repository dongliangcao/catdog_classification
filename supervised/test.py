import argparse, os

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from models.vgg import VGG16

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
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    # prepare model
    print('##### Prepare Model #####')
    model = VGG16().cuda()
    # load checkpoint
    model.load_state_dict(torch.load(args.model_path)['state_dict'])

    # compute accuracy
    print('##### Compute Accuracy #####')
    test_acc = 0.0
    model.eval()
    for (imgs, target) in test_loader:
        imgs, target = imgs.cuda(), target.cuda()
        with torch.no_grad():
            pred = model(imgs)
            pred = torch.argmax(pred, dim=1)
            acc = (pred == target).float().mean()
        test_acc += acc.item()
    test_acc /= len(test_loader)
    print(f'Accuracy on test data: {test_acc:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test the trained model on test data')
    parser.add_argument('--model_path', help='the saved checkpoint for trained model')
    parser.add_argument('--data_dir', help='folder stores the data', default='../../catdog/')
    parser.add_argument('--batch_size', help='batch size (default: 32)', type=int, default=32)

    args = parser.parse_args()
    test(args)