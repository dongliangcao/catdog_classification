# catdog_classification
binary classification for cat and dog


### Dependencies

* Python 3.6
* PyTorch 1.0.0
* torchvision 0.2.1
* tensorboardX 2.3
* faiss 1.7.0
* numpy, scipy, scikit-learn


### Usage

1. Download the ImageNet-pretrained weights of VGG16 network from `torchvision`: [https://download.pytorch.org/models/vgg16-397923af.pth](https://download.pytorch.org/models/vgg16-397923af.pth) and put it under `./pretrained_model` folder.

2. Activate the virtual environment `conda activate deepcluster`.

3. Depending on which way you want to choose (supervised, deepcluster), moving to the corresponding folder, modify the configuration in the `train.py` and run `python train.py`

4. During the training process, you can visualize the losses via TensorBoard in your browser by running `tensorboard --logdir runs/`. 

5. After finishing the training, modify the configuration in the `test.py` and run `python test.py --model_path [MODEL_PATH]`.
