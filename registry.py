import os
import paddle
import paddle.nn as nn
import paddle.vision.transforms as T
from PIL import Image
from paddle.io import Dataset, DataLoader
import pdb
from datafree.models import classifiers_p

NORMALIZE_DICT = {
    'mnist':    dict( mean=(0.5,),                std=(0.5,) ),
    'fmnist': dict(mean=(0.5,), std=(0.5,)),
    'cifar10': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'cifar100': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'tiny': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'svhn': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
}


MODEL_DICT = {
    'lenet': classifiers_p.lenet.LeNet5,
}

def get_dataset(name: str, data_root: str='data', return_transform=False, split=['A', 'B', 'C', 'D']):
    name = name.lower()
    data_root = os.path.expanduser(data_root)

    if name == 'mnist':
        num_classes = 10
        train_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        data_root = os.path.join(data_root)
        # pdb.set_trace()
        train_dst = paddle.vision.datasets.MNIST(mode='train', transform=train_transform)
        val_dst = paddle.vision.datasets.MNIST(mode='test', transform=val_transform)
        

    elif name == 'fmnist':
        num_classes = 10
        train_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        data_root = os.path.join(data_root)
        train_dst = paddle.vision.datasets.FashionMNIST(data_root, mode='train', transform=train_transform)
        val_dst = paddle.vision.datasets.FashionMNIST(data_root, mode='test', transform=val_transform)

    elif name == 'cifar10':
        num_classes = 10
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        data_root = os.path.join(data_root)
        train_dst = paddle.vision.datasets.Cifar10(data_root, mode='train', transform=train_transform)
        val_dst = paddle.vision.datasets.Cifar10(data_root, mode='test', transform=val_transform)
    elif name == 'cifar100':
        num_classes = 100
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        data_root = os.path.join(data_root)
        train_dst = paddle.vision.datasets.Cifar100(data_root, mode='train', transform=train_transform)
        val_dst = paddle.vision.datasets.Cifar100(data_root, mode='test', transform=val_transform)
    elif name == 'svhn':
        num_classes = 10
        train_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        data_root = os.path.join(data_root, 'torchdata')
        train_dst = paddle.vision.datasets.SVHN(data_root, split='train', transform=train_transform)
        val_dst = paddle.vision.datasets.SVHN(data_root, split='test', transform=val_transform)
    elif name == "tiny":
        num_classes = 200
        transform = T.Compose([T.Resize(64),
                               T.ToTensor(),
                               T.Normalize(mean=[0.5, 0.5, 0.5],
                                           std=[0.5, 0.5, 0.5])])
        root_dir = "/gdata/dairong/fedsam/Data/Raw/tiny-imagenet-200/"
        trn_img_list, trn_lbl_list, tst_img_list, tst_lbl_list = [], [], [], []
        trn_file = os.path.join(root_dir, 'train_list.txt')
        tst_file = os.path.join(root_dir, 'val_list.txt')
        with open(trn_file) as f:
            line_list = f.readlines()
            for line in line_list:
                img, lbl = line.strip().split()
                trn_img_list.append(img)
                trn_lbl_list.append(int(lbl))
        with open(tst_file) as f:
            line_list = f.readlines()
            for line in line_list:
                img, lbl = line.strip().split()
                tst_img_list.append(img)
                tst_lbl_list.append(int(lbl))

        train_dst = DatasetFromDir(img_root=root_dir, img_list=trn_img_list, label_list=trn_lbl_list,
                                   transformer=transform)
        val_dst = DatasetFromDir(img_root=root_dir, img_list=tst_img_list, label_list=tst_lbl_list,
                                 transformer=transform)

    else:
        raise NotImplementedError
    if return_transform:
        return num_classes, train_dst, val_dst, train_transform, val_transform
    return num_classes, train_dst, val_dst

def get_model(name: str, num_classes, pretrained=False, **kwargs):
    if 'imagenet' in name:
        model = IMAGENET_MODEL_DICT[name](pretrained=pretrained)
        if num_classes!=1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'deeplab' in name:
        model = SEGMENTATION_MODEL_DICT[name](num_classes=num_classes, pretrained_backbone=kwargs.get('pretrained_backbone', False))
    else:
        model = MODEL_DICT[name](num_classes=num_classes)
    return model 

class DatasetFromDir(Dataset):
    def __init__(self, img_root, img_list, label_list, transformer):
        super(DatasetFromDir, self).__init__()
        self.root_dir = img_root
        self.img_list = img_list
        self.label_list = label_list
        self.size = len(self.img_list)
        self.transform = transformer

    def __getitem__(self, index):
        img_name = self.img_list[index % self.size]
        img_path = os.path.join(self.root_dir, img_name)
        img_id = self.label_list[index % self.size]

        img_raw = Image.open(img_path).convert('RGB')
        img = self.transform(img_raw)
        return img, img_id

    def __len__(self):
        return len(self.img_list)
