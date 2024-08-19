
import os
import pdb
import paddle
from PIL import Image
import os, random, math
import numpy as np
import paddle.nn.functional as F
import paddle.nn as nn
import copy
from paddle.io import DataLoader, Dataset
from paddle.vision import transforms

def setup_seed(seed):
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def test(model, test_loader, logger):
    model.eval()
    test_loss = 0
    correct = 0
    with paddle.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = paddle.argmax(output, axis=1)
            correct += (pred == target).numpy().sum()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    return acc, test_loss

def record_net_data_stats(y_train, net_dataidx_map, logger=None):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
        logger.info("Client ID %3d: %s" %(net_i, tmp))
    return net_cls_counts

def load_data(trn_dst, tst_dst, args=None):
    X_train = []; y_train = []
    # for idx in range(len(trn_dst)):
    for idx in range(200):
        X_train.append(trn_dst[idx][0]); y_train.append(trn_dst[idx][1])

    X_test = []; y_test = []

    # for idx in range(len(tst_dst)):
    for idx in range(200):
        X_test.append(tst_dst[idx][0]); y_test.append(tst_dst[idx][1])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    return X_train, y_train, X_test, y_test, trn_dst, tst_dst


def partition_data(trn_dst,tst_dst, partition, beta=0.4, num_users=5,logger=None,args=None):
    n_parties = num_users
    X_train, y_train, X_test, y_test, train_dataset, test_dataset = load_data(trn_dst, tst_dst, args)
    data_size = y_train.shape[0]
    # pdb.set_trace()
    if partition == "iid":
        idxs = np.random.permutation(data_size)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}


    elif partition == "dir":
        n_client = n_parties
        n_cls = np.unique(y_test).shape[0]
        alpha = beta

        n_data_per_clnt = len(y_train) / n_client
        clnt_data_list = np.random.lognormal(mean=np.log(n_data_per_clnt), sigma=args.sigma, size=n_client)
        clnt_data_list = (clnt_data_list / np.sum(clnt_data_list) * len(y_train)).astype(int)
        cls_priors = np.random.dirichlet(alpha=[alpha] * n_cls, size=n_client)
        prior_cumsum = np.cumsum(cls_priors, axis=1)

        idx_list = [np.where(y_train == i)[0] for i in range(n_cls)]
        cls_amount = [len(idx_list[i]) for i in range(n_cls)]
        net_dataidx_map = {}
        for j in range(n_client):
            net_dataidx_map[j] = []

        while np.sum(clnt_data_list) != 0:
            curr_clnt = np.random.randint(n_client)
            # If current node is full resample a client
            # print('Remaining Data: %d' %np.sum(clnt_data_list))
            if clnt_data_list[curr_clnt] <= 0:
                continue
            clnt_data_list[curr_clnt] -= 1
            curr_prior = prior_cumsum[curr_clnt]
            while True:
                cls_label = np.argmax(np.random.uniform() <= curr_prior)
                # Redraw class label if trn_y is out of that class
                if cls_amount[cls_label] <= 0:
                    continue
                else:
                    cls_amount[cls_label] -= 1
                    net_dataidx_map[curr_clnt].append(idx_list[cls_label][cls_amount[cls_label]])
                    break

    elif partition == "n_cls":
        n_client = n_parties
        n_cls = np.unique(y_test).shape[0]
        alpha = beta

        n_data_per_clnt = len(y_train) / n_client
        clnt_data_list = np.random.lognormal(mean=np.log(n_data_per_clnt), sigma=0, size=n_client)
        clnt_data_list = (clnt_data_list / np.sum(clnt_data_list) * len(y_train)).astype(int)
        cls_priors = np.zeros(shape=(n_client, n_cls))
        if n_client <= 5:
            for i in range(n_client):
                for j in range(int(alpha)):
                    cls_priors[i][int((alpha*i+j))%n_cls] = 1.0 / alpha
        else:
            for i in range(n_client):
                cls_priors[i][random.sample(range(n_cls), int(alpha))] = 1.0 / alpha

        prior_cumsum = np.cumsum(cls_priors, axis=1)

        idx_list = [np.where(y_train == i)[0] for i in range(n_cls)]
        cls_amount = [len(idx_list[i]) for i in range(n_cls)]
        net_dataidx_map = {}
        for j in range(n_client):
            net_dataidx_map[j] = []
        # pdb.set_trace()
        while np.sum(clnt_data_list) != 0:
            curr_clnt = np.random.randint(n_client)
            # If current node is full resample a client
            # print('Remaining Data: %d' %np.sum(clnt_data_list))
            if clnt_data_list[curr_clnt] <= 0:
                continue
            clnt_data_list[curr_clnt] -= 1
            curr_prior = prior_cumsum[curr_clnt]
            while True:
                cls_label = np.argmax(np.random.uniform() <= curr_prior)
                # Redraw class label if trn_y is out of that class
                if cls_amount[cls_label] <= 0:
                    continue
                else:
                    cls_amount[cls_label] -= 1
                    net_dataidx_map[curr_clnt].append(idx_list[cls_label][cls_amount[cls_label]])
                    break

    train_data_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logger)
    # pdb.set_trace()
    return train_dataset, test_dataset, net_dataidx_map, train_data_cls_counts


def feature_compute(model, test_loader, logger):
    classes = range(10)
    model.eval()
    mean_cls = paddle.zeros((10, 2048))
    std_cls = paddle.zeros((10, 2048))

    with paddle.no_grad():
        for data, target in test_loader:
            for cls in classes:
                output, feat = model(data[target == cls], return_features=True)
                mean_cls[cls] = paddle.mean(feat, axis=0)
                std_cls[cls] = paddle.std(feat, axis=0)

    return mean_cls, std_cls

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around PaddlePaddle Dataset class."""

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return paddle.to_tensor(image), paddle.to_tensor(label)

class WEnsemble(paddle.nn.Layer):
    def __init__(self, model_list, mdl_w_list):
        super(WEnsemble, self).__init__()
        self.models = model_list
        self.mdl_w_list = mdl_w_list

    def forward(self, x, return_features=False):
        logits_total = 0
        feat_total = 0
        for i in range(len(self.models)):
            if return_features:
                logits, feat = self.models[i](x, return_features=return_features)
                feat_total += self.mdl_w_list[i] * feat
            else:
                logits = self.models[i](x)
            logits_total += self.mdl_w_list[i] * logits
        logits_e = logits_total / paddle.sum(self.mdl_w_list)
        if return_features:
            feat_e = feat_total / paddle.sum(self.mdl_w_list)
            return logits_e, feat_e
        return logits_e

class ImageDataset(paddle.io.Dataset):
    def __init__(self, names, labels, img_transformer=None):
        self.names = names
        self.labels = labels
        self._img_transformer = img_transformer

    def get_image(self, index):
        name = self.names[index]
        img = Image.open(name).convert('RGB')
        return self._img_transformer(img)

    def __getitem__(self, index):
        img = self.get_image(index)
        return img, int(self.labels[index]), self.names[index]

    def __len__(self):
        return len(self.names)

def get_one_train_dataloader(args, data_path, domain):
    labels_name = os.listdir(data_path)
    labels_name.sort()
    names_train, labels_train = get_data(data_path, labels_name)
    train_img_transformer = get_train_transformers(args)
    train_dataset = ImageDataset(names_train, labels_train, img_transformer=train_img_transformer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    return train_loader

def get_train_transformers(args):
    img_tr = [transforms.RandomResizedCrop(args.imgsize, scale=(0.8, 1.0)),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              get_normalization_transform(args)]
    return transforms.Compose(img_tr)

def get_normalization_transform(args):
    transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return transform

def get_data(folder, labels_name):
    image_paths = get_paths(folder)
    image_labels = get_labels(folder, labels_name)
    return image_paths, image_labels

def get_labels(folder, labels_name):
    image_paths = get_paths(folder)
    image_labels = []
    for image_path in image_paths:
        image_labels.append(labels_name.index(os.path.split(os.path.split(image_path)[0])[1]))
    return image_labels

def get_paths(folder):
    image_paths = []
    for dir_path, dir_names, file_names in os.walk(folder):
        for file_name in file_names:
            if file_name.endswith(".jpg") or file_name.endswith(".png"):
                image_paths.append(os.path.join(dir_path, file_name))
    return image_paths

def get_test_transformers(args):
    img_tr = [transforms.Resize((args.imgsize, args.imgsize)),
              transforms.ToTensor(),
              get_normalization_transform(args)]
    return transforms.Compose(img_tr)

def get_one_test_dataloader(args, data_path):
    labels_name = os.listdir(data_path)
    labels_name.sort()
    names_test, labels_test = get_data(data_path, labels_name)
    test_img_transformer = get_test_transformers(args)
    test_dataset = ImageDataset(names_test, labels_test, img_transformer=test_img_transformer)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=0)
    return test_loader

def DG_train(model, train_loader, loss_fun, optimizer=None):
    device = 'gpu'
    model.to(device)
    model.train()
    num_data = 0
    class_correct = 0
    loss_all = 0
    it = 0
    for (x_batch, y_batch, _) in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.clear_grad()
        logits = model(x_batch)
        loss = loss_fun(logits, y_batch)
        loss_all += loss.item()
        class_correct += (logits.argmax(1) == y_batch).sum().item()
        num_data += x_batch.shape[0]
        loss.backward()
        optimizer.step()
    train_loss = loss_all / (it + 1)
    train_acc = class_correct / num_data
    return train_loss, train_acc

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        if 'num_batches_tracked' in key:
            w_avg[key] = w_avg[key] / len(w)
        else:
            w_avg[key] = paddle.divide(w_avg[key], paddle.to_tensor(float(len(w))))
    return w_avg

def class_test(model, test_loader, logger):
    model.eval()
    test_loss = 0
    correct = 0
    classes = None
    g_acc = 0
    with paddle.no_grad():
        try:
            for data, target, _ in test_loader:
                data, target = data.cuda(), target.cuda()
                output = model(data)
                if classes is None:
                    classes = range(output.shape[1])
                    correct_pred = {classname: 0 for classname in classes}
                    total_pred = {classname: 0 for classname in classes}
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = paddle.argmax(output, axis=1)
                correct += (pred == target).numpy().sum()
                
                # 遍历 target 和 pred 的每一个元素
                for label, p_l in zip(target.flatten().numpy(), pred.flatten().numpy()):
                    if int(label) == int(p_l):
                        correct_pred[int(label)] += 1
                    total_pred[int(label)] += 1

                del data, target
            paddle.device.cuda.empty_cache()
            g_acc = 100. * correct / len(test_loader.dataset)
            for classname, correct_count in correct_pred.items():
                accuracy = 100 * float(correct_count) / total_pred[classname]
                logger.info(
                    "Accuracy for class {:2d} is: {:.1f} ({:4d}/{:4d})%".format(classname, accuracy, correct_count,
                                                                                total_pred[classname]))
            logger.info(
                "Overall ACC is ({:4d}/{:4d}%) {:.1f} with loss {:.3f}".format(correct, len(test_loader.dataset), g_acc,
                                                                               test_loss))
            logger.info("-" * 20)
        except:
            for data, target in test_loader:
                data, target = data.cuda(), target.cuda()
                output = model(data)
                if classes is None:
                    classes = range(output.shape[1])
                    correct_pred = {classname: 0 for classname in classes}
                    total_pred = {classname: 0 for classname in classes}
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = paddle.argmax(output, axis=1)
                correct += (pred == target).numpy().sum()
                
                # 遍历 target 和 pred 的每一个元素
                for label, p_l in zip(target.flatten().numpy(), pred.flatten().numpy()):
                    if int(label) == int(p_l):
                        correct_pred[int(label)] += 1
                    total_pred[int(label)] += 1

                del data, target
            paddle.device.cuda.empty_cache()
            g_acc = 100. * correct / len(test_loader.dataset)
            for classname, correct_count in correct_pred.items():
                accuracy = 100 * float(correct_count) / total_pred[classname]
                logger.info("Accuracy for class {:2d} is: {:.1f} ({:4d}/{:4d})%".format(classname, accuracy, correct_count, total_pred[classname]))
            logger.info("Overall ACC is ({:4d}/{:4d}%) {:.1f} with loss {:.3f}".format(correct, len(test_loader.dataset), g_acc, test_loss))
            logger.info("-" * 20)

    return g_acc, test_loss

class Ensemble(nn.Layer):
    def __init__(self, model_list):
        super(Ensemble, self).__init__()
        self.models = model_list

    def forward(self, x, return_features=False):
        logits_total = 0
        feat_total = 0
        for i in range(len(self.models)):
            if return_features:
                logits, feat = self.models[i](x, return_features=return_features)
                feat_total += feat
            else:
                logits = self.models[i](x, return_features=return_features)
            logits_total += logits
        logits_e = logits_total / len(self.models)
        if return_features:
            feat_e = feat_total / len(self.models)
            return logits_e, feat_e
        return logits_e

    def feat_forward(self, x):
        out_total = 0
        for i in range(len(self.models)):
            out = self.models[i].feat_forward(x)
            out_total += out
        return out_total / len(self.models)