import argparse
import os
import random
import shutil
import time
import warnings
import pdb
import copy
import sys
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader
from paddle.vision import datasets, transforms
import datafree
import registry
from utils_fl import *
import datafree

def setup_seed(seed):
    paddle.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

parser = argparse.ArgumentParser(description='Data-free Knowledge Distillation')

# Data Free
parser.add_argument('--method', required=True)
parser.add_argument('--adv', default=1.0, type=float, help='scaling factor for adversarial distillation')
parser.add_argument('--bn', default=0, type=float, help='scaling factor for BN regularization')
parser.add_argument('--ohg', default=1.0, type=float, help='scaling factor for one hot loss (cross entropy)')
parser.add_argument('--save_dir', default='run/synthesis', type=str)
parser.add_argument('--batchonly', action='store_true')
parser.add_argument('--batchused', action='store_true')
parser.add_argument('--sam', default=0.0, type=float)
parser.add_argument('--his', action='store_false')
parser.add_argument('--wdc', default=0.99, type=float)

# 参数设置
parser.add_argument('--mv', default=1.0, type=float)
parser.add_argument('--weighted', action='store_true')
parser.add_argument('--mu', default=0.01, type=float)
parser.add_argument('--wa_steps', default=1, type=int)

# 基本设置
parser.add_argument('--data_root', default='/home/dairong/Co-Boosting-main/Data')
parser.add_argument('--fl_model', default='')
parser.add_argument('--teacher', default='resnet18')
parser.add_argument('--student', default='resnet18')
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--kd_lr', default=0.1, type=float, help='initial learning rate for KD')
parser.add_argument('--lr_decay_milestones', default="120,150,180", type=str, help='milestones for learning rate decay')
parser.add_argument('--lr_g', default=1e-3, type=float, help='initial learning rate for generation')
parser.add_argument('--kd_T', default=4, type=float)
parser.add_argument('--odseta', default=8, type=float)
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--g_steps', default=1, type=int, metavar='N', help='number of iterations for generation')
parser.add_argument('--kd_steps', default=400, type=int, metavar='N', help='number of iterations for KD after generation')

# co-boosting inherent
parser.add_argument('--ods', action='store_true', help='是否在KD阶段使用ODS技术')
parser.add_argument('--hast', action='store_true', help='是否使用modified CE loss')
parser.add_argument('--hs', default=1.0, type=float, metavar='N', help='number of total iterations in each epoch')
parser.add_argument('--evaluate_only', action='store_true', help='evaluate model on validation set')
parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--synthesis_batch_size', default=None, type=int, metavar='N', help='mini-batch size for synthesis')

# 其他设置
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')
parser.add_argument('--identity', default='')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float, metavar='W', help='weight decay', dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=20, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--imgsize', default=32, type=int, help='sam')

best_acc1 = 0

def main():
    args = parser.parse_args()
    setup_seed(args.seed)
    main_worker(args)

def main_worker(args):
    global best_acc1
    paddle.set_device('gpu')
    
    # Logger 设置 (略过部分代码)
    args.his = not args.batchonly
    log_name = '%s_%s_adv%s_ohg%s_KDlr%s_KDT%s_GANlr%s_GANs%s_Epoch%s_seed%s' % (
        args.method, args.student, args.adv, args.ohg, args.kd_lr, args.kd_T, args.lr_g, args.g_steps, args.epochs,
        args.seed)
    if args.method == 'co_boosting':
        args.weighted = True
        args.hast = True
        args.ods = True
        log_name += '_eta' + str(args.odseta) 
        log_name += '_hast' + str(args.hs)
        args.odseta = args.odseta / 255
        log_name += '_wmu' + str(args.mu) + '_was' + str(args.wa_steps) + '_wdc' + str(args.wdc)

    prefix_path = '/home/dairong/coboost/'

    args.identity = log_name
    args.logger = datafree.utils.logger.get_logger(log_name, output=prefix_path + 'LOG/%s/%s.txt' % (
    args.fl_model, args.identity))
    os.makedirs(prefix_path + 'checkpoints/%s/' % (args.fl_model), exist_ok=True)
    for k, v in datafree.utils.flatten_dict(vars(args)).items():  # print args
        args.logger.info("%s: %s" % (k, v))
    
    num_classes = None; ori_dataset = None; val_dataset = None; val_loader = None; evaluator = None; method_transform = None
    num_classes, ori_dataset, val_dataset = registry.get_dataset(name=args.dataset, data_root=args.data_root)
    method_transform = ori_dataset.transform
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    evaluator = datafree.evaluators.classification_evaluator(val_loader)
    student = registry.get_model(args.student, num_classes=num_classes)
    local_weights = paddle.load(prefix_path + 'checkpoints/FL_pretrain/%s.pkl' % (args.fl_model))
    args.normalizer = normalizer = datafree.utils.Normalizer(**registry.NORMALIZE_DICT[args.dataset])

    model_list = []
    for i in range(len(local_weights)):
        net = copy.deepcopy(registry.get_model(args.teacher, num_classes=num_classes))
        net.set_dict(local_weights[i])
        net.eval()
        model_list.append(net)
    ensemble_model = Ensemble(model_list)

    ww = paddle.ones(shape=[len(model_list), 1]) * (1.0 / len(model_list))
    ensemble_model = WEnsemble(model_list, ww)
    
    student = student.to('gpu')
    teacher = ensemble_model.to('gpu')
    args.logger.info("NOW TESTING TEACHER MODEL")
    class_test(ensemble_model, val_loader, args.logger)
    
    if args.synthesis_batch_size is None:
        args.synthesis_batch_size = args.batch_size

    if args.dataset in ["mnist",'fmnist']:
        real_img_size = (1, 32, 32); nc = 1
    else:
        real_img_size = (3, 32, 32); nc = 3

    if args.method == 'co_boosting':
        args.save_dir = prefix_path + 'checkpoints/%s/%s/' % (args.fl_model, args.identity)
        os.makedirs(prefix_path + 'checkpoints/%s/%s/' % (args.fl_model, args.identity), exist_ok=True)
        nz = 256
        generator = datafree.models.generator.Generator(nz=nz, ngf=64, img_size=32, nc=nc)
        generator = generator.to('gpu')
        criterion = None
        synthesizer = datafree.synthesis.COBOOSTSynthesizer(
            teacher=teacher, mdl_list=model_list, student=student, generator=generator, nz=nz, num_classes=num_classes,
            img_size=real_img_size, iterations=args.g_steps, lr_g=args.lr_g,
            synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size,
            adv=args.adv, bn=args.bn, oh = args.ohg, criterion=criterion,
            transform=method_transform,
            save_dir=args.save_dir, normalizer=args.normalizer, args=args)
    
    optimizer = paddle.optimizer.Momentum(parameters=student.parameters(), learning_rate=args.kd_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=args.kd_lr, T_max=args.epochs)
    
    if args.evaluate_only:
        student.eval()
        eval_results = evaluator(student)
        print('[Eval] Acc={acc:.4f}'.format(acc=eval_results['Acc']))
        return
    
    for epoch in range(args.epochs):
        args.current_epoch = epoch
        vis_results = synthesizer.synthesize(cur_ep=epoch)
        del teacher
        teacher = synthesizer.teacher
        teacher.eval()
        
        kd_criterion = paddle.nn.KLDivLoss(reduction='batchmean')
        if args.method ==  'co_boosting':
            cb_kd_train(synthesizer, [student, teacher], kd_criterion, optimizer, args)
        
        student.eval()
        eval_results = evaluator(student)
        (acc1, acc5), val_loss = eval_results['Acc'], eval_results['Loss']
        args.logger.info('[Eval] Epoch={current_epoch} Acc@1={acc1:.4f} Acc@5={acc5:.4f} Loss={loss:.4f} Lr={lr:.4f}'
                        .format(current_epoch=args.current_epoch, 
                                acc1=float(acc1),  # 将 Tensor 转换为 float
                                acc5=float(acc5),  # 将 Tensor 转换为 float
                                loss=float(val_loss),  # 将 Tensor 转换为 float
                                lr=float(optimizer.get_lr())))  # 将学习率转换为 float

        if epoch % args.print_freq == 0 or epoch == args.epochs - 1:
            class_test(student, val_loader, args.logger)
            args.logger.info(teacher.mdl_w_list)
            args.logger.info("Now testing weighted ENSEMBLE")
            class_test(teacher, val_loader, args.logger)
        
        scheduler.step()
        
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.student,
            'state_dict': student.state_dict(),
            'best_acc1': float(best_acc1),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best)

def cb_kd_train(synthesizer, model, criterion, optimizer, args):
    student, teacher = model
    student.train()
    teacher.eval()
    for idx, (images, labels) in enumerate(synthesizer.get_data(labeled=True)):
        optimizer.clear_grad()
        images = paddle.to_tensor(images, stop_gradient=False)
        labels = paddle.to_tensor(labels)
        
        try:
            random_w = paddle.uniform(shape=teacher(images).shape, min=-1.0, max=1.0)
            loss_ods = paddle.sum(random_w * F.softmax(teacher(images) / 4))
        except:
            random_w = paddle.uniform(shape=teacher(images).shape, min=-1.0, max=1.0)
            loss_ods = paddle.sum(random_w * F.softmax(teacher(images) / 4))
        
        loss_ods.backward()
        images = (paddle.sign(images.grad) * args.odseta + images).detach()
        
        s_out = student(images)
        with paddle.no_grad():
            t_out = teacher(images)
            loss_ce = F.cross_entropy(s_out, labels)
        
        loss_kd = criterion(s_out, t_out)
        loss_kd.backward()
        optimizer.step()

def save_checkpoint(state, is_best, filename='checkpoint.pdparams'):
    if is_best:
        paddle.save(state, filename)

if __name__ == '__main__':
    main()