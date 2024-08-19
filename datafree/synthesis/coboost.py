import pdb

import random
import numpy as np

from .base import BaseSynthesis
from datafree.hooks import DeepInversionHook
from datafree.utils import ImagePool, DataIter, dense_kldiv

from utils_fl import *


class COBOOSTSynthesizer(BaseSynthesis):
    def __init__(self, teacher, mdl_list, student, generator, nz, num_classes, img_size, save_dir, iterations=1,
                 lr_g=1e-3, synthesis_batch_size=128, sample_batch_size=128,
                 adv=0, bn=0, oh=0, balance=0, criterion=None,transform=None,
                 normalizer=None,
                 # TODO: FP16 and distributed training
                 autocast=None, use_fp16=False, distributed=False, args=None):
        super(COBOOSTSynthesizer, self).__init__(teacher, student)
        self.mdl_list = mdl_list
        self.args = args
        assert len(img_size) == 3, "image size should be a 3-dimension tuple"
        self.img_size = img_size
        self.iterations = iterations
        self.save_dir = save_dir
        self.transform = transform

        self.nz = nz
        self.num_classes = num_classes
        if criterion is None:
            criterion = dense_kldiv
        self.criterion = criterion
        self.normalizer = normalizer
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size

        self.data_pool = ImagePool(root=self.save_dir)
        self.data_iter = None
        # scaling factors
        self.lr_g = lr_g
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.balance = balance
        # generator
        self.generator = generator.to('gpu')
        self.generator.train()
        self.distributed = distributed
        self.use_fp16 = use_fp16
        self.autocast = autocast  # for FP16
        self.hooks = []
        # hooks for deepinversion regularization

        for m_list in self.mdl_list:
            for m in m_list.sublayers():
                if isinstance(m, nn.BatchNorm2D):
                    self.hooks.append(DeepInversionHook(m))
        self.clnt_cls_weight_matrix = paddle.ones(shape=(len(self.mdl_list), self.num_classes))

    def synthesize(self, cur_ep=None):
        self.student.eval()
        self.generator.train()
        self.teacher.eval()
        for m in self.mdl_list:
            m.eval()

        if self.bn == 0:
            self.hooks = []

        best_cost = 1e6
        # 生成随机张量 z
        # 创建 z 参数
        z = paddle.create_parameter(
        shape=[self.synthesis_batch_size, self.nz], 
        dtype='float32', 
        default_initializer=paddle.nn.initializer.Normal())
        z.stop_gradient = False  # 设置 stop_gradient 为 False 以允许梯度计算

        # 生成目标标签
        targets = paddle.randint(low=0, high=self.num_classes, shape=[self.synthesis_batch_size], dtype='int64')
        
        reset_model(self.generator)  # 调用重置模型函数

        # 定义优化器
        optimizer = paddle.optimizer.Adam(parameters=[
            {'params': self.generator.parameters()},
            {'params': [z]}
        ], learning_rate=self.lr_g, beta1=0.5, beta2=0.999)

        for it in range(self.iterations):
            optimizer.clear_grad()  # 清零梯度
            inputs = self.generator(z)
            inputs = self.normalizer(inputs)
            t_out = self.teacher(inputs)

            if len(self.hooks) == 0 or self.bn == 0:
                loss_bn = paddle.to_tensor(0.0, dtype='float32')
            else:
                loss_bn = sum([h.r_feature for h in self.hooks]) / len(self.mdl_list)

            a = F.softmax(t_out, axis=1)  # 在 PaddlePaddle 中使用 axis 指定维度
            mask = paddle.zeros_like(a)
            b = paddle.unsqueeze(targets, axis=1)
            mask = paddle.scatter(mask, b, paddle.ones_like(b).astype('float32'))

            # 这里确保 mask 和 a 的形状是兼容的
            p = a[paddle.cast(mask, dtype='bool')]  # 使用 paddle.cast 将 mask 转换为 bool 类型

            # 计算损失
            loss_oh = ((1 - p.detach()).reshape([-1, 1]).pow(self.args.hs) * F.cross_entropy(t_out, targets, reduction='none')).mean()

            s_out = self.student(inputs)

            # 决策对抗蒸馏损失
            loss_adv = -(dense_kldiv(s_out, t_out, T=3, reduction='none').sum(1)).mean()

            loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv
            if it % self.args.print_freq == 0 or it == self.iterations - 1:
                self.args.logger.info('[GAN_Train] Iter={iter} L_BN={a_bn:.3f} * {l_bn:.3f}; L_oh={a_oh:.3f} * {l_oh:.3f};'
                                    ' L_adv={a_adv:.3f} * {l_adv:.3f}; LR={lr:.5f}'
                                    .format(iter=it, a_bn=self.bn, l_bn=float(loss_bn), a_oh=self.oh, l_oh=float(loss_oh),
                                            a_adv=self.adv, l_adv=float(loss_adv),
                                            lr=optimizer.get_lr()))

            loss.backward()
            paddle.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=10)
            for m in self.mdl_list:
                m.clear_gradients()  # 在 PaddlePaddle 中使用 clear_gradients() 来清空梯度
            optimizer.step()
            # scheduler.step()

            paddle.device.cuda.empty_cache()

            if best_cost > float(loss) or it ==0:
                best_cost = float(loss)
                best_inputs = inputs.detach()

        if self.args.weighted and cur_ep != 0:
            mix_weight = self.teacher.mdl_w_list.detach()
            ori_weight = self.teacher.mdl_w_list
            best_loss = 1e3
            for w_adjust in range(self.args.wa_steps):
                for idx, (images, labels) in enumerate(self.get_data(labeled=True)):
                    images = paddle.to_tensor(images, place=paddle.CUDAPlace(0))
                    labels = paddle.to_tensor(labels, place=paddle.CUDAPlace(0))
                    mix_weight.stop_gradient = False
                    tmp_model = WEnsemble(self.mdl_list, mix_weight)
                    tmp_model.to(paddle.CUDAPlace(0))

                    # forward pass
                    tmp_logits = tmp_model(images)
                    loss = F.cross_entropy(tmp_logits, labels)

                    # backward pass
                    loss.backward()
                    mix_weight = mix_weight - self.args.mu * pow(self.args.wdc, cur_ep) * paddle.sign(mix_weight.grad)
                    eta = paddle.clip(mix_weight - ori_weight, min=-1, max=1)
                    mix_weight = paddle.clip(ori_weight + eta, min=0.0, max=1.0).detach()

                    self.teacher.mdl_w_list = mix_weight
                del tmp_model

        self.student.train()
        if self.normalizer:
            best_inputs = self.normalizer(best_inputs, True)
        self.data_pool.add(best_inputs, batch_id=cur_ep, targets=targets, his=self.args.his)
        dst = self.data_pool.get_dataset(transform=self.transform, labeled=True)

        train_sampler = None

        loader = paddle.io.DataLoader(
            dst, batch_size=self.sample_batch_size, shuffle=(train_sampler is None),
            num_workers=4, return_list=True, batch_sampler=train_sampler)

        self.data_iter = DataIter(loader)
        del z, targets
        return {'synthetic': best_inputs}


    def sample(self):
        return None


    def get_data(self,labeled=True):
        datasets = self.data_pool.get_dataset(transform=self.transform, labeled=labeled)  # 获取程序运行到现在所有的图片
        self.data_loader = paddle.io.DataLoader(
            datasets, batch_size=self.sample_batch_size, shuffle=True,
            num_workers=4, use_shared_memory=True)
        return self.data_loader

def reset_model(model):
    for m in model.sublayers():
        if isinstance(m, (nn.Conv2DTranspose, nn.Linear, nn.Conv2D)):
            nn.initializer.Normal(mean=0.0, std=0.02)(m.weight)
            if m.bias is not None:
                nn.initializer.Constant(value=0.0)(m.bias)
        if isinstance(m, nn.BatchNorm2D):
            nn.initializer.Normal(mean=1.0, std=0.02)(m.weight)
            nn.initializer.Constant(value=0.0)(m.bias)