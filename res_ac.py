'''Active Learning Procedure in PyTorch.

Reference:
[Yoo et al. 2019] Learning Loss for Active Learning (https://arxiv.org/abs/1905.03677)
'''

# Python
import os
import random

# Torch
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets.svhn import SVHN

# Torchvison
import torchvision.transforms as T
import torchvision.models as models

# Utils
from tqdm import tqdm

# Custom
import model.resnet_learning_loss as resnet
import model.lossnet as lossnet
from loaders.data_list import  Imagelists_VISDA
from utils.return_dataset import ResizeImage
from sample.sampler import SubsetSequentialSampler

# Data
train_transform = T.Compose([
    ResizeImage(256),
    T.RandomHorizontalFlip(),
    T.RandomCrop(224),#修改过
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# train_transform = T.Compose([
#     ResizeImage(32),
#
#     T.ToTensor()
# ])
test_transform = T.Compose([
    ResizeImage(256),
    T.CenterCrop(224),#修改过
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# test_transform = T.Compose([
#     ResizeImage(32),
#
#     T.ToTensor()
# ])

##
# Loss Prediction Loss
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape

    input = (input - input.flip(0))[
            :len(input) // 2]  # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1  # 1 operation which is defined by the authors

    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0)  # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()

    return loss


##
# Train Utils
iters = 0

def train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, args):
    models['backbone'].train()
    models['module'].train()
    global iters

    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        inputs = data[0].cuda()
        labels = data[1].cuda()
        iters += 1

        optimizers['backbone'].zero_grad()
        optimizers['module'].zero_grad()

        scores, features = models['backbone'](inputs)
        target_loss = criterion(scores, labels)

        if epoch > epoch_loss:
            # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model.
            features[0] = features[0].detach()
            features[1] = features[1].detach()
            features[2] = features[2].detach()
            features[3] = features[3].detach()
        pred_loss = models['module'](features)
        pred_loss = pred_loss.view(pred_loss.size(0))

        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        m_module_loss = LossPredLoss(pred_loss, target_loss, margin=args.MARGIN)
        loss = m_backbone_loss + args.WEIGHT * m_module_loss

        loss.backward()
        # 使用训练集训练这两个网络
        optimizers['backbone'].step()
        optimizers['module'].step()


def test(models, dataloaders, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()
    models['module'].eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:  # mode == 'test'时加载test_loader
            inputs = inputs.cuda()
            labels = labels.cuda()

            scores, _ = models['backbone'](inputs)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return 100 * correct / total


#
def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss, args):
    print('>> Train a Model.')
    best_acc = 0.
    checkpoint_dir = os.path.join('./cifar10', 'train', 'weights')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for epoch in range(num_epochs):
        schedulers['backbone'].step()
        schedulers['module'].step()

        train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, args)

        # Save a checkpoint
        if False and epoch % 5 == 4:
            acc = test(models, dataloaders, 'test')
            if best_acc < acc:
                best_acc = acc
            print('Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc))
    print('>> Finished.')


#
def get_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    models['module'].eval()
    uncertainty = torch.tensor([]).cuda()

    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()
            # labels = labels.cuda()

            scores, features = models['backbone'](inputs)
            pred_loss = models['module'](features)  # pred_loss = criterion(scores, labels) # ground truth loss
            pred_loss = pred_loss.view(pred_loss.size(0))

            uncertainty = torch.cat((uncertainty, pred_loss), 0)  # uncertainty和pred_loss按行拼接

    return uncertainty.cpu()

def return_classlist(image_list):
    with open(image_list) as f:
        label_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[1].strip()
            if label not in label_list:   #与之前不同的label值，存入label_list，有几个分类
                label_list.append(label)
    return label_list

def save(args,cycle,temp_lab):
    path = './data/%s/%s' % (args.dataset, args.source)
    s_name = 'r_labeled_set_cf%d.txt' %(cycle)
    image_train_s = \
        os.path.join(path,
                     'all' + '.txt')
    labeled_path = \
        os.path.join(path,s_name)
    # unlabeled_path = \
    #     os.path.join(path,
    #                  'r_unlabeled_set' + '.txt')
    label_f = open(labeled_path, 'w')
    # unlabel_f = open(unlabeled_path, 'w')
    with open(image_train_s) as f:
        for ind, x in enumerate(f.readlines()):  # ind为索引
            label = x.split(' ')[1].strip()  # strip移除字符串头尾指定的字符（默认为空格或换行符）
            image_path = x.split(' ')[0]
            if ind in temp_lab:
                label_f.writelines([image_path, ' ', label, '\n'])
            # else:
            #     unlabel_f.writelines([image_path, ' ', label, '\n'])
    label_f.close()
    # unlabel_f.close()

##
# Main
def active(args,flag):
    t_path = './data/%s/%s' % (args.dataset, args.target)
    s_path = './data/%s/%s' % (args.dataset, args.source)
    # s_path = './data/%s' % (args.source)
    # root = './data'
    root = './data/multi'
    image_train_t = \
        os.path.join(t_path,
                     'labeled_set_0' + '.txt')
    image_train_s = \
        os.path.join(s_path,
                     'all' + '.txt')
    # image_val_t = \
    #     os.path.join(t_path,
    #                  'new_val_t' + '.txt')
    image_val_t = \
        os.path.join(t_path,
                     'new_u2' + '.txt')
    target_train = Imagelists_VISDA(image_train_t, root, transform=train_transform)
    # target_unlabeled = Imagelists_VISDA(image_train_t, root, transform=test_transform)
    source_train = Imagelists_VISDA(image_train_s, root, transform=train_transform)
    # source_train = SVHN(root=s_path, split='train', transform=train_transform)
    source_unlabeled = Imagelists_VISDA(image_train_s, root, transform=test_transform)
    # source_unlabeled = SVHN(root=s_path, split='train', transform=test_transform)
    source_test = Imagelists_VISDA(image_val_t, root, transform=test_transform) ##无用测试集
    # source_test = SVHN(root=s_path, split='test', transform=test_transform)

    for trial in range(args.TRIALS):

        if flag == 1:
            f_lab_set = list(range(args.NUM_TRAIN_s))
            break
        f_acc = 0

        indices_t = list(range(378))  # NUM_TRAIN表示训练数据个数：50000
        f_lab_set = []
        # random.shuffle(labeled_set)
        # class_list = return_classlist(image_train_t)
        # print(class_list)
        # print("目标域标签类别数为：",len(class_list))

        indices = list(range(args.NUM_TRAIN_s))##源域训练集数量real:70358  svhn:?
        random.shuffle(indices)
        labeled_set = []
        # unlabeled_set = []
        # labeled_set = indices[:1000]
        unlabeled_set = indices#[1000:]
        # labeled_set = []
        # unlabeled_set = indices
        # with open(image_train_s) as f:
        #     label_list = []  # 第一次出现的标签加入到list中
        #     label_list2 = []  # 第二次出现的标签加入list2,确保每个31个类每个类2张
        #     label_list3=[]
        #     label_list4=[]
        #     label_list5=[]
        #     for ind, x in enumerate(f.readlines()):  # ind为索引
        #         label = x.split(' ')[1].strip()  # strip移除字符串头尾指定的字符（默认为空格或换行符）
        #         if label not in label_list:
        #             label_list.append(label)
        #             labeled_set.append(ind)
        #         elif label in label_list and label not in label_list2:
        #             label_list2.append(label)
        #             labeled_set.append(ind)
        #         elif label in label_list2 and label not in label_list3:
        #             label_list3.append(label)
        #             labeled_set.append(ind)
        #         elif label in label_list3 and label not in label_list4:
        #             label_list4.append(label)
        #             labeled_set.append(ind)
        #         elif label in label_list4 and label not in label_list5:
        #             label_list5.append(label)
        #             labeled_set.append(ind)
        #
        #
        # # 把剩下的索引加入到unlabeled_set中
        # for i in range(ind + 1):
        #     if i not in labeled_set:
        #         unlabeled_set.append(i)
        #
        # print(len(labeled_set))
        # print(len(unlabeled_set))
        #
        # class_list = return_classlist(image_train_t)
        # with open(image_train_s) as f:







#######################################################################################################

        ##得到最小个数

        # with open(image_train_s) as f:
        #     label_list = []
        #     for ind, x in enumerate(f.readlines()):
        #         label = x.split(' ')[1].strip()
        #         label_list.append(int(label))
        # print(len(label_list))
        # times = [0 for x in range(0, 126)]  # 长度为31，初始值为0的列表，记录每个类出现次数
        # for i in range(len(indices)):
        #
        #     times[label_list[indices[i]]] += 1  # 对应一个image_train_t中的索引
        #
        # print(times)
        # print(min(times))
#######################################################################################################






################################################################################################################################



        # train_loader=DataLoader(source_train, batch_size=args.BATCH_AC,
        #                           sampler=SubsetRandomSampler(labeled_set),
        train_loader = DataLoader(target_train, batch_size=args.BATCH_AC0,
                                  sampler=SubsetRandomSampler(indices_t),
        #################################################修改##########################################
                                  pin_memory=True)
        test_loader = DataLoader(source_test, batch_size=args.BATCH_AC0)
        #################################################修改##########################################
        dataloaders = {'train': train_loader, 'test': test_loader}

        # Model
        # resnet18 = resnet.resnet18(pretrained=False, num_classes=31).cuda()
        resnet18 = resnet.resnet18(pretrained=False, num_classes=126).cuda()
        loss_module = lossnet.LossNet().cuda()
        # 2个网络模型
        models = {'backbone': resnet18, 'module': loss_module}
        torch.backends.cudnn.benchmark = True

        # with open(image_train_s) as f:
        #     label_list = []
        #     for ind, x in enumerate(f.readlines()):
        #         label = x.split(' ')[1].strip()
        #         label_list.append(int(label))

        # times0 = [0 for x in range(0, 126)]
        # for i in range(len(unlabeled_set)):
        #     times0[label_list[unlabeled_set[i]]] += 1
        #
        # min_n = min(times0)
###################################################################################################
#         criterion = nn.CrossEntropyLoss(reduction='none')
#         optim_backbone = optim.SGD(models['backbone'].parameters(), lr=args.LR_AC,
#                                    momentum=args.MOMENTUM, weight_decay=args.WDECAY)
#         optim_module = optim.SGD(models['module'].parameters(), lr=args.LR_AC,
#                                  momentum=args.MOMENTUM, weight_decay=args.WDECAY)
#         sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=args.MILESTONES)
#         sched_module = lr_scheduler.MultiStepLR(optim_module, milestones=args.MILESTONES)
#
#         optimizers = {'backbone': optim_backbone, 'module': optim_module}
#         schedulers = {'backbone': sched_backbone, 'module': sched_module}
#
#         # Training and test
#         train(models, criterion, optimizers, schedulers, dataloaders, args.EPOCH_AC, args.EPOCHL_AC, args)
# #################################################################################################
#
#
#         for cycle in range(args.CYCLES0):
#
#             random.shuffle(unlabeled_set)
#             subset = unlabeled_set[:10000]
#             unlabeled_loader = DataLoader(source_unlabeled, batch_size=args.BATCH_AC,
#                                           sampler=SubsetSequentialSampler(subset),
#                                           # more convenient if we maintain the order of subset
#                                           pin_memory=True)
#
#             # Measure uncertainty of each data points in the subset
#             uncertainty = get_uncertainty(models, unlabeled_loader)
#
#             # Index in ascending order，返回loss从小到大的索引值
#             arg = np.argsort(uncertainty)
#             # 每个类要选2张，加入labeled_set，同时更新unlabeled_set
#             arg = list(arg.numpy())
#             # labeled_set_r += list(torch.tensor(subset)[arg][-1000:].numpy())
#             labeled_set_r += list(torch.tensor(subset)[arg][:500].numpy())
#             unlabeled_set = list(torch.tensor(subset)[arg][500:].numpy()) + unlabeled_set[10000:]



        # Active learning cycles
        '''
        for cycle in range(args.CYCLES1):
            # Loss, criterion and scheduler (re)initialization
            criterion = nn.CrossEntropyLoss(reduction='none')
            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=args.LR_AC,
                                       momentum=args.MOMENTUM, weight_decay=args.WDECAY)
            optim_module = optim.SGD(models['module'].parameters(), lr=args.LR_AC,
                                     momentum=args.MOMENTUM, weight_decay=args.WDECAY)
            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=args.MILESTONES)
            sched_module = lr_scheduler.MultiStepLR(optim_module, milestones=args.MILESTONES)

            optimizers = {'backbone': optim_backbone, 'module': optim_module}
            schedulers = {'backbone': sched_backbone, 'module': sched_module}

            # Training and test
            train(models, criterion, optimizers, schedulers, dataloaders, args.EPOCH_AC, args.EPOCHL_AC, args)
            acc = test(models, dataloaders, mode='test')
            # 将acc保存到acc.txt中
            f = open('./acc.txt', 'a')
            trialstr = str(trial + 1)
            cyclestr = str(cycle + 1)
            accstr = str(acc)
            f.writelines([trialstr, ',', cyclestr, ',', accstr, '\n'])
            f.close()
            print(
                'Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial + 1, args.TRIALS, cycle + 1,
                                                                                      args.CYCLES1, len(labeled_set),
                                                                                      acc))

            #  Update the labeled dataset via loss prediction-based uncertainty measurement

            # labeled_set_r=[]
            random.shuffle(unlabeled_set)
            subset=[]

            with open(image_train_s) as f:
                label_list = []
                for ind, x in enumerate(f.readlines()):
                    label = x.split(' ')[1].strip()
                    label_list.append(int(label))
            times = [0 for x in range(0, 126)]  # 长度为31，初始值为0的列表，记录每个类出现次数
            for i in range(len(unlabeled_set)):
                if times[label_list[unlabeled_set[i]]] < 100:
                    times[label_list[unlabeled_set[i]]] += 1  # 对应一个image_train_t中的索引
                    subset.append(unlabeled_set[i])
                    # unlabeled_set.remove(unlabeled_set[i])



            # subset = unlabeled_set[:10000]
            unlabeled_loader = DataLoader(source_unlabeled, batch_size=args.BATCH_AC,
                                          sampler=SubsetSequentialSampler(subset),
                                          # more convenient if we maintain the order of subset
                                          pin_memory=True)

            # Measure uncertainty of each data points in the subset
            uncertainty = get_uncertainty(models, unlabeled_loader)

            # Index in ascending order，返回loss从小到大的索引值
            arg = np.argsort(uncertainty)
            #每个类要选2张，加入labeled_set，同时更新unlabeled_set
            arg = list(arg.numpy())

            arg.reverse()
            # labeled_set_r += list(torch.tensor(subset)[arg][-1000:].numpy())
            # labeled_set_r += list(torch.tensor(subset)[arg][-500:].numpy())
            # unlabeled_set = list(torch.tensor(subset)[arg][:-500].numpy()) + unlabeled_set[10000:]
            # labeled_set_r += list(torch.tensor(subset)[arg][:].numpy())


            labeled_set_r1 = list(torch.tensor(subset)[arg][:].numpy())

            with open(image_train_s) as f:
                label_list = []
                for ind, x in enumerate(f.readlines()):
                    label = x.split(' ')[1].strip()
                    label_list.append(int(label))
            times = [0 for x in range(0, 126)]  # 长度为31，初始值为0的列表，记录每个类出现次数
            for i in range(len(labeled_set_r1)):
                if times[label_list[labeled_set_r1[i]]] < 10:
                    times[label_list[labeled_set_r1[i]]] += 1  # 对应一个image_train_t中的索引
                    labeled_set.append(labeled_set_r1[i])
                    unlabeled_set.remove(labeled_set_r1[i])

            # unlabeled_set = list(torch.tensor(subset)[arg][500:].numpy()) + unlabeled_set[10000:]
            # labeled_set +=arg[:40]
            # Create a new dataloader for the updated labeled dataset

            print("cycle:",cycle,len(labeled_set))

            dataloaders['train'] = DataLoader(target_train, batch_size=args.BATCH_AC,
                                              sampler=SubsetRandomSampler(labeled_set),
                                              pin_memory=True)




###########################################################################################################################
        

        '''
        for cycle in range(args.CYCLES0):
            # Loss, criterion and scheduler (re)initialization
            criterion = nn.CrossEntropyLoss(reduction='none')
            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=args.LR_AC,
                                       momentum=args.MOMENTUM, weight_decay=args.WDECAY)
            optim_module = optim.SGD(models['module'].parameters(), lr=args.LR_AC,
                                     momentum=args.MOMENTUM, weight_decay=args.WDECAY)
            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=args.MILESTONES)
            sched_module = lr_scheduler.MultiStepLR(optim_module, milestones=args.MILESTONES)

            optimizers = {'backbone': optim_backbone, 'module': optim_module}
            schedulers = {'backbone': sched_backbone, 'module': sched_module}

            # Training and test
            train(models, criterion, optimizers, schedulers, dataloaders, args.EPOCH_AC, args.EPOCHL_AC, args)
            acc = test(models, dataloaders, mode='test')

            if f_acc <= acc:
                f_acc = acc
                if cycle != 0:
                    f_lab_set = labeled_set
                if cycle == args.CYCLES0-1:
                    f_lab_set = indices
            # 将acc保存到acc.txt中
            f = open('./acc.txt', 'a')
            trialstr = str(trial + 1)
            cyclestr = str(cycle + 1)
            accstr = str(acc)
            f.writelines([trialstr, ',', cyclestr, ',', accstr, '\n'])
            f.close()
            if cycle == 0:
                print(
                    'Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial + 1, args.TRIALS, cycle
                                                                                          + 1, args.CYCLES0,
                                                                                          len(indices_t),
                                                                                          acc))
            else:
                print(
                    'Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial + 1, args.TRIALS,
                                                                                          cycle + 1,
                                                                                          args.CYCLES0,
                                                                                          len(labeled_set),
                                                                                          acc))
            # print(
            #     'Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial + 1, args.TRIALS, cycle
            #                                                                           + 1, args.CYCLES0,
            #                                                                           len(labeled_set),
            #                                                                           acc))


            # random.shuffle(unlabeled_set)
            subset = unlabeled_set#[:30000]##偶数

            unlabeled_loader = DataLoader(source_unlabeled, batch_size=args.BATCH_AC0,
                                          sampler=SubsetSequentialSampler(subset),
                                          # more convenient if we maintain the order of subset
                                          pin_memory=True)

            # Measure uncertainty of each data points in the subset
            uncertainty = get_uncertainty(models, unlabeled_loader)

            # Index in ascending order，返回loss从小到大的索引值
            arg = np.argsort(uncertainty)
            # 每个类要选2张，加入labeled_set，同时更新unlabeled_set
            arg = list(arg.numpy())

            # labeled_set_r = list(torch.tensor(subset)[arg][:].numpy())
            # source_unlabeled0 = Imagelists_VISDA(image_train_s, root, transform=test_transform)
            # train_loader = DataLoader(source_unlabeled0, batch_size=args.BATCH_AC1,
            #                           sampler=SubsetRandomSampler(labeled_set_r),
            #                           pin_memory=True)
            #
            # times = [0 for x in range(0, 126)]
            # labeled_set_iter = []
            # for batch_idx, (data, target) in enumerate(train_loader):
            #     if batch_idx == 0:
            #         target = target.numpy()
            #         print(target)
            #     if times[target[0]] < 100:
            #         times[target[0]] += 1
            #         labeled_set_iter.append(labeled_set_r[batch_idx])
            #         labeled_set_r.remove(labeled_set_r[batch_idx])
            #
            # labeled_set = labeled_set + labeled_set_iter
            # if len(labeled_set) != (cycle + 1) * 1260:
            #     labeled_set = labeled_set + labeled_set_r[:(1260 - len(labeled_set_iter))]
            #     labeled_set_r = labeled_set_r[(1260 - len(labeled_set_iter)):]
            #
            # unlabeled_set = labeled_set_r


            labeled_set_r = list(torch.tensor(subset)[arg][:].numpy())
            labeled_set_iter = []
            labeled_set_r_copy = []
            #
            with open(image_train_s) as f:
                label_list = []
                for ind, x in enumerate(f.readlines()):
                    label = x.split(' ')[1].strip()
                    label_list.append(int(label))
            times = [0 for x in range(0, 126)]  # 长度为31，初始值为0的列表，记录每个类出现次数
            for i in range(len(labeled_set_r)):
                if times[label_list[labeled_set_r[i]]] < 50:
                    times[label_list[labeled_set_r[i]]] += 1  # 对应一个image_train_t中的索引
                    labeled_set_iter.append(labeled_set_r[i])
                    # labeled_set_r_copy.remove(labeled_set_r[i])

            for ind in range(len(labeled_set_r)):
                if labeled_set_r[ind] not in labeled_set_iter:
                    labeled_set_r_copy.append(labeled_set_r[ind])
            labeled_set = labeled_set + labeled_set_iter
            print(len(labeled_set))
            print(len(labeled_set_r_copy))
            if len(labeled_set) != (cycle + 1)*6300:
                labeled_set = labeled_set + labeled_set_r_copy[:(6300 - len(labeled_set_iter))]
                labeled_set_r_copy = labeled_set_r_copy[(6300 - len(labeled_set_iter)):]
            unlabeled_set = labeled_set_r_copy
            # save(args,cycle,labeled_set)
            print(len(labeled_set))
            print(len(unlabeled_set))


            # labeled_set += list(torch.tensor(subset)[arg][:1000].numpy())
            # unlabeled_set = list(torch.tensor(subset)[arg][1000:].numpy()) + unlabeled_set[10000:]
            # labeled_set += list(torch.tensor(subset)[arg][-1000:].numpy())
            # unlabeled_set = list(torch.tensor(subset)[arg][:-1000].numpy()) + unlabeled_set[30000:]

            dataloaders['train'] = DataLoader(source_train, batch_size=args.BATCH_AC0,
                                              sampler=SubsetRandomSampler(labeled_set),
                                              pin_memory=True)



         path = './data/%s/%s' % (args.dataset, args.source)
         labeled_path = \
             os.path.join(path,
                          'r_labeled_set' + '.txt')
         unlabeled_path = \
             os.path.join(path,
                          'r_unlabeled_set' + '.txt')
         label_f = open(labeled_path, 'w')
         unlabel_f = open(unlabeled_path, 'w')
         with open(image_train_s) as f:
             for ind, x in enumerate(f.readlines()):  # ind为索引
                 label = x.split(' ')[1].strip()  # strip移除字符串头尾指定的字符（默认为空格或换行符）
                 image_path = x.split(' ')[0]
                 if ind in labeled_set:
                     label_f.writelines([image_path, ' ', label, '\n'])
                 else:
                     unlabel_f.writelines([image_path, ' ', label, '\n'])
         label_f.close()
         unlabel_f.close()
    '''
    print("最后选取的源域数量为")
    num = len(f_lab_set)
    print(num)
    if num == 0:
        f_lab_set = indices
    random.shuffle(f_lab_set)
    path = './data/%s/%s' % (args.dataset, args.source)
    labeled_path = \
        os.path.join(path,
                     'r_labeled_set' + '.txt')
    unlabeled_path = \
        os.path.join(path,
                     'r_unlabeled_set' + '.txt')
    label_f = open(labeled_path, 'w')
    unlabel_f = open(unlabeled_path, 'w')
    with open(image_train_s) as f:
        for ind, x in enumerate(f.readlines()):  # ind为索引
            label = x.split(' ')[1].strip()  # strip移除字符串头尾指定的字符（默认为空格或换行符）
            image_path = x.split(' ')[0]
            if ind in f_lab_set:
                label_f.writelines([image_path, ' ', label, '\n'])
            else:
                unlabel_f.writelines([image_path, ' ', label, '\n'])
    label_f.close()
    unlabel_f.close()
    '''




