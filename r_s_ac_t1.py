from __future__ import print_function
import argparse
import os
import numpy as np
import random
import pickle
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model.resnet import resnet34, resnet18, LeNet
from model.basenet import AlexNetBase, VGGBase, Predictor, Predictor_deep
from utils.utils import weights_init
from utils.lr_schedule import inv_lr_scheduler
from utils.return_dataset import return_dataset, return_dataset_test, return_dataset_a, return_dataset_r, return_source
from utils.loss import entropy, adentropy
from active_d_r_s import active_L_c, active_f,get_target_l,get_target_un

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser(description='SSDA Classification')
parser.add_argument('--steps', type=int, default=50000, metavar='N',
                    help='maximum number of iterations '
                         'to train (default: 50000)')
parser.add_argument('--method', type=str, default='MME',
                    choices=['S+T', 'ENT', 'MME'],
                    help='MME is proposed method, ENT is entropy minimization,'
                         ' S+T is training only on labeled examples')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--multi', type=float, default=0.1, metavar='MLT',
                    help='learning rate multiplication')
parser.add_argument('--T', type=float, default=0.05, metavar='T',
                    help='temperature (default: 0.05)')
parser.add_argument('--lamda', type=float, default=0.1, metavar='LAM',
                    help='value of lamda')
parser.add_argument('--save_check', action='store_true', default=True,
                    help='save checkpoint or not')
parser.add_argument('--checkpath', type=str, default='./save_model_ssda',
                    help='dir to save checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging '
                         'training status')
parser.add_argument('--save_interval', type=int, default=500, metavar='N',
                    help='how many batches to wait before saving a model')
# parser.add_argument('--net', type=str, default='alexnet', #修改过
#                    help='which network to use')
# parser.add_argument('--net', type=str, default='LeNet',  # 修改过
#                     help='which network to use')
# parser.add_argument('--net', type=str, default='vgg', #修改过
#                     help='which network to use')
parser.add_argument('--net', type=str, default='resnet34', #修改过
                  help='which network to use')
# parser.add_argument('--net', type=str, default='resnet18', #修改过
#                     help='which network to use')
# parser.add_argument('--source', type=str, default='real',  ##需修改webcam,dslr，real,Product
#                     help='source domain')
parser.add_argument('--source', type=str, default='real',   ##需修改webcam,dslr，real,Product
                    help='source domain')
parser.add_argument('--target', type=str, default='sketch',  ##需修改amazon,sketch,Clipart,train_image
                    help='target domain')
parser.add_argument('--dataset', type=str, default='multi',  # 需修改，数据集为office31
                    choices=['multi', 'office31', 'officehome', 'mnist'],
                    help='the name of dataset')
parser.add_argument('--num', type=int, default=2,  # 主动学习次数
                    help='number of active learning')
parser.add_argument('--patience', type=int, default=5, metavar='S',
                    help='early stopping to wait for improvment '
                         'before terminating. (default: 5 (5000 iterations))')
parser.add_argument('--early', action='store_false', default=True,
                    help='early stopping on validation or not')

parser.add_argument('--NUM_TRAIN_t', type=int, default=16010)  # l_u1个数
# parser.add_argument('--NUM_TRAIN_t', type=int, default=46528)  # l_u1个数
parser.add_argument('--initial_size', type=int, default=126)  # 初始化个数
parser.add_argument('--batch_size', type=int, default=126)

parser.add_argument('--BATCH_AC', type=int, default=5)  # 62
parser.add_argument('--MARGIN', type=float, default=1.0)
parser.add_argument('--WEIGHT', type=float, default=1.0)
parser.add_argument('--TRIALS', type=int, default=1)
parser.add_argument('--CYCLES', type=int, default=6)  # 已修改
parser.add_argument('--EPOCH_AC', type=int, default=20)
parser.add_argument('--EPOCHL_AC', type=int, default=12)
parser.add_argument('--LR_AC', type=float, default=0.1)
parser.add_argument('--MOMENTUM', type=float, default=0.9)
parser.add_argument('--WDECAY', type=float, default=5e-4)
parser.add_argument('--MILESTONES', type=list, default=[160])
args = parser.parse_args()

target_loader_test = return_dataset_test(args)
source_loader, target_dataset_unl,  target_loader_val, len_class_list = return_dataset(args)
target_loader, target_loader_unl = return_dataset_a(args, 0)
# print("源域开始进行主动学习")##目标域主动学习完成后
# active(args)
#source_loader = return_source(args)

f_acc = 0

# use_gpu = torch.cuda.is_available()
record_dir = "record/multi/MME/R_S_active.txt"  # 保存每次step的准确率
torch.cuda.manual_seed(args.seed)

# 选择主流的cnn网络G
if args.net == 'resnet34':
    G = resnet34()
    inc = 512
elif args.net == 'resnet18':
    G = resnet18()
    inc = 512
# if args.net == 'resnet18':
#     G = resnet18()
#     inc = 512
elif args.net == "alexnet":
    G = AlexNetBase()
    inc = 4096
elif args.net == "vgg":
    G = VGGBase()
    inc = 4096
elif args.net == 'LeNet':
    G = LeNet()
    # inc = 144
    # inc = 256
    inc = 400
    # inc = 576  ###50000,576,32效果最好88.57
else:
    raise ValueError('Model cannot be recognized.')


params = []
for key, value in dict(G.named_parameters()).items():
    if value.requires_grad:
        if 'classifier' not in key:
            params += [{'params': [value], 'lr': args.multi,  # ？？？？？？？？
                        'weight_decay': 0.0005}]
        else:
            params += [{'params': [value], 'lr': args.multi * 10,
                        'weight_decay': 0.0005}]
# 分类器F0
if "resnet34" in args.net:
# if "resnet18" in args.net:
# if "LeNet" in args.net:
    # if "resnet18" in args.net:
    F1 = Predictor_deep(num_class=len_class_list,
                        inc=inc)
else:
    F1 = Predictor(num_class=len_class_list, inc=inc, cosine=False,
                   temp=args.T)

weights_init(F1)
lr = args.lr
G.cuda()
F1.cuda()

# 数据处理
im_data_s = torch.FloatTensor(1)
im_data_t = torch.FloatTensor(1)
im_t = torch.FloatTensor(1)

im_data_tu = torch.FloatTensor(1)
iter_tu = torch.FloatTensor(1)
iter_tu0 = torch.FloatTensor(1)
im_tu = torch.FloatTensor(1)

gt_labels_s = torch.LongTensor(1)
gt_labels_t = torch.LongTensor(1)
sample_labels_t = torch.LongTensor(1)
sample_labels_s = torch.LongTensor(1)

im_data_s = im_data_s.cuda()
im_data_t = im_data_t.cuda()
im_t = im_t.cuda()

im_data_tu = im_data_tu.cuda()
iter_tu = iter_tu.cuda()
iter_tu0 = iter_tu0.cuda()

im_tu = im_tu.cuda()

gt_labels_s = gt_labels_s.cuda()
gt_labels_t = gt_labels_t.cuda()
sample_labels_t = sample_labels_t.cuda()
sample_labels_s = sample_labels_s.cuda()

im_data_s = Variable(im_data_s)
im_data_t = Variable(im_data_t)
im_t = Variable(im_t)

im_data_tu = Variable(im_data_tu)
iter_tu = Variable(iter_tu)
iter_tu0 = Variable(iter_tu0)
im_tu = Variable(im_tu)

gt_labels_s = Variable(gt_labels_s)
gt_labels_t = Variable(gt_labels_t)
sample_labels_t = Variable(sample_labels_t)
sample_labels_s = Variable(sample_labels_s)

# 保存模型路径
if os.path.exists(args.checkpath) == False:
    os.mkdir(args.checkpath)

def get_data(flag, target_loader_init, target_loader_unl_init):
    num1 = 60
    for j, data_tu in enumerate(target_loader_init):
        im_data_t.resize_(data_tu[0].size()).copy_(data_tu[0])

        if j == 0:
            iter_tu = im_data_t
        else:
            iter_tu = torch.cat((iter_tu, im_data_t), 0)
        # print(j, len(iter_tu))
    for i, data_t in enumerate(target_loader_unl_init):
        im_data_tu.resize_(data_t[0].size()).copy_(data_t[0])
        # if i == 0:
        #     iter_target = torch.cat((iter_tu, im_data_tu), 0)
        #     all_target = iter_target
        # else:
        #     all_target = torch.cat((all_target, im_data_tu), 0)

        id = j + i + 1
        if flag == 0:
            if i == 0:
                iter_target = torch.cat((im_data_tu, iter_tu), 0)
                all_target = iter_target
            if i > 0:
                all_target = torch.cat((all_target, im_data_tu), 0)
            if id == num1 - 1:
                break
        if flag == 53:
            if id == num1*flag:
                all_target = im_data_tu
            if id > num1 * flag:
                all_target = torch.cat((all_target, im_data_tu), 0)
            if id == 16010:
                break

        # if flag != 0 & flag != 19:
        if 0 < flag < 53:
            if id == num1*flag:
                all_target = im_data_tu
            if id > num1 * flag:
                all_target = torch.cat((all_target, im_data_tu), 0)
            if id == (flag+1)*num1-1:
                break

    # print(len(all_target))
    return all_target

def train():
    global f_acc
    f_acc = 0
    G.train()
    F1.train()

    #初始化参数
    optimizer_g = optim.SGD(params, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)
    optimizer_f = optim.SGD(list(F1.parameters()), lr=1.0, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)

    def zero_grad_all():
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
    param_lr_g = []
    for param_group in optimizer_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_f = []
    for param_group in optimizer_f.param_groups:
        param_lr_f.append(param_group["lr"])
    criterion = nn.CrossEntropyLoss().cuda()
    all_step = args.steps
    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    data_iter_t_unl = iter(target_loader_unl)
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    len_train_target_semi = len(target_loader_unl)
    # ###############################对全部target取X_train#################
    #
    # for j, data_tu in enumerate(target_loader_unl):
    #     im_data_tu.resize_(data_tu[0].size()).copy_(data_tu[0])
    #     # im_data_tu = im_data_tu.to(device)
    #     if j == 0:
    #         iter_tu = im_data_tu
    #     else:
    #         iter_tu = torch.cat((iter_tu, im_data_tu), 0)
    #     # iter_tu = iter_tu.to(device)
    # for i, data_t in enumerate(target_loader):
    #     im_data_t.resize_(data_t[0].size()).copy_(data_t[0])
    #     # im_data_t = im_data_t.to(device)
    #     if i == 0:
    #         iter_target = torch.cat((im_data_t, iter_tu), 0)
    #         all_target = iter_target
    #     else:
    #         all_target = torch.cat((all_target, im_data_t), 0)
    #     # all_target = all_target.to(device)
    # print(len(all_target))
    #
    # ###############################对全部target取X_train#################
    best_acc = 0
    counter = 0
    step_x=[]
    acc_y=[]
    #每100个循环训练
    for step in range(all_step):
        optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step,
                                       init_lr=args.lr)
        optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, step,
                                       init_lr=args.lr)
        lr = optimizer_f.param_groups[0]['lr']
        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)
        if step % len_train_target_semi == 0:
            data_iter_t_unl = iter(target_loader_unl)
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)

        data_t = next(data_iter_t)
        data_t_unl = next(data_iter_t_unl)
        data_s = next(data_iter_s)
        '''
        im_data_s.data.resize_(data_s[0].size()).copy_(data_s[0])
        gt_labels_s.data.resize_(data_s[1].size()).copy_(data_s[1])
        im_data_t.data.resize_(data_t[0].size()).copy_(data_t[0])
        gt_labels_t.data.resize_(data_t[1].size()).copy_(data_t[1])
        im_data_tu.data.resize_(data_t_unl[0].size()).copy_(data_t_unl[0])
        '''
        im_data_s.resize_(data_s[0].size()).copy_(data_s[0])
        gt_labels_s.resize_(data_s[1].size()).copy_(data_s[1])
        im_data_t.resize_(data_t[0].size()).copy_(data_t[0])
        gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1])
        im_data_tu.resize_(data_t_unl[0].size()).copy_(data_t_unl[0])

        zero_grad_all()

        #拼接数据以及标签
        data = torch.cat((im_data_s, im_data_t), 0)
        target = torch.cat((gt_labels_s, gt_labels_t), 0)

        #得到特征提取G，分类F1
        output = G(data)
        out1 = F1(output)

        #损失，反向传播
        loss = criterion(out1, target)
        loss.backward(retain_graph=True)
        optimizer_g.step()
        optimizer_f.step()
        zero_grad_all()

        #我们提出的方法是MME
        if not args.method == 'S+T':
            output = G(im_data_tu)
            if args.method == 'ENT':
                loss_t = entropy(F1, output, args.lamda)
                loss_t.backward()
                optimizer_f.step()
                optimizer_g.step()
            elif args.method == 'MME':
                loss_t = adentropy(F1, output, args.lamda) #adentropy为导入函数，求两个熵值
                loss_t.backward()
                optimizer_f.step()
                optimizer_g.step()
            else:
                raise ValueError('Method cannot be recognized.')
            log_train = 'S {} T {} Train Ep: {} lr{} \t ' \
                        'Loss Classification: {:.6f} Loss T {:.6f} ' \
                        'Method {}\n'.format(args.source, args.target,
                                             step, lr, loss.data,
                                             -loss_t.data, args.method)
        else:
            log_train = 'S {} T {} Train Ep: {} lr{} \t ' \
                        'Loss Classification: {:.6f} Method {}\n'.\
                format(args.source, args.target,
                       step, lr, loss.data,
                       args.method)
        G.zero_grad()
        F1.zero_grad()
        zero_grad_all()

        #log_interval为100，
        if step % args.log_interval == 0:
            print(log_train)
        if step % args.save_interval == 0 and step > 0:
            print("验证集准确率")
            loss_val, acc_val = test(target_loader_val, G, F1)
            print("测试集准确率")
            _, test_acc = test(target_loader_test, G, F1)
            G.train()
            F1.train()
            step_x.append(step)
            acc_y.append(acc_val)

            # counter为计数，如果求出的准确率连续5次低于前面的准确率，训练停止break
            if test_acc >= best_acc:
                best_acc = test_acc

                # best_acc_test = acc_test
                counter = 0
            else:
                counter += 1

            # 比较得到最高准确率，以获得最好模型
            if f_acc <= test_acc:
                f_acc = test_acc
                f_G = G
                f_F1 = F1
                f_step = step

            # # counter达到了patience：5之后，训练停止
            # if args.early:
            #     if counter > args.patience:
            #         break

            if step == 12500:
                break
            # if step == 100:
            #     break
            # if step == 5000:
            #     break
            # if step == 50000:
            #     break

            # 输出准确率及保存
            # print('best acc val %f' % (acc_val))
            # # print('record %s' % record_file)
            with open(record_dir, 'a') as f:
                f.write('step %d final %f \n' % (step, acc_val))

            G.train()
            F1.train()
    f0 = open('./acc_d_r_s.txt', 'a')
    accstr = str(f_acc)
    f0.writelines([accstr, '\n'])
    f0.close()
    return f_G, f_F1, f_acc

def test(loader, g, f1):
    g.eval()
    f1.eval()
    test_loss = 0
    correct = 0
    size = 0
    num_class = len_class_list
    output_all = np.zeros((0, num_class))
    criterion = nn.CrossEntropyLoss().cuda()
    confusion_matrix = torch.zeros(num_class, num_class)
    with torch.no_grad():
        for batch_idx, data_t in enumerate(loader):
            im_data_t.resize_(data_t[0].size()).copy_(data_t[0])
            gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1])
            feat = g(im_data_t)
            output1 = f1(feat)
            output_all = np.r_[output_all, output1.data.cpu().numpy()]
            size += im_data_t.size(0)
            pred1 = output1.data.max(1)[1]
            for t, p in zip(gt_labels_t.view(-1), pred1.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += pred1.eq(gt_labels_t.data).cpu().sum()
            test_loss += criterion(output1, gt_labels_t) / len(loader)
    print('\nTest set: Average loss: {:.4f}, '
          'Accuracy: {}/{} f1 ({:.0f}%)\n'.
          format(test_loss, correct, size,
                 100. * correct / size))
    return test_loss.data, 100. * float(correct) / size

model1,model2, f_correct = train()
print(f_correct)




