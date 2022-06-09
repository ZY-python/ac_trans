import numpy as np
from PIL import Image
from scipy.spatial import distance_matrix
import os
import pickle
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from active_core_d_r_s import *

def get_unlabeled_idx(args, labeled_idx):
    """
    Given the training set and the indices of the labeled examples, return the indices of the unlabeled examples.
    """
    # return np.arange(args.NUM_TRAIN_t)[np.logical_not(np.in1d(np.arange(args.NUM_TRAIN_t), labeled_idx))]
    if args.dataset == 'office31':
        count = 1785
    if args.dataset == 'mnist':
        count = 59880
    # if args.dataset == 'multi':
    #     count = 46528
    if args.dataset == 'multi':
        count = 16010
    if args.dataset == 'officehome':
        count = 2715
    return np.arange(count)[np.logical_not(np.in1d(np.arange(count), labeled_idx))]

def get_rand_unlabeled_idx(args, labeled_idx, num):
    """
    Given the training set and the indices of the labeled examples, return the indices of the unlabeled examples.
    """
    # return np.arange(args.NUM_TRAIN_t)[np.logical_not(np.in1d(np.arange(args.NUM_TRAIN_t), labeled_idx))]
    # if args.dataset == 'office31':
    #     count = 1785
    # if args.dataset == 'mnist':
    #     count = 59880
    return np.arange(num)[np.logical_not(np.in1d(np.arange(num), labeled_idx))]

##初始得到的target_un
def get_target_un(target_dataset_unl, args, labeled_idx):
    all_list = list(range(args.NUM_TRAIN_t))
    if len(labeled_idx) == 0:
        target_loader_unl = DataLoader(target_dataset_unl, batch_size=args.BATCH_AC,
                                       sampler=SubsetRandomSampler(all_list),
                                       pin_memory=True)
    else:
        unlabeled_idx = get_unlabeled_idx(args, labeled_idx)
        # for i in range(len(labeled_idx)):
        #     if labeled_idx[i] in all_list:
        #         unlabeled_idx.remove(labeled_idx[i])
        print("unlabeled_idx为%d"%(len(unlabeled_idx)))
        target_loader_unl = DataLoader(target_dataset_unl, batch_size=args.BATCH_AC,
                                       sampler=SubsetRandomSampler(unlabeled_idx),
                                       pin_memory=True)
    return target_loader_unl

##初始得到的target_l
def get_target_l(target_dataset_unl, args, labeled_idx):

    target_loader = DataLoader(target_dataset_unl, batch_size=args.BATCH_AC,
                               sampler=SubsetRandomSampler(labeled_idx),
                               pin_memory=True)
    return target_loader

def active_L_c(args, len_class_list, model1, labeled_idx,  X_train,num):
    input_shape = (3, 32, 32)
    method = CoreSetSampling
    query_method = method(None, input_shape, len_class_list)
    query_method.update_model(model1)
    labeled_idx,rep = query_method.query(X_train, labeled_idx, num, args)
    return labeled_idx,rep

def active_rand(args, len_class_list, model1, labeled_idx,  X_train, num):
    input_shape = (3, 32, 32)
    method = RandomSampling
    query_method = method(None, input_shape, len_class_list)
    query_method.update_model(model1)
    labeled_idx, rep = query_method.query(X_train, labeled_idx, num, args)
    return labeled_idx, rep

def active_f(representation, labeled_idx, num, len_class_list):
    input_shape = (3, 32, 32)
    method = CoreSetSampling
    query_method = method(None, input_shape, len_class_list)
    labeled_idx= query_method.query_f(representation, labeled_idx, num)
    return labeled_idx

def init_rand(args):

    labeled_idx = np.random.choice(args.NUM_TRAIN_t,  args.initial_size, replace=False)

    return labeled_idx
