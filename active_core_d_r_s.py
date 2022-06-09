# import gc
# import numpy as np
# from scipy.spatial import distance_matrix
# import torch
#
# def get_unlabeled_idx(labeled_idx, flag):
#     """
#     Given the training set and the indices of the labeled examples, return the indices of the unlabeled examples.
#     """
#     if flag == 0: ##mnist
#         num = 59880
#     if flag == 1: ##office31
#         num = 1785
#     if flag == 2: ##multi
#         num == 12284
#     return np.arange(num)[np.logical_not(np.in1d(np.arange(num), labeled_idx))]
#
# class QueryMethod:
#     """
#     A general class for query strategies, with a general method for querying examples to be labeled.
#     """
#
#     def __init__(self, model, input_shape=(28,28), num_labels=10):
#         self.model = model
#         self.input_shape = input_shape
#         self.num_labels = num_labels
#
#     def query(self, labeled_idx, amount,  flag):
#         """
#         get the indices of labeled examples after the given amount have been queried by the query strategy.
#         :param labeled_idx: the indices of the labeled examples
#         :param amount: the amount of examples to query
#         :return: the new labeled indices (including the ones queried)
#         """
#         return NotImplemented
#
#     def query_f(self, representation, amount, labeled_idx):
#         return NotImplemented
#
#     def update_model(self, new_model):
#         del self.model
#         gc.collect()
#         self.model = new_model
#
# class CoreSetSampling(QueryMethod):
#     """
#     An implementation of the greedy core set query strategy.
#     """
#
#     def __init__(self, model, input_shape, num_labels):
#         super().__init__(model, input_shape, num_labels)
#
#     def greedy_k_center(self, labeled, unlabeled, amount):
#
#         greedy_indices = []
#
#         # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
#         min_dist = np.min(distance_matrix(labeled[0, :].reshape((1, labeled.shape[1])), unlabeled), axis=0)
#         min_dist = min_dist.reshape((1, min_dist.shape[0]))
#         for j in range(1, labeled.shape[0], 100):
#             if j + 100 < labeled.shape[0]:
#                 dist = distance_matrix(labeled[j:j+100, :], unlabeled)
#             else:
#                 dist = distance_matrix(labeled[j:, :], unlabeled)
#             min_dist = np.vstack((min_dist, np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
#             min_dist = np.min(min_dist, axis=0)
#             min_dist = min_dist.reshape((1, min_dist.shape[0]))
#
#         # iteratively insert the farthest index and recalculate the minimum distances:
#         farthest = np.argmax(min_dist)
#         greedy_indices.append(farthest)
#         for i in range(amount-1):
#             dist = distance_matrix(unlabeled[greedy_indices[-1], :].reshape((1,unlabeled.shape[1])), unlabeled)
#             min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
#             min_dist = np.min(min_dist, axis=0)
#             min_dist = min_dist.reshape((1, min_dist.shape[0]))
#             farthest = np.argmax(min_dist)
#             greedy_indices.append(farthest)
#
#         return np.array(greedy_indices)
#
#     #初试的query函数
#
#     def re_p(self, X_train):
#
#         representation_model = self.model
#         representation = representation_model(X_train)
#
#         return  representation
#
#     def query(self, X_train):
#
#         # unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
#         representation = self.re_p(X_train)
#         return representation
#
#
#         # with torch.no_grad():
#         #     representation = representation.cpu()
#         #     new_indices = self.greedy_k_center(representation[labeled_idx, :], representation[unlabeled_idx, :], amount)
#         # return np.hstack((labeled_idx, unlabeled_idx[new_indices])), representation
#
#     def query_f(self, representation, amount, labeled_idx, flag):
#         unlabeled_idx = get_unlabeled_idx(labeled_idx,flag)
#         with torch.no_grad():
#             representation = representation.cpu()
#             new_indices = self.greedy_k_center(representation[labeled_idx, :], representation[unlabeled_idx, :],
#                                                amount)
#         return np.hstack((labeled_idx, unlabeled_idx[new_indices]))

import gc
import numpy as np
from scipy.spatial import distance_matrix
import torch


def get_unlabeled_idx(X_train, labeled_idx):
    """
    Given the training set and the indices of the labeled examples, return the indices of the unlabeled examples.
    """
    return np.arange(X_train.shape[0])[np.logical_not(np.in1d(np.arange(X_train.shape[0]), labeled_idx))]


def get_rand_unlabeled_idx(args, labeled_idx):
    """
    Given the training set and the indices of the labeled examples, return the indices of the unlabeled examples.
    """
    # return np.arange(args.NUM_TRAIN_t)[np.logical_not(np.in1d(np.arange(args.NUM_TRAIN_t), labeled_idx))]
    if args.dataset == 'office31':
        count = 1785
    if args.dataset == 'mnist':
        count = 59880
    return np.arange(count)[np.logical_not(np.in1d(np.arange(count), labeled_idx))]


class QueryMethod:
    """
    A general class for query strategies, with a general method for querying examples to be labeled.
    """

    def __init__(self, model, input_shape=(28, 28), num_labels=10):
        self.model = model
        self.input_shape = input_shape
        self.num_labels = num_labels

    def query(self, X_train, labeled_idx, amount, args):
        """
        get the indices of labeled examples after the given amount have been queried by the query strategy.
        :param X_train: the training set
        :param Y_train: the training labels
        :param labeled_idx: the indices of the labeled examples
        :param amount: the amount of examples to query
        :return: the new labeled indices (including the ones queried)
        """
        return NotImplemented

    def query_f(self, representation, labeled_idx, amount):
        return NotImplemented

    def update_model(self, new_model):
        del self.model
        gc.collect()
        self.model = new_model


class RandomSampling(QueryMethod):
    """
    A random sampling query strategy baseline.
    """

    def __init__(self, model, input_shape, num_labels):
        super().__init__(model, input_shape, num_labels)

    def query(self, X_train, labeled_idx, amount, args):
        unlabeled_idx = get_rand_unlabeled_idx(args, labeled_idx)
        return np.hstack((labeled_idx, np.random.choice(unlabeled_idx, amount, replace=False)))


# class UncertaintySampling(QueryMethod):
#     """
#     The basic uncertainty sampling query strategy, querying the examples with the minimal top confidence.
#     """
#
#     def __init__(self, model, input_shape, num_labels):
#         super().__init__(model, input_shape, num_labels)
#
#     def query(self, X_train, labeled_idx, amount, args):
#
#         unlabeled_idx = get_unlabeled_idx(args, labeled_idx)
#         # predictions = self.model.predict(X_train[unlabeled_idx, :])
#         # predictions = self.model.eval(X_train[unlabeled_idx, :])
#         # X_train = torch.tensor(X_train)
#         predictions = self.model(X_train[unlabeled_idx, :])
#         unlabeled_predictions = np.amax(predictions, axis=1)
#
#         selected_indices = np.argpartition(unlabeled_predictions, amount)[:amount]
#         return np.hstack((labeled_idx, unlabeled_idx[selected_indices]))

class CoreSetSampling(QueryMethod):
    """
    An implementation of the greedy core set query strategy.
    """

    def __init__(self, model, input_shape, num_labels):
        super().__init__(model, input_shape, num_labels)

    def greedy_k_center(self, labeled, unlabeled, amount):

        greedy_indices = []

        # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
        min_dist = np.min(distance_matrix(labeled[0, :].reshape((1, labeled.shape[1])), unlabeled), axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        for j in range(1, labeled.shape[0], 100):
            if j + 100 < labeled.shape[0]:
                dist = distance_matrix(labeled[j:j + 100, :], unlabeled)
            else:
                dist = distance_matrix(labeled[j:, :], unlabeled)
            min_dist = np.vstack((min_dist, np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))

        # iteratively insert the farthest index and recalculate the minimum distances:
        farthest = np.argmax(min_dist)
        greedy_indices.append(farthest)
        for i in range(amount - 1):
            dist = distance_matrix(unlabeled[greedy_indices[-1], :].reshape((1, unlabeled.shape[1])), unlabeled)
            min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            farthest = np.argmax(min_dist)
            greedy_indices.append(farthest)

        return np.array(greedy_indices)

    def query(self, X_train, labeled_idx, amount, args):

        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
        # X_train = torch.tensor(X_train)
        # use the learned representation for the k-greedy-center algorithm:
        representation_model = self.model
        # representation_model = Model(inputs=self.model.input, outputs=self.model.get_layer('softmax').input)
        representation = representation_model(X_train)
        # representation = representation_model.eval(X_train)
        with torch.no_grad():
            representation = representation.cpu()
            new_indices = self.greedy_k_center(representation[labeled_idx, :], representation[unlabeled_idx, :], amount)
        return np.hstack((labeled_idx, unlabeled_idx[new_indices])), representation

    # def query_f(self, representation, labeled_idx, amount):
    #
    #     unlabeled_idx = np.arange(1785)[np.logical_not(np.in1d(np.arange(1785), labeled_idx))]
    #     # # X_train = torch.tensor(X_train)
    #     # # use the learned representation for the k-greedy-center algorithm:
    #     # representation_model= self.model
    #     # # representation_model = Model(inputs=self.model.input, outputs=self.model.get_layer('softmax').input)
    #     # representation =representation_model(X_train)
    #     # # representation = representation_model.eval(X_train)
    #     with torch.no_grad():
    #         representation = representation.cpu()
    #         new_indices = self.greedy_k_center(representation[labeled_idx, :], representation[unlabeled_idx, :], amount)
    #     return np.hstack((labeled_idx, unlabeled_idx[new_indices]))

    def query_f(self, representation, labeled_idx, amount):

        unlabeled_idx = np.arange(16010)[np.logical_not(np.in1d(np.arange(16010), labeled_idx))]
        # unlabeled_idx = np.arange(46528)[np.logical_not(np.in1d(np.arange(46528), labeled_idx))]
        # # X_train = torch.tensor(X_train)
        # # use the learned representation for the k-greedy-center algorithm:
        # representation_model= self.model
        # # representation_model = Model(inputs=self.model.input, outputs=self.model.get_layer('softmax').input)
        # representation =representation_model(X_train)
        # # representation = representation_model.eval(X_train)
        with torch.no_grad():
            representation = representation.cpu()
            new_indices = self.greedy_k_center(representation[labeled_idx, :], representation[unlabeled_idx, :], amount)
        return np.hstack((labeled_idx, unlabeled_idx[new_indices]))


