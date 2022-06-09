import numpy as np
import os
import os.path
from PIL import Image


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def make_dataset_fromlist(image_list):
    with open(image_list) as f:
        image_index = [x.split(' ')[0] for x in f.readlines()] #存放的是图片
    with open(image_list) as f:
        label_list = []
        selected_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[1].strip() 
            label_list.append(int(label))   #存放的是图片标签
            selected_list.append(ind)     #存的是一系列数字0,1...len()
        image_index = np.array(image_index)
        label_list = np.array(label_list)    #存放的是图片标签
    image_index = image_index[selected_list]   #c存放的图片对应的序号
    return image_index, label_list  #返回的是[0,0]等


def return_classlist(image_list):
    with open(image_list) as f:
        label_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[0].split('/')[-2]
            if label not in label_list:   #与之前不同的label值，存入label_list，有几个分类
                label_list.append(str(label))
    return label_list


class Imagelists_VISDA(object):
    def __init__(self, image_list, root="./data/multi/",
                 transform=None, target_transform=None, test=False):
        imgs, labels = make_dataset_fromlist(image_list) #对应分别是序号，标签
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root
        self.test = test

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.test:
            return img, target    #test为false,返回img, target 
        else:
            return img, target, self.imgs[index] #test为true,返回img, target 和？

    def __len__(self):
        return len(self.imgs)
