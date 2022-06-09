import numpy as np
from PIL import Image
import os
import pickle
import random

labeled_idx0 = random.sample(range(0, 3202), 25)
labeled_idx1 = random.sample(range(3202, 6404), 25)
labeled_idx2 = random.sample(range(6404, 9606), 25)
labeled_idx3 = random.sample(range(9606, 12808), 25)
labeled_idx4 = random.sample(range(12808, 16010), 26)
# labeled_idx0 = random.sample(range(0, 3202), 50)
# labeled_idx1 = random.sample(range(3202, 6404), 50)
# labeled_idx2 = random.sample(range(6404, 9606), 50)
# labeled_idx3 = random.sample(range(9606, 12808), 50)
# labeled_idx4 = random.sample(range(12808, 16010), 52)
# labeled_idx0 = random.sample(range(0, 6498), 100)
# labeled_idx1 = random.sample(range(6498, 12966), 100)
# labeled_idx2 = random.sample(range(12966, 19494), 100)
# labeled_idx3 = random.sample(range(19494, 25992), 100)
# labeled_idx4 = random.sample(range(25992, 32490), 100)

#初始化的id
labeled_idx = labeled_idx0 + labeled_idx1 + labeled_idx2 + labeled_idx3 + labeled_idx4

# labeled_idx = random.sample(range(0, 16010), 2)

result_path = os.path.join('./initial_d_r_s.pkl')
with open(result_path, 'wb') as f0:
    pickle.dump(labeled_idx, f0)
dirname = './data/multi/sketch'
fpath = os.path.join(dirname, 'l_u1.txt')
with open(fpath) as f:
    labeled_path =  os.path.join(dirname, 'labeled_set_0' + '.txt')
    unlabeled_path =os.path.join(dirname,'unlabeled_set_0' + '.txt')
    label_f = open(labeled_path, 'w')
    unlabel_f = open(unlabeled_path, 'w')
    for ind, x in enumerate(f.readlines()):
        label = x.split(' ')[1].strip()  # strip移除字符串头尾指定的字符（默认为空格或换行符）
        image_path = x.split(' ')[0]
        if ind in labeled_idx:
            label_f.writelines([image_path, ' ', label, '\n'])
        else:
            unlabel_f.writelines([image_path, ' ', label, '\n'])
label_f.close()
unlabel_f.close()
