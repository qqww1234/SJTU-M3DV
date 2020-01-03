#import
import keras
import os
import os.path
from matplotlib import numpy as np
import pandas as pd
os.environ['KERAS_BACKEND']='tensorflow'
#数据地址和大小
x_train_path='./train_val'
x_test_path='./test'
size=32


#预处理，选取32×32数据
def get_rec(sign,n):
    sign_min = sign[n, :].min()
    sign_max = sign[n, :].max()
    len = sign_max - sign_min + 1
    if len - size > 0:
        sign_min += (len - size) // 2
        sign_max -= (len - size) - [(len - size) // 2]
        len = size
    low = (size // 2) - (len // 2)
    return sign_min,low,len



#读取训练集数据
def get_dataset():
    x_return_train = np.zeros((465, size, size, size))
    x_name_train=pd.read_csv("train_val.csv") ['name']
    filenum = 0
    for i in range(465):
        x_file_temp = os.path.join(x_train_path, x_name_train[i] + '.npz')
        x_voxel = np.array(np.load(x_file_temp)['voxel'])
        x_seg = np.array(np.load(x_file_temp)['seg'])
        x_temp = x_voxel * x_seg * 0.8 + x_voxel * 0.2
        s = x_seg * x_voxel
        sign = np.array(np.nonzero(s))
        min1, low1, len1 = get_rec(sign, 0)
        min2, low2, len2 = get_rec(sign, 1)
        min3, low3, len3 = get_rec(sign, 2)
        i_temp = 0
        for i in range(low1, low1 + len1):
            j_temp = 0
            for j in range(low2, low2 + len2):
                k_temp = 0
                for k in range(low3, low3 + len3):
                    x_return_train[filenum, i, j, k] = x_temp[min1 + i_temp, min2 + j_temp, min3 + k_temp]
                    k_temp += 1
                j_temp += 1
            i_temp += 1
        filenum += 1
        return x_return_train


def get_label():
    x_label=pd.read_csv("train_val.csv") ['lable']
    x_train_label=keras.utils.to_categorical(x_label,2)[0:465]

    return  x_train_label

def get_test_dataset():
    x_return_test = np.zeros((117, size, size, size))
    x_name_test = pd.read_csv("test.csv")['Id']
    filenum = 0
    for i in range(117):
        x_file_temp = os.path.join(x_test_path, x_name_test[i] + '.npz')
        x_voxel = np.array(np.load(x_file_temp)['voxel'])
        x_seg = np.array(np.load(x_file_temp)['seg'])
        x_temp = x_voxel * x_seg * 0.8 + x_voxel * 0.2
        s = x_seg * x_voxel
        sign = np.array(np.nonzero(s))
        min1, low1, len1 = get_rec(sign, 0)
        min2, low2, len2 = get_rec(sign, 1)
        min3, low3, len3 = get_rec(sign, 2)
        i_temp = 0
        for i in range(low1, low1+len1):
            j_temp = 0
            for j in range(low2, low2+len2):
                k_temp = 0
                for k in range(low3, low3+len3):
                    x_return_test[filenum, i, j, k] = x_temp[min1 + i_temp, min2 + j_temp, min3 + k_temp]
                    k_temp += 1
                j_temp += 1
            i_temp += 1
        filenum += 1
    return  x_return_test


#数据增强
def get_mixup(x_candidate, x_candidate_label, alpha):
    if alpha == 0:
        return x_candidate, x_candidate_label
    if alpha > 0:
        x_mixup = np.zeros(np.shape(x_candidate))
        y_mixup = np.zeros(np.shape(x_candidate_label), 'float')
        length = len(x_candidate)
        lam = np.random.beta(alpha, alpha, length)
        indexs = np.random.randint(0, length, length)
        for i in range(length):
            x_mixup[i] = x_candidate[i]*lam[i]+(1-lam[i])*x_candidate[indexs[i]]
            y_mixup[i] = x_candidate_label[i]*lam[i]+(1-lam[i])*x_candidate_label[indexs[i]]
        return x_mixup, y_mixup

#数据匹配
def data_reshape(dataset):
    dataset_re=np.array(dataset)
    dataset_re = dataset_re.reshape(dataset_re.shape[0], 32, 32, 32, 1)
    dataset_re=dataset_re.astype('float32')/255
    return dataset_re



