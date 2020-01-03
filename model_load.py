import os
import os.path
os.environ['KERAS_BACKEND']='tensorflow'
from keras.models import load_model
from matplotlib import numpy as np
import pandas as pd
x_test_path='./test'
size=32

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


def data_reshape(dataset):
    dataset_re=np.array(dataset)
    dataset_re = dataset_re.reshape(dataset_re.shape[0], 32, 32, 32, 1)
    dataset_re=dataset_re.astype('float32')/255
    return dataset_re

def pred(model_path,path,x_predict):
    model=load_model(model_path)
    y = model.predict(x_predict)
    y_pred = pd.DataFrame(y)
    y_pred.to_csv(path)
    y_predicted = pd.read_csv(path)['1']
    return y_predicted

x_predict = get_test_dataset()
x_predict = data_reshape(x_predict)
y1_predicted=pred('model1.h5','last.csv',x_predict)
y2_predicted=pred('model2.h5','last.csv',x_predict)

data = pd.read_csv('Submission.csv')
data['Predicted']=y1_predicted*0.7+y2_predicted*0.3
data.to_csv('Submission.csv',index=False)