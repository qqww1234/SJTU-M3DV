#import
import keras
import os.path
from matplotlib import numpy as np
from .DenseNet import createDenseNet
from keras.optimizers  import SGD
from keras.losses import categorical_crossentropy
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from .getdata import get_dataset,get_test_dataset,get_label,get_mixup,data_reshape
os.environ['KERAS_BACKEND']='tensorflow'


#训练深度、batch_size定义
densenet_depth = 7
densenet_growth_rate = 12
batch_size = 256
size = 32

#数据读取
x_train_set = get_dataset()
x_train_label = get_label()
x_predict = get_test_dataset()
x_train_set = data_reshape(x_train_set)
x_predict = data_reshape(x_predict)

#数据增强
x_train_mixup1, x_train_label_mixup1=get_mixup(x_train_set, x_train_label, 0.1)
x_train_mixup2, x_train_label_mixup2=get_mixup(x_train_set, x_train_label, 0.1)
x_train_mixup3, x_train_label_mixup3=get_mixup(x_train_set, x_train_label, 0.1)
x_train_mixup4, x_train_label_mixup4=get_mixup(x_train_set, x_train_label, 0.1)
x_train_mixup5, x_train_label_mixup5=get_mixup(x_train_set, x_train_label, 0.1)
x_train_mixup6, x_train_label_mixup6=get_mixup(x_train_set, x_train_label, 0.1)
x_train_mixup7, x_train_label_mixup7=get_mixup(x_train_set, x_train_label, 0.1)
x_train_mixup8, x_train_label_mixup8=get_mixup(x_train_set, x_train_label, 0.1)
x_train_mixup9, x_train_label_mixup9=get_mixup(x_train_set, x_train_label, 0.1)

x_train_set=np.r_[x_train_set,x_train_mixup1,x_train_mixup2,x_train_mixup3,x_train_mixup4,x_train_mixup5,x_train_mixup6,x_train_mixup7,x_train_mixup8,x_train_mixup9]
x_train_label=np.r_[x_train_label,x_train_label_mixup1,x_train_label_mixup2,x_train_label_mixup3,x_train_label_mixup4,x_train_label_mixup5,x_train_label_mixup6,x_train_label_mixup7,x_train_label_mixup8,x_train_label_mixup9]


#生成训练集测试集
x_train_train, x_train_test, y_train_train, y_train_test = train_test_split(x_train_set, x_train_label, test_size=0.2, random_state=3)


#训练
early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=1)
nb_classes = 2
saveBestModel = keras.callbacks.ModelCheckpoint('./bestweight_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
model = createDenseNet(nb_classes=nb_classes, img_dim=[size,size,size,1], depth=densenet_depth, growth_rate=densenet_growth_rate, dropout_rate=0.1)
model.compile(loss=categorical_crossentropy, optimizer=SGD(lr=0.001, decay=1e-5, momentum=0.9, nesterov=True), metrics=['accuracy'])
model.summary()

model.fit(x_train_train, y_train_train,batch_size= batch_size, epochs=2000, validation_data=(x_train_test, y_train_test), verbose=2, shuffle=False, callbacks=[early_stopping,saveBestModel])
loss,accuracy = model.evaluate(x_train_train,y_train_train)
print('Training loss: %.4f, Training accuracy: %.2f%%' % (loss,accuracy))
loss,accuracy = model.evaluate(x_train_test,y_train_test)
print('Testing loss: %.4f, Testing accuracy: %.2f%%' % (loss,accuracy))

#保存模型
model.save("model.h5")
