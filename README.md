# SJTU-M3DV
-------------------------------------------纵之以高宴，接之以清谈，请日试万言，倚马可待-----------------------------------------------------
    本篇为代码使用说明，具体内容可以参见SJTU M3DV: Medical 3D Voxel Classification（https://www.kaggle.com/c/
sjtu-m3dv-medical-3d-voxel-classification/overview），其中有详细对项目的描述。

    作者：陈昊
    班级：F1703407
    学号：517021910970
    
    
    本项目共包含8个文件，分别是：
      Submission.csv：输出的最终结果，AUC为0.70997
      train_val.csv：训练集ID及label，用于训练集的数据输入检索
      test.csv：测试集ID，用于测试集的数据输入检索
      model1.h5：第一个模型，权重为0.7
      model2.h5：第二个模型，权重为0.3
      getdata.py：包含数据输入、数据预处理、数据增强等函数
      DenseNet.py：主要使用的模型
      train.py：训练函数，调用以上两个文件，需要train_val.csv检索数据
      model_load.py：预测函数，需要model1.h5、model2.h5，需要train_val.csv检索数据
    
    由于训练集及测试集数据量大，因此不放在Github中。将model_load.py、两个模型与test.csv、Submission.csv放入与测试
集同一路径下，即可运行model_load.py。（仍然输出在Submission.csv中）
    
    由于时间仓促并不能很好地完善所有细节，有很多不足的地方希望得到指正！
