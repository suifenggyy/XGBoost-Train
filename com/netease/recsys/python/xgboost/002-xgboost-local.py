#coding=utf-8
import xgboost as xgb
import pandas as pd
import time
import numpy as np


now = time.time()

dataset = pd.read_csv("data1/train-part35.csv")

train = dataset.iloc[:, :32].values
labels = dataset.iloc[:, 32:].values

tests = pd.read_csv("test.csv")
test = tests.iloc[:, :32].values


params={
'booster':'gbtree',
'objective': 'binary:logistic',
'gamma': 0.05,  # 在树的叶子节点下一个分区的最小损失，越大算法模型越保守 。[0:]
'max_depth': 10,  # 构建树的深度 [1:]
#'lambda':450,  # L2 正则项权重
'subsample': 0.6,  # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1]
'colsample_bytree': 0.7,  # 构建树树时的采样比率 (0:1]
#'min_child_weight':12,  # 节点的最少特征数
'eta': 0.005,  # 如同学习率
'seed':710,
}

plst = list(params.items())

offset = 3000000  #

num_rounds = 100  # 迭代次数
xgtest = xgb.DMatrix(test)

# 划分训练集与验证集
xgtrain = xgb.DMatrix(train[:offset,:], label=labels[:offset])
xgval = xgb.DMatrix(train[offset:,:], label=labels[offset:])

# return 训练和验证的错误率
watchlist = [(xgtrain, 'train'), (xgval, 'val')]

model = xgb.train(plst, xgtrain, num_rounds, watchlist)
#model.save_model('./model/xgb.model')  # 用于存储训练出的模型
preds = model.predict(xgtest)

# 将预测结果写入文件，方式有很多，自己顺手能实现即可
np.savetxt('submission_xgb_MultiSoftmax.csv', np.c_[range(1,len(test)+1), preds],
                 delimiter=',', header='ImageId,Label', comments='', fmt='%d')


cost_time = time.time()-now
print "end ......", '\n', "cost time:", cost_time, "(s)......"
