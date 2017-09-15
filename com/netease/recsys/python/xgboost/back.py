# coding=utf-8
import xgboost as xgb
import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
now = time.time()
dataset = pd.read_csv("E:/Python/XGBoost-Train/com/netease/recsys/python/xgboost/data0/train.csv")
print "read done"

train_value = dataset.iloc[:, :32]
label_value = dataset.iloc[:, 32]

train_data, test_data, train_label, test_label = train_test_split(train_value, label_value, test_size=0.5, random_state=1)

params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'gamma': 0.05,  # 在树的叶子节点下一个分区的最小损失，越大算法模型越保守 。[0:]
    'max_depth': 10,  # 构建树的深度 [1:]
    # 'lambda':450,  # L2 正则项权重
    'subsample': 0.6,  # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1]
    'colsample_bytree': 0.7,  # 构建树树时的采样比率 (0:1]
    # 'min_child_weight':12,  # 节点的最少特征数
    'eta': 0.005,  # 如同学习率
    'seed': 710,
}
plst = list(params.items())

num_rounds = 100  # 迭代次数


# 划分训练集与验证集
xgtrain = xgb.DMatrix(train_data, label=train_label)
xgtest = xgb.DMatrix(test_data, label=test_label)

# return 训练和验证的错误率
watchlist = [(xgtrain, 'train'), (xgtest, 'eval')]

model = xgb.train(plst, xgtrain, num_rounds, watchlist)

# model.save_model('./xgb.model')  # 用于存储训练出的模型

tests = pd.read_csv("E:/Python/XGBoost-Train/com/netease/recsys/python/xgboost/test.csv")
test = tests.iloc[:, :32].values
xgtest = xgb.DMatrix(test)
preds = model.predict(xgtest, ntree_limit=model.best_iteration)

# 将预测结果写入文件，方式有很多，自己顺手能实现即可
np.savetxt('res.csv', preds)

#
# cost_time = time.time()-now
# print "end ......", '\n', "cost time:", cost_time, "(s)......"
