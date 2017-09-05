import numpy as np
import xgboost as xgb

data = np.random.rand(5, 10)  # 5 entities, each contains 10 features
print data
label = np.random.randint(2, size=5)  # binary target
dtrain = xgb.DMatrix(data, label=label)

dtest = dtrain

param = {'bst:max_depth': 2, 'bst:eta': 1, 'silent': 1, 'objective': "binary:logistic", 'nthread': 4,
         'eval_metric': 'auc'}

evallist = [(dtest, 'eval'), (dtrain, 'train')]

num_round = 10
bst = xgb.train(param, dtrain, num_round, evallist)

bst.dump_model('dump.raw.txt')
