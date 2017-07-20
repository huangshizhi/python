# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 09:49:53 2017

@author: huangshizhi

https://github.com/huangshizhi/pyLightGBM
修改如下例子
https://github.com/huangshizhi/pyLightGBM/blob/master/notebooks/regression_example_kaggle_allstate.ipynb

"""
import matplotlib.pyplot as plt
import gc
import numpy as np
import pandas as pd

from sklearn import metrics, model_selection
from sklearn.preprocessing import LabelEncoder
from pylightgbm.models import GBMRegressor

#定义随机种子
seed = 42
#加载数据
df_train = pd.read_csv(r"D:\kaggle\allstate\data\train.csv")
df_test = pd.read_csv(r"D:\kaggle\allstate\data\test.csv")

#Extracting loss from train and id from test
#损失值取对数，并转换为float类型
y = np.log(df_train['loss']+1).as_matrix().astype(np.float)
id_test = np.array(df_test['id'])

#合并数据
df = df_train.append(df_test, ignore_index=True)
del df_test, df_train
gc.collect()

#删除没用的列
df.drop(labels=['loss', 'id'], axis=1, inplace=True)
feature_list = df.columns.tolist()

#对取值离散的列进行编码，转换成数值，Transfrom categorical features cat from 1 to 116
le = LabelEncoder()
for col in df.columns.tolist():
    if 'cat' in col:
        df[col] = le.fit_transform(df[col])

#拆分成训练集和测试集          
df_train, df_test = df.iloc[:len(y)], df.iloc[len(y):]
del df
gc.collect()

#df_train_1 = df_train[:100]
#df_test_1 = df_test[:100]
#拆分成训练集和验证集
X = df_train.as_matrix()
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
X_test = df_test.as_matrix()

del df_train, df_test
gc.collect()


#模型训练,TRAINING GBMRegressor

gbmr = GBMRegressor(exec_path=r'F:\software\PYTHON_LIB\LightGBM\Release', # LighGBM安装目录
    config='',
    application='regression',
    num_iterations=100,
    learning_rate=0.1,
    num_leaves=10,
    tree_learner='serial',
    num_threads=4,
    min_data_in_leaf=10,
    metric='l2',
    feature_fraction=1.0,
    feature_fraction_seed=seed,
    bagging_fraction=1.0,
    bagging_freq=0,
    bagging_seed=seed,
    metric_freq=1,
    early_stopping_round=10
)

#learning_rate=0.1,MSE =1148
#learning_rate=0.05,MSE =1144
#learning_rate=0.01,MSE =1143.42
#learning_rate=0.01,num_leaves=30,MSE =1141.719
#learning_rate=0.01,num_leaves=300,num_threads=20,MSE =1141.719
#learning_rate=0.001,num_leaves=100,num_threads=100,MSE =1137
#metric='huber',learning_rate=0.01,num_leaves=100,MSE =1134.7
#metric='huber',learning_rate=0.001,num_leaves=30,MSE =1149.38

#模型训练
gbmr.fit(X_train, y_train, test_data=[(X_valid, y_valid)])

print("Mean Square Error:", metrics.mean_absolute_error(y_true=(np.exp(y_valid)-1), y_pred=(np.exp(gbmr.predict(X_valid))-1)))
print('Best round', gbmr.best_round)

feature_dict = dict(zip(range(len(feature_list)), feature_list))

#python3用法，dict-->DataFrame
df_fi = pd.DataFrame(list(gbmr.feature_importance().items()),columns=['feature', 'importance'])

df_fi = df_fi.replace({"feature": feature_dict})

del feature_dict, feature_list

top = 10

plt.figure()
df_fi.head(top).plot(kind='barh',
                     x='feature',
                     y='importance',
                     sort_columns=False,
                     legend=False,
                     figsize=(10, 6),
                     facecolor='#1DE9B6',
                     edgecolor='white')

plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')

#在测试集上预测
y_test_preds = gbmr.predict(X_test)
y_test_preds=(np.exp(y_test_preds)-1)


df_submission = pd.read_csv(r'D:\kaggle\allstate\data\sample_submission.csv')
df_submission['loss'] = y_test_preds
#保存预测结果
df_submission.to_csv(r'D:\kaggle\allstate\data\submission.csv',index=False)
