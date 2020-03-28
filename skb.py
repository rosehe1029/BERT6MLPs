#环境配置：https://github.com/charles9n/bert-sklearn

import os
import sys
import csv

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import classification_report
from bert_sklearn import BertClassifier
from bert_sklearn import load_model
from sklearn.model_selection import train_test_split

data_df = pd.read_csv('data/nCoV_100k_train.labled.csv',
                      skiprows=[0], dtype={'label': np.str},
                      names=['id', 'time', 'account', 'content', 'pic', 'video', 'label'],
                      skipinitialspace=True)
## 354 个数据没有微博内容
data_df_na = data_df[data_df['content'].isna()]
data_df_na

## 查看data_df_na 的标签分布,标签为0的占93.79%
data_df_na['label'].value_counts()

## 取微博内容做训练数据
data_df_not_na = data_df[data_df['content'].notna()]
data_df_not_na

## 查看标签分布
data_df_not_na['label'].value_counts()

## 去除噪声标签,获得训练数据
data_df_not_na_label = data_df_not_na[data_df_not_na['label'].isin(['0', '1', '-1'])]
data_df_not_na_label['label'].value_counts()
train_df, dev_df = train_test_split(data_df_not_na_label, test_size=0.2, shuffle=True)

## 准备模型的数据

X_train, y_train = train_df['content'], train_df['label']
X_dev, y_dev = dev_df['content'], dev_df['label']

# define model
model = BertClassifier('bert-base-uncased')
model.validation_fraction = 0.0
model.learning_rate = 3e-5
model.gradient_accumulation_steps = 1
model.max_seq_length = 64
model.train_batch_size = 1
model.eval_batch_size = 1
model.epochs = 1

# fit
model.fit(X_train, y_train)

# score
accy = model.score(X_dev, y_dev)

test_df = pd.read_csv('data/nCov_10k_test.csv',
                      skiprows=[0],
                      names=['id', 'time', 'account', 'content', 'pic', 'video'])
test_df_not_na = test_df[test_df['content'].notna()]
## 直接设定没有微博内容的label为0
test_df_na = test_df[test_df['content'].isna()]
test_df_na['label'] = 0

X_test = test_df_not_na['content']
y_test_pred = model.predict(X_test)
test_df_not_na['label'] = y_test_pred
new_test_df = pd.concat([test_df_not_na, test_df_na]).sort_index()
temp_df = pd.DataFrame(columns=['id', 'y'])
temp_df['id'] = new_test_df['id']
temp_df['y'] = new_test_df['label']
temp_df.to_csv('submit.csv', encoding='utf-8', index=None)
