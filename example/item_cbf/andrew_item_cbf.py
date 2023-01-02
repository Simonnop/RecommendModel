import csv
from collections import defaultdict

import numpy as np
import numpy.ma as ma
import pandas as pd
import tabulate
import tensorflow as tf
from numpy import genfromtxt
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras import Model

pd.set_option("display.precision", 1)

# 构建神经网络
num_outputs = 32
tf.random.set_seed(1)
item_NN = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_outputs),
])

# 导入数据
with open('data/content_item_train_header.txt', newline='') as f:
    item_features = list(csv.reader(f))[0]
item_train = genfromtxt('data/content_item_train.csv', delimiter=',')
item_vecs = genfromtxt('data/content_item_vecs.csv', delimiter=',')
# 把csv读成dict
movie_dict = defaultdict(dict)
count = 0
with open('data/content_movie_list.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for line in reader:
        if count == 0:
            count += 1  # skip header
        else:
            count += 1
            movie_id = int(line[0])
            movie_dict[movie_id]["title"] = line[1]
            movie_dict[movie_id]["genres"] = line[2]
ivs = 3  # item genre vector start
i_s = 1  # start of columns to use in training, items
num_item_features = item_train.shape[1] - 1

# 预处理
scalerItem = StandardScaler()
scalerItem.fit(item_train)
item_train = scalerItem.transform(item_train)

# 向神经网络放置数据
input_item_m = tf.keras.layers.Input(shape=num_item_features)
vm_m = item_NN(input_item_m)
vm_m = tf.linalg.l2_normalize(vm_m, axis=1)
model_m = Model(input_item_m, vm_m)
model_m.summary()


# 计算样本距离函数
def sq_dist(a, b):
    d = 0.0
    for i in range(len(a)):
        d = d + np.square(a[i] - b[i])
    return d


# 获得对应的item的属性(genre)
def get_item_genre(item, ivs, item_features):
    offset = np.where(item[ivs:] == 1)[0][0]
    genre = item_features[ivs + offset]
    return genre, offset


scaled_item_vecs = scalerItem.transform(item_vecs)
vms = model_m.predict(scaled_item_vecs[:, i_s:])
print(f"size of all predicted movie feature vectors: {vms.shape}")

count = 5
dim = len(vms)
dist = np.zeros((dim, dim))

for i in range(dim):
    for j in range(dim):
        dist[i, j] = sq_dist(vms[i, :], vms[j, :])

m_dist = ma.masked_array(dist, mask=np.identity(dist.shape[0]))  # mask the diagonal

disp = [["movie1", "genres", "movie2", "genres"]]
for i in range(count):
    min_idx = np.argmin(m_dist[i])
    movie1_id = int(item_vecs[i, 0])
    movie2_id = int(item_vecs[min_idx, 0])
    genre1, _ = get_item_genre(item_vecs[i, :], ivs, item_features)
    genre2, _ = get_item_genre(item_vecs[min_idx, :], ivs, item_features)

    disp.append([movie_dict[movie1_id]['title'], genre1,
                 movie_dict[movie2_id]['title'], genre2])

table = tabulate.tabulate(disp, tablefmt='grid', headers="firstrow", floatfmt=[".1f", ".1f", ".0f", ".2f", ".2f"])
print(table)
