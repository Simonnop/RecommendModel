import pymongo
import numpy as np
from sklearn.neighbors import NearestNeighbors

# 设置精度
np.set_printoptions(precision=3)

# mongodb uri
conn_str = "mongodb://1.15.118.125:27017/"
# 客户端连接
client = pymongo.MongoClient(conn_str, serverSelectionTimeoutMS=5000)
collection_user = client.get_database("Lab").get_collection("Mission")

title_with_tags = {}
for doc in collection_user.find():
    mission_tags = doc.get("missionTags")
    mission_title = doc.get("title")
    title_with_tags[mission_title] = mission_tags


# 获取所有的 tag 以及其频次
def get_tag_times(dict_value_tags):
    tag_with_time = {}
    for key in dict_value_tags:
        for tag in dict_value_tags[key]:
            if tag not in tag_with_time:
                tag_with_time[tag] = 1
            else:
                tag_with_time[tag] += 1
    return tag_with_time


# 定义 0 矩阵
dist_matrix = np.zeros((len(title_with_tags), len(get_tag_times(title_with_tags)),))
# 获取 tags 列表
tags_list = list(get_tag_times(title_with_tags).keys())
# 获取 title 的 list
title_list = list(title_with_tags.keys())
# 根据 tag 生成矩阵
for mission_index in range(len(title_with_tags)):
    for tag in title_with_tags[title_list[mission_index]]:
        for tag_index in range(len(tags_list)):
            if tag == tags_list[tag_index]:
                dist_matrix[mission_index][tag_index] = 1

# knn算法计算距离
calcu_nbr = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(dist_matrix)
distances, indices = calcu_nbr.kneighbors(dist_matrix)


# distance: 距离
# indices: 对应序号

# 根据矩阵进行寻找
def find_similar_mission(title_str, num, max_distance):
    similar_mission_title_list = []
    my_index = title_list.index(title_str)
    print("为 " + title_str + " 寻找相似任务: " + str(title_with_tags[title_str]))
    pointer = 0
    count = 0
    while count < num:
        if indices[my_index][pointer] == my_index:
            pointer += 1
            continue
        elif distances[my_index][pointer] > max_distance:
            break
        else:
            similar_mission_title_list.append(title_list[indices[my_index][pointer]])
            count += 1
            pointer += 1
    for mis_title in similar_mission_title_list:
        print(" -> " + mis_title + " " + str(title_with_tags[mis_title]))
    return similar_mission_title_list


# 需要的个数
require_num = 5
max_distance = 1.5
# 测试输出
find_similar_mission(
    '喻园管理论坛: Intelligent Simulation Optimization: An Example in Multi-fidelity Simulation Modeling',
    require_num, max_distance)
find_similar_mission(
    '二十大宣讲',
    require_num, max_distance)
find_similar_mission(
    '管理学院第二届工商管理学科高端论坛系列活动: 学科建设座谈会',
    require_num, max_distance)
