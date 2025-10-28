import numpy as np
import cv2
from sklearn.cluster import DBSCAN



def cluster(mask:np.ndarray, epsilon=2, min_samples=3):
    '''
    epsilon: 邻域半径
    min_samples: 最小样本数
    '''
    # 获取非零像素点的坐标
    points = np.argwhere(mask)

    # 使用DBSCAN算法进行聚类
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    labels = dbscan.fit_predict(points)

    # 创建标记数组，与mask相同大小
    labeled_mask = np.zeros_like(mask)

    # 将每个区域的像素点标记为相应的区域标签
    for point, label in zip(points, labels):
        x, y = point
        labeled_mask[x, y] = label + 1  # 标签从1开始，避免与背景标签0重叠

    # 连通组件分析
    num_labels, labeled_mask = cv2.connectedComponents(labeled_mask.astype(np.uint8))

    # 后处理，将每个聚类内部的零标签设置为1
    for label in range(1, num_labels):
        cluster_mask = np.where(labeled_mask == label, 1, 0)
        if np.sum(cluster_mask) == 0:  # 如果聚类内部没有非零像素点
            labeled_mask[cluster_mask == 1] = 1

    return labeled_mask

# 假设有一张h*w的 mask，存储在二维numpy数组中
mask = np.array([[0, 1, 0, 0, 1],
                 [1, 1, 0, 0, 0],
                 [0, 0, 1, 0, 1],
                 [0, 0, 1, 1, 0]])
cluster(mask)