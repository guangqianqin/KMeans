import numpy as np
import random
import matplotlib.pyplot as plt
import time
def distEclud(vecA, vecB):
    return np.sqrt(sum(vecA-vecB)**2)
# 随机生成质心
def initCent(dataset, k):
    numsamples, dim = dataset.shape
    centroids = np.zeros((k, dim))
    for i in range(k):
        index = int(np.random.uniform(0, numsamples))
        centroids[i,:] = dataset[index,:]
    return centroids
def Kmeans(data, k):
    numsamples = data.shape[0]
    #print(data[0]) # 98, 2
    #print(numsamples) # 99
    # 样本的属性，第一列保存该样本属于哪个簇，第二列保存该样本跟它所属簇的误差
    clusterData = np.array(np.zeros((numsamples, 2)))
    clusterchanged = True
    # 初始化
    centroids = initCent(data, k)

    #print(centroids)

    while clusterchanged:
        clusterchanged=False
        for i in range(numsamples):
            # 最小距离
            minDist = 10000.0
            # 样本所属的簇
            minIndex=0
            for j in range(k):
                distance = distEclud(centroids[j,:], data[i,:])

                if distance < minDist:
                    minDist=distance
                    clusterData[i, 1] = minDist
                    minIndex=j
            # 如果样本的所属的簇发生了变化
            if clusterData[i, 0]!=minIndex:
                clusterchanged=True
                clusterData[i, 0] = minIndex
        #print(clusterData)
        for j in range(k): # 更新质心
            #获取第j个簇所有的样本所在的索引
            cluster_index = (np.argwhere([clusterData[:,0]==j]))[:,1]
            #print(cluster_index)
            #time.sleep(1000)
            points_cluster = data[cluster_index]
            # 计算新质心
            centroids[j,:] = np.mean(points_cluster, axis=0)
    return centroids, clusterData

def show(data, k, centroids, clusterdata):
    num, dim = data.shape
    if dim!=2:
        print('The shape of original data is error!')
        return 1
    # 用不同颜色形状来表示各个类别
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'dr', '<r', 'pr']
    if k > len(mark):
        print('your k is too large!')
        return 1
    # 画样本点
    for i in range(num):
        markIndex = int(clusterdata[i,0])
        plt.plot(data[i,0], data[i,1], mark[markIndex])
    # 用不同颜色形状来表示各个类别
    mark = ['*r', '*b', '*g', '*k', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # 画质心点
    for i in range(k):
        plt.plot(centroids[i,0], centroids[i,1], mark[i],markersize=20)
    plt.show()

def main():
    import time
    datamat=[]
    for i in range(100):
        random_data = [random.randint(1,100), random.randint(1,100)]
        if not random_data in datamat:
            datamat.append(random_data)
    #print(datamat)
    datamat = np.array(datamat)
    #print(datamat)
    #time.sleep(1000)
    myCentroids, clusterdata = Kmeans(datamat, 4)
    print(myCentroids)
    show(datamat, 4, myCentroids, clusterdata)

if __name__=='__main__':
    main() 
