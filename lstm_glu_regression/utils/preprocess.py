import numpy as np

def compute_max_min(data):
    """
    data: list type of data
    返回最大最小值,
    并写入txt文件中
    """
    data = np.vstack(data)
    data[data>180] = data[data>180]-360
    max_vector = np.max(data, axis=0)
    min_vector = np.min(data, axis=0)
    np.savetxt("/data/statistic/max.txt",max_vector)
    np.savetxt("/data/statistic/min.txt",min_vector)
    return max_vector,min_vector

def transform(data):
    """
    大于180的转到-180到0
    """
    data[data>180] = data[data>180]-360
    return data

def preprocessing(data):
    data = np.vstack(data)
    data[data>180] = data[data>180]-360
    max_vector = np.max(data,axis=0)
    min_vector = np.min(data,axis=0)
    data = (data-min_vector)/(max_vector-min_vector)
    return data
