import os
import glob
import numpy as np
import math
from .util import mkdir
def process_eye_result(result):
    dims = 54
    eyes = [1,3,5,9,13,15,16,18]

    others = [0,2,4,8,12,14,16,17]
    new_eyes = [index+3 for index in eyes]
    new_others = [index+3 for index in others]
    rows, _ = result.shape
    new_result = np.zeros([rows, 54])
    new_result[:, new_eyes] = result[:,:]
    new_result[:, new_others] = result[:,:]
    new_result = new_result * 100
    return new_result

def process_head_result(result_file,target_root,gain,smooth=True,padding=True):
    mkdir(target_root)
    dims = 54
    result = np.loadtxt(result_file)
    to_result = process_head(data = result,padding=padding,smooth = smooth,gain=gain)
    basename = os.path.basename(result_file)
    target_name = os.path.join(target_root, basename).replace("phoneme","skeleton")
    
    np.savetxt(target_name,to_result)

def euler2quaternion(euler):
    X, Y, Z = euler[0], euler[1], euler[2]
    X, Y, Z =   (X - 360) * math.pi / 180 if X > 180 else X * math.pi / 180, \
                (Y - 360) * math.pi / 180 if Y > 180 else Y * math.pi / 180, \
                (Z - 360) * math.pi / 180 if Z > 180 else Z * math.pi / 180
    c1 = math.cos(Y * 0.5)
    s1 = math.sin(Y * 0.5)

    c2 = math.cos(Z * 0.5)
    s2 = math.sin(Z * 0.5)

    c3 = math.cos(X * 0.5)
    s3 = math.sin(X * 0.5)

    w = c1 * c2 * c3 + s1 * s2 * s3
    x = s1 * s2 * c3 + c1 * c2 * s3
    y = s1 * c2 * c3 - c1 * s2 * s3
    z = c1 * s2 * c3 - s1 * c2 * s3
    return np.array([x, y, z, w])

def process_head(data,padding,smooth=True,gain=None):
    """
    padding: head
    """
    dims = 54
    max_vector = np.loadtxt("./statistic/max.txt")
    min_vector = np.loadtxt("./statistic/min.txt")
    to_result = data * (max_vector - min_vector) + min_vector
    if padding:
        temp = np.zeros([data.shape[0],dims])
        temp[:,:3] = to_result
        to_result = temp
    if gain != None:
        to_result = to_result*gain
    if smooth == True:
        to_result = smoothing(to_result)
    return to_result

def smoothing(output):
    result = np.zeros_like(output)
    result[0,:] = output[0,:]
    for i in range(1,len(result)):
        result[i,:] = 0.5*result[i-1,:] + 0.5*output[i,:]
    return result


def process_for_new(result_file,gain,target_root="./new",smooth=True):
    """
    动作：不用padding
    """
    mkdir(target_root)
    if not os.path.exists(target_root):
        os.makedirs(target_root)
    head = np.loadtxt(result_file)
    head = process_head(data = head,padding=False,smooth=smooth,gain=gain)
    head = list(head)

    basename = os.path.basename(result_file)
    target_name = os.path.join(target_root, basename).replace("phoneme","skeleton")
    with open(target_name, "w") as f:
            
        for head_ in head:

            head_ = list(head_)
            head_.insert(0,"Bip001 Head")
            head_ = [str(i) for i in head_]
            f.write(",".join(head_))
            f.write("\n")

def process_for_new_siyuanshu(result_file, gain,target_root="./siyuanshu",smooth=True):
    """
    动作：不用padding
    四元数
    """
    mkdir(target_root)
    if not os.path.exists(target_root):
        os.makedirs(target_root)
    head = np.loadtxt(result_file)
    head = process_head(data=head,gain=gain,padding=False,smooth=smooth)
    head = list(head)

    basename = os.path.basename(result_file)
    target_name = os.path.join(target_root, basename).replace("phoneme","skeleton")
    with open(target_name, "w") as f:
            
        for head_ in head:
            head_ = euler2quaternion(head_)
            head_ = list(head_)
            head_.insert(0,"Bip001 Head")
            head_ = [str(i) for i in head_]
            f.write(",".join(head_))
            f.write("\n")



