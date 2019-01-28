import os

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(path_list):
    for path in path_list:
        mkdir(path)