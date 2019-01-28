import os
import numpy as np
import json
from pprint import pprint
"""
输入file，得到index作为模型输入，主要inference使用
三个功能:
1.函数：input_repeat_file:
    输入repeat过后的file：里面存放着<s><s>欢欢欢迎迎...
2.函数：input_phoneme_file：
    输入原始phoneme文件
        中间会自动生成repeat词，其中除了开始和结尾的silence，中间部分不检测
        开头的silence <s>,结尾的silence </s>
3.函数：input_phoneme_file_with_silence:
    输入原始phoneme文件
        中间会自动repeat词，所有的silence都会检测
        所有的silence都标记为<s>
"""

def input_phoneme_file(in_file,print_or_not):
    """
    输入原始的phoneme_file
    """
    data = get_repeat_file(in_file)
    word_dict = load_dict(r"D:\Listener\DataDriven\data\statistic\char2index_new.txt")
    index_list = word2index(data, word_dict)
    if print_or_not:
        print("repeat:")
        print(data)
        print(index_list)
    return index_list

def input_phoneme_file_with_silence(in_file, word_dict):
    """
    输入原始的phoneme_file
    """
    data = get_repeat_file_with_silence(in_file)
    word_dict = load_json_dict(word_dict)
    

    index_list = word2index(data, word_dict)

    return index_list

def load_json_dict(json_file):
    with open(json_file,"r") as f:
        d = json.load(f)
    return d

def input_repeat_file(in_file,print_or_not=False):
    """
    输入in_file,是repeat之后的文件
    """
    with open(in_file,"r",encoding="utf-8") as f:
        line = f.readlines()[0].strip()
        data = line.split(" ")
    word_dict = load_dict("./char2index_new.txt")

    
    index_list = word2index(data, word_dict)
    if print_or_not:
        print("repeat:")
        print(data)
        print(index_list)
    return index_list
    
def word2index(word_list, word_dict):
    """
    输入需要转成index的word_list，和需要词典word_dict.
    word:词典存在word，返回对应index,不存在则返回"UNKNOW"对应的
    """
    index_list = []
    for word in word_list:
        if word in word_dict:
            index_list.append(word_dict[word])
        else:
            # if word == "</s>":

            index_list.append(word_dict["<UNKNOW>"])
    return index_list

def load_dict(dict_file):
    """
    输入dict file,返回词典
    """
    word_dict = {"PAD":0,"UNKNOW":1}
    with open(dict_file,"r",encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            word, index = line.split("\t")
            index = int(index)
            word_dict[word] = index
    return word_dict

def get_repeat_file(file):
    """
    输入file，返回repeat的word list
    """
    w, s_time, e_time = get_time_single_file(file)
    sentence = get_sentence_with_time(w, s_time, e_time)
    return sentence

def get_repeat_file_with_silence(file):
    """
    检测中间部分的silence部分，
    此处对应的词典是开头和结尾以及中间部分的silence都embedding为"<s>"
    """
    w, s_time, e_time = get_time_single_file_with_silence(file)
    sentence = get_sentence_with_time(w, s_time, e_time)
    return sentence


def get_time_single_file(file):
    """
    输入file,返回检测出的词列表，以及开始和结束时间，
    分别存放在w,s_time,e_time
    只检测中间的词，以及开头和结尾的silence，中间的silence不检测
    """
    with open(file, encoding="utf-8") as f:

        lines = f.readlines()
        lines = lines[1:]

        length = len(lines)
        w_index = []
        w = []
        s_time = []
        e_time = []
        if length > 0:
            for index in range(length):
                line = lines[index].strip()
                line = line.split(" ")
                if len(line) == 4:
                    w_index.append(index)
                    w.append(line[-1])
                    s_time.append(float(line[0]))
            e_time = s_time[1:]
            e_time.append(float(lines[-1].split(" ")[1]))

    return w, s_time, e_time

def get_time_single_file_with_silence(in_file):
    """
    检测开始的、中间的、结尾的silence，以及词 
    """
    with open(in_file, encoding="utf-8") as f:
        lines = f.readlines()
        lines = lines[1:]
        length = len(lines)
    w = []
    s_time = []
    e_time = []
    if length > 0:
        for index in range(length):
            line = lines[index].strip()
            line = line.split(" ")
            phoneme = line[-1]
            if len(line) == 4:
                if phoneme == "</s>":
                    w.append("<s>")
                else:
                    w.append(phoneme)
                s_time.append(float(line[0]))
            if len(line) == 3:
                if phoneme == "sil":
                    w.append("<s>")
                    s_time.append(float(line[0]))
    e_time = s_time[1:]
    e_time.append(float(lines[-1].split(" ")[1]))
    return w, s_time, e_time


                

def get_sentence_with_time(w_list, s_time, e_time):
    """
    输入词列表：w_list,
    对应的开始时间列表:s_time
    和结束时间列表：e_time
    """
    fps = 30
    sentence_with_time = []
    a = 0
    for w, s, e in zip(w_list, s_time, e_time):

        start_fps = int(float(s) * fps)
        end_fps = int(float(e) * fps)
        frames = end_fps - start_fps
        # print("start:{}\tend:{}".format(start_fps, end_fps))
        # a += frames
        # print(a)
        sentence_with_time.extend([w] * frames)
    return sentence_with_time

if __name__ == "__main__":
    # file = r"D:\Listener\DataDriven\danlu\lstm-head\test\fenduan-test\test1\1_phoneme.txt"
    # index = input_phoneme_file_with_silence(file)
    # print(index)
    word_dict = load_json_dict(r"D:\Listener\DataDriven\data\DataProcessingCode\word_dict.json")
    print(word_dict)
    for word in word_dict:
        print(word_dict[word])