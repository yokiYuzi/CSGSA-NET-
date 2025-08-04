# dataset.py (修改后)

# -*- coding: utf-8 -*-

import matplotlib; 
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import pywt, os, copy
import torch
import numpy as np
import pandas as pd
from config import config
from torch.utils.data import Dataset
from sklearn.preprocessing import scale
from sklearn import preprocessing
from scipy import signal

import pyedflib
from scipy.signal import butter, filtfilt

from sklearn.model_selection import train_test_split

# 【修改】删除了全局变量 select_index，因为我们将通过参数传入文件索引

def NormalizeData1(v):
    v_min = v.min(axis=1).reshape((v.shape[0],1))
    v_max = v.max(axis=1).reshape((v.shape[0],1))
    return (v - v_min) / (v_max-v_min)

def NormalizeData(v):
    return (v - v.mean(axis=1).reshape((v.shape[0],1))) / (v.max(axis=1).reshape((v.shape[0],1)) + 2e-12) 

# wangxu add starting (这部分代码保持不变)
def get_filelist(path): 
    # ... (此处代码无变化) ...
    Filelist = []
    Filelist_final = []
    Filelist_test = []
    for home, dirs, files in os.walk(path):
        for filename in files: 
            spl = filename.split('.')
            filen = os.path.join(home, spl[0])
            if filen not in Filelist:
                Filelist.append(filen)         
                
    for string in Filelist: 
        string1 = string.rsplit('/', 1)
        string2 = string1[-1].rsplit('_',1)
        string_file = string1[0] + '/' + string2[0]
        
        if os.path.exists(string_file + '_fecg1.dat') or os.path.exists(string_file + '_fecg2.dat') :
            Filelist_final.append(string_file)
        
    test = pd.DataFrame(data=Filelist_final) 
    Filelist_final = test.drop_duplicates()
    
    return Filelist_final

def windowingSig(sig1, sig2, windowSize=128):
        signalLen = sig2.shape[1]
        signalsWindow1 = [sig1[:, int(i):int(i + windowSize)] for i in range(0, signalLen - windowSize +1, windowSize)]
        signalsWindow2 = [sig2[:, int(i):int(i + windowSize)] for i in range(0, signalLen - windowSize +1, windowSize)]

        return signalsWindow1, signalsWindow2

txt = []
class FECGDataset(Dataset):
    # 【修改】__init__ 方法现在接收一个 file_indices 列表
    def __init__(self, data_path="./ADFECGDB/", file_indices=[0]):
        super(FECGDataset, self).__init__()
        # 原始文件名列表，索引 0-4 分别对应 r01, r04, r07, r08, r10
        self.fileNames= ["r01.edf", "r04.edf", "r07.edf", "r08.edf", "r10.edf"]
        
        # 【修改】将接收到的文件索引列表传入数据准备函数
        ecgWindows, fecgWindows = self.prepareData(delay=5, file_indices=file_indices)
        
        # 数据集分割逻辑保持不变
        # 注意：对于验证和测试集，我们通常不需要再次分割，但这里保持原逻辑以减少改动
        # 当只有一个文件时，trainPercent=len-1 意味着几乎所有数据都用于 "train" 部分
        self.X_train, _, self.Y_train, _ = self.trainTestSplit(ecgWindows, fecgWindows, len(ecgWindows)-1)
        
        
    def readData(self, sigNum, path="./ADFECGDB/"):
        file_name = path + self.fileNames[sigNum]
        f = pyedflib.EdfReader(file_name)
        n = f.signals_in_file
        abdECG = np.zeros((n - 1, f.getNSamples()[0]))
        fetalECG = np.zeros((1, f.getNSamples()[0]))
        fetalECG[0, :] = f.readSignal(0)
        fetalECG[0, :] = scale(self.butter_bandpass_filter(fetalECG, 1, 100, 1000), axis=1)
        for i in np.arange(1, n):
            abdECG[i - 1, :] = f.readSignal(i)
        abdECG = scale(self.butter_bandpass_filter(abdECG, 1, 100, 1000), axis=1)

        abdECG = signal.resample(abdECG, int(abdECG.shape[1] / 5), axis=1)
        fetalECG = signal.resample(fetalECG, int(fetalECG.shape[1] / 5), axis=1)
        return abdECG, fetalECG
    
    # 【修改】prepareData 方法现在接收一个 file_indices 列表
    def prepareData(self, delay=5, file_indices=[0]):
        ecgAll, fecgAll = None, None
        
        # 标志位，用于处理第一个文件
        is_first_file = True
        
        # 遍历指定的文件索引列表
        for i in file_indices:
            ecg, fecg = self.readData(i)
            # 保持原有的通道选择和归一化逻辑
            ecg = ecg[range(2,3), :]
            ecg = NormalizeData1(ecg)
            delayNum = ecg.shape[0]
            fecgDelayed = self.createDelayRepetition(fecg, delayNum, delay)
            fecgDelayed = NormalizeData1(fecgDelayed)
            
            # 如果是第一个文件，直接赋值；否则，进行拼接
            if is_first_file:
                ecgAll = ecg
                fecgAll = fecgDelayed
                is_first_file = False
            else:
                ecgAll = np.append(ecgAll, ecg, axis=1)
                fecgAll = np.append(fecgAll, fecgDelayed, axis=1)

        # 分窗逻辑保持不变
        ecgWindows, fecgWindows = windowingSig(ecgAll, fecgAll, windowSize=128)
        return ecgWindows, fecgWindows

    # trainTestSplit 等其他辅助函数保持不变
    def trainTestSplit(self, sig, label, trainPercent, shuffle=False):
        # ... (此处代码无变化) ...
        X_train, X_test, y_train, y_test = train_test_split(sig, label, train_size=trainPercent, shuffle=shuffle)
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        return X_train, X_test, y_train, y_test
    
    def createDelayRepetition(self, signal, numberDelay=4, delay=10):
        # ... (此处代码无变化) ...
        signal = np.repeat(signal, numberDelay, axis=0)
        for row in range(1, signal.shape[0]):
            signal[row, :] = np.roll(signal[row, :], shift=delay * row)
        return signal

    def __butter_bandpass(self, lowcut, highcut, fs, order=5):
        # ... (此处代码无变化) ...
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=3, axis=1):
        # ... (此处代码无变化) ...
        b, a = self.__butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data, axis=axis)
        return y
    
    def __getitem__(self, index):        
        dataset_x = self.X_train[index,:,:]
        dataset_y = self.Y_train[index,:,:]
        return dataset_x, dataset_y

    def __len__(self):
        return self.X_train.shape[0]
    
if __name__ == '__main__':
    # 示例：加载 r01 和 r04 文件作为数据集
    d = FECGDataset(file_indices=[0, 1])
    print(d[0][0].shape)