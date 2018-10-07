# encoding:utf-8

from __future__ import print_function
from PIL import Image
from numpy import *
import numpy as np

def ImageToMatrix(filename):
    # 读取图片
    im = Image.open(filename)
    im = im.convert("L")
    data = im.getdata()
    data = np.matrix(data,dtype="float")/255
    data = data.tolist()
    return data
filepath = 'D:\\Code_python\\SVM\\new_pics\\'
savepath = "D:\\Code_python\\SVM\\pic_matrix\\"
picsize = 92*112
picnum = 2*34
data_all = np.zeros(shape=(picsize,picnum))
n = 0
for j in range(34):
    for i in range(2):
        filename =  filepath + str(i) + "_" + str(j) +".jpg"
        data = ImageToMatrix(filename)
        data_single = []
        for k in range(len(data)):
            data_single.append(data[k])
        data_all[:,n] = list(data[0])
        n = n+1
#print data_all
path = 'D:\\Code_python\\SVM\\pic_matrix\\pic_matrix.txt'
np.savetxt(path, data_all)

traindata = data_all[:,0:34]
testdata = data_all[:,34:68]

dp = [-1,1] * 17
trainNum=len(dp)
sigma=0.5
kMatrix = np.zeros((34,34))
Multimatrix = np.zeros((34,34))
for i in range(34):
    for j in range(34):

        a = np.array(traindata[:, i] - traindata[:, j]) ** 2

        kMatrix[i, j] = exp(-(sum(list(a))) / (2*sigma**2))
        print ("kMtrix",kMatrix[i,j])
        Multimatrix[i,j] = dp[i]*dp[j]*kMatrix[i,j]

inverse_Multimatrix = mat(Multimatrix).I
print(inverse_Multimatrix)
e = mat(np.ones((1, trainNum)))
alpha =inverse_Multimatrix * e.T
print("alpha",alpha)

