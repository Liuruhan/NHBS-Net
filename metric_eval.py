#!/usr/bin/python2.6
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt
import SimpleITK as sitk

# 计算DICE系数，即DSI
def calDSI(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    DSI_s, DSI_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
                DSI_s += 1
            if binary_GT[i][j] == 255:
                DSI_t += 1
            if binary_R[i][j] == 255:
                DSI_t += 1
    if DSI_t == 0:
        DSI_t = 1
    DSI = 2 * DSI_s / DSI_t
    # print(DSI)
    return DSI


# 计算VOE系数，即VOE
def calVOE(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    VOE_s, VOE_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255:
                VOE_s += 1
            if binary_R[i][j] == 255:
                VOE_t += 1
    if (VOE_t + VOE_s) == 0:
        VOE_t = 1
    VOE = 2 * (VOE_t - VOE_s) / (VOE_t + VOE_s)
    return VOE


# 计算RVD系数，即RVD
def calRVD(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    RVD_s, RVD_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255:
                RVD_s += 1
            if binary_R[i][j] == 255:
                RVD_t += 1
    if RVD_s == 0:
        RVD_s = 1
    #print(RVD_t, RVD_s)
    RVD = RVD_t / RVD_s - 1
    return RVD


# 计算Prevision系数，即Precison
def calPrecision(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    P_s, P_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
                P_s += 1
            if binary_R[i][j] == 255:
                P_t += 1
    Precision = P_s / P_t
    return Precision


# 计算Recall系数，即Recall
def calRecall(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    R_s, R_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
                R_s += 1
            if binary_GT[i][j] == 255:
                R_t += 1
    if R_t == 0:
        Recall = 0.0001
    else:
        Recall = R_s / R_t
    return Recall

def calDice(binary_GT, binary_R):
    if np.max(binary_GT) == 255:
        pred = binary_GT / 255
    else:
        pred = binary_GT
    if np.max(binary_R) == 255:
        true = binary_R / 255
    else:
        true = binary_R
    union = true * pred
    dice = 2 * np.sum(union) / (np.sum(true) + np.sum(pred))
    return dice

def calHausdorff(binary_GT, binary_R):
    img_GT = cv2.imread(binary_GT, 0)
    img_R = cv2.imread(binary_R, 0)
    mrk = 1
    for i in range(img_GT.shape[0]):
        for j in range(img_GT.shape[1]):
            if np.array(img_GT)[i][j] == 255:
                mrk = 0
    if mrk == 1:
        return 0.0, 1000, 100 
    lP = sitk.ReadImage(binary_GT) #pred
    lT = sitk.ReadImage(binary_R) #real
    labelTrue = lT
    labelPred = lP
    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue, labelPred)
    aveHausdorff = hausdorffcomputer.GetAverageHausdorffDistance()
    hausdorff = hausdorffcomputer.GetHausdorffDistance()

    dicecomputer = sitk.LabelOverlapMeasuresImageFilter()
    dicecomputer.Execute(labelTrue, labelPred)
    h_dice = dicecomputer.GetDiceCoefficient()
    return h_dice, hausdorff, aveHausdorff

def metric_eval(pred_img, real_img):
    # step 1：读入图像，并灰度化
    img_GT = cv2.imread(pred_img, 0)
    img_R = cv2.imread(real_img, 0)
    # imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   # 灰度化
    # img_GT = img_GT[:,:,[2, 1, 0]]
    # img_R  = img_R[:,: [2, 1, 0]]

    # step2：二值化
    # 利用大律法,全局自适应阈值 参数0可改为任意数字但不起作用
    ret_GT, binary_GT = cv2.threshold(img_GT, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ret_R, binary_R = cv2.threshold(img_R, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # step 3： 显示二值化后的分割图像与真值图像
    plt.figure()
    plt.subplot(121), plt.imshow(binary_GT), plt.title('真值图')
    plt.axis('off')
    plt.subplot(122), plt.imshow(binary_R), plt.title('分割图')
    plt.axis('off')
    #plt.show()

    # step 4：计算DSI
    #print('（1）DICE计算结果，      DSI       = {0:.4}'.format(calDSI(binary_GT, binary_R)))  # 保留四位有效数字

    # step 5：计算VOE
    #print('（2）VOE计算结果，       VOE       = {0:.4}'.format(calVOE(binary_GT, binary_R)))

    # step 6：计算RVD
    #print('（3）RVD计算结果，       RVD       = {0:.4}'.format(calRVD(binary_GT, binary_R)))

    # step 7：计算Precision
    #print('（4）Precision计算结果， Precision = {0:.4}'.format(calPrecision(binary_GT, binary_R)))

    # step 8：计算Recall
    #print('（5）Recall计算结果，    Recall    = {0:.4}'.format(calRecall(binary_GT, binary_R)))
    #DSI = calDSI(binary_GT, binary_R)
    VOE = calVOE(binary_GT, binary_R)
    RVD = calRVD(binary_GT, binary_R)
    Precision = calPrecision(binary_GT, binary_R)
    Recall = calRecall(binary_GT, binary_R)
    Dice = calDice(binary_GT, binary_R)
    h_dice, hausdorff, aveHausdorff = calHausdorff(pred_img, real_img)
    return  VOE, RVD, Precision, Recall, Dice, h_dice, hausdorff, aveHausdorff

if __name__ == '__main__':
    VOE, RVD, Precision, Recall, Dice, h_dice, hausdorff, aveHausdorff = metric_eval('1pred.png', '1true.png')
    print("VOE", VOE)
    print("RVD", RVD)
    print("Precision", Precision)
    print("Recall", Recall)
    print("Dice", Dice)
    print("h_dice", h_dice)
    print("hausdorff", hausdorff)
    print("aveHausdorff", aveHausdorff)

