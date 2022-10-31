#!/usr/bin/env python3

import json
import sys
import math
import argparse
import cv2
import os
from tqdm import tqdm
import nori2 as nori
import pickle as pkl
import numpy as np
import megbrain as mgb
from landstack.utils.detect import mdetect
from landstack.megface.lmkcore import loadmodel, getlm

if __name__ == '__main__':
    mgb.config.set_default_device('gpu0')
    detmodel = mdetect('gpu0', 'x')
    lmkpath = '/unsullied/sharefs/_research_facelm/Isilon-modelshare/model_zoo/release/lmk.detect.v4.xlarge.171020/model.pred.prob'
    lmkmodel = loadmodel(lmkpath, 'pred')

    # get 81
    img = cv2.imread('tmp/datasets/test.jpg', cv2.IMREAD_COLOR)
    print(img.shape)
    rect = [10, 10, 100, 100]
    pack = detmodel.rect81(img, rect)
    ld = pack['gt81'].reshape(-1,2)
    for i in range(len(ld)):
        cv2.circle(img, (int(ld[i,0]), int(ld[i,1])), 3, (0,0,255), -1)
    cv2.imshow('rtt', img)
    cv2.waitKey(0)

    ld = getlm(img, pack['gt81'], lmkmodel, False)
    ld = ld.reshape(-1,2)
    for i in range(len(ld)):
        cv2.circle(img, (int(ld[i,0]), int(ld[i,1])), 3, (255,0,0), -1)
    cv2.imshow('rtt', img)
    cv2.waitKey(0)



