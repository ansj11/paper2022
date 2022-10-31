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
from megskull.graph import FpropEnv
from meghair.utils import visualize, io
from landstack.megface.detectcore import track, MegFaceAPI
from landstack.utils import misc

if __name__ == '__main__':
    videodir = '/unsullied/sharefs/_research_facelm/Isilon-datashare/tmp/dingding/faceAll/yuv960_540/'
    paths = os.listdir(videodir)
    paths.sort(key=lambda x:int(x.split('.')[0]))

    idx = 0
    trackids = []
    for path in paths:
        imgpath = '%s/%s'%(videodir, path)
        h, w = 540, 960
        with open(imgpath, 'rb') as f:
            img = f.read()
            if img is None:
                continue                        
        frame = misc.yuv2gray(np.fromstring(img, np.uint8),h,w)
        cv2.imwrite('%d_%s.jpg'%(idx, path), frame)
        idx += 1

