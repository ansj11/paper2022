#!/usr/bin/env python3
from meghair.utils import io
import megbrain as mgb
import megskull
import megskull.opr.all as O
from megskull.graph import FpropEnv
from tqdm import tqdm
import nori2 as nori

import json
import pickle as pkl
import numpy as np
from numpy import random as nr
import os
import sys
import cv2
import argparse
import time
from IPython import embed
from landstack.utils import misc
import mgfpy
from landstack.megface.detectcore import loadyuv, detect, MegFaceAPI

def detectrot(mgf, img, dtype='align', thres=0.99):
    maxscore = -1
    maxface = None
    isMaxFace = True
    for rottype in (mgfpy.Orient.MGF_UP, mgfpy.Orient.MGF_RIGHT, mgfpy.Orient.MGF_DOWN, mgfpy.Orient.MGF_LEFT):
        face = detect(mgf, img, stage=dtype, thres=thres, maxface=isMaxFace, gt=None, options={'orient':rottype})
        if face:
            score = face['conf']
            if maxscore < score:
                maxscore = score
                maxface = face
            if maxscore >= thres:
                return maxface
    return maxface

def detectnori():
    device = 'gpu0'
    mgf = MegFaceAPI(
        megface_lib_path='/home/xiongpengfei/megface-v2/lib/libmegface.so',
        face_model_root='/unsullied/sharefs/xiongpengfei/Isilon-alignmentModel/3rdparty/Model24/FaceModel/',
        version='2.4',
        device=device,
    )
    mgf.register_det_rect_config('detector_rect.densebox.small.v1.3.conf')
    mgf.register_det_81_config('lmk.postfilter.small.v1.2.conf')
    mgf.register_lm_81_config('detector.mobile.v3.accurate.conf')

    fn = nori.Fetcher()
    noripath = '/unsullied/sharefs/_research_facerec/face-recognition-benchmark/repos/byjc1000-query/origin.pkl'
    packs = pkl.load(open(noripath, 'rb'))
    
    for pack in packs:
        nori_id = pack['nori_key']
        tmp = pkl.loads(fn.get(nori_id))
        img = cv2.imdecode(np.fromstring(tmp['img'], np.uint8), cv2.IMREAD_COLOR)
        
        face = detectrot(mgf, img, 'lm_81', 0.2)
        if face is None:
            continue

        for i in range(face['pts'].shape[0]):
            cv2.circle(img, (int(face['pts'][i,0]), int(face['pts'][i,1])), 3, (0,0,255), -1)
        cv2.imwrite('byjc1000-query-%s-%.2f.jpg'%(nori_id, face['conf']), img)

if __name__ == '__main__':
    detectnori()






