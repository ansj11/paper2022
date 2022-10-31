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
from neupeak.utils.misc import set_mgb_default_device
from detect.detxp import detect

if __name__ == '__main__':
    #mgb.config.set_default_device(gpu0)
    set_mgb_default_device(0)

    nori_name = '/unsullied/sharefs/xiongpengfei/Isilon-datashare/baseimg/landmark_dectest'
    os.system('rm -rf %s.nori'%(nori_name))
    nw = nori.open('%s.nori'%(nori_name), "w")
    outdata = {}

    nori_path = '/unsullied/sharefs/xiongpengfei/Isilon-datashare/baseimg/landmark_alltest.info'
    test_nori = pkl.load(open(nori_path, 'rb'))
    f = nori.Fetcher()
    tasks = ['1026-valid','1029-valid','1036-valid','334-valid','858-valid','909-valid','939-valid','validation-valid']
    tasks += ['2161-valid','1319-valid','2148-valid','2462-valid','2252-valid']
    tasks += ['hard-valid','front-valid','halfprofile-valid','profile-valid','up-valid','down-valid','glasses-valid','mouthhalf-valid', 'mouthopen-valid', 'mouthextreme-valid']

    for task in tasks:
        nori_id_list = test_nori[task]
        outdata[task] = []
        for nori_id in tqdm(nori_id_list):
            pack = pkl.loads(f.get(nori_id))
            img = cv2.imdecode(np.fromstring(pack['img'], np.uint8), cv2.IMREAD_COLOR)
            gt = pack['ld'].reshape(-1,2)
            x2,y2,x3,y3 = gt[:,0].min(),gt[:,1].min(),gt[:,0].max(),gt[:,1].max()
            ss = (x3-x2)*(y3-y2)*1.
            
            dets = detect(img)

            h,w = img.shape[0], img.shape[1]
            maxolp = -1
            maxrect, maxgt8, maxgt81 = [],[],[]
            for det in dets:
                rect = det['rect']
                x0,y0,x1,y1 = rect[0],rect[1],rect[0]+rect[2],rect[1]+rect[3]
                xx = min(x1,x3)-max(x0,x2)
                yy = min(y1,y3)-max(y0,y2)
                olp = xx*yy/ss
                if (xx>0)and(yy>0)and(olp>0.5)and(olp>maxolp):
                    maxolp = olp
                    maxrect = rect
                    maxgt8 = det['gt8']
                    maxgt81 = det['gt81']
            if maxrect==[]:
                #print(rects)
                continue
            #maxrect[0] = int(maxrect[0]-maxrect[2]*0.1)
            #maxrect[2] = int(maxrect[2]*1.2)
            #maxrect[1] = int(maxrect[1]-maxrect[3]*0.1)
            #maxrect[3] = int(maxrect[3]*1.2)
            outdata[task].append(nw.put(pkl.dumps({'img':pack['img'], 'bbox':maxrect, 'ld':gt, 'gt8':maxgt8, 'gt81':maxgt81}), filename=''))
        print(task, len(outdata[task]))

    s = sorted(outdata.keys())
    for i in s:
        print(i, len(outdata[i]))
    pkl.dump(outdata, open('%s.info'%(nori_name),'wb'))


