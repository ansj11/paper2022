#!/usr/bin/env python3

import json
import sys
import megbrain as mgb

import math
import argparse
import cv2
import os
from tqdm import tqdm
import nori2 as nori
import pickle as pkl
import numpy as np
from megskull.graph import FpropEnv
from meghair.utils import visualize, io
from neupeak.utils.misc import set_mgb_default_device
from neupeak.utils.cli import load_network
from neupeak.utils import inference as inf

INPSIZE = 112
MODELDIR = '/unsullied/sharefs/_research_facelm/Isilon-modelshare/model_zoo/others/testlib/detect/models/'

class mdetect(object):
    def __init__(self, gpu, mtype):
        mgb.config.set_default_device(gpu)
        self.mat_align8to81 = pkl.load(open('%s/mat8to813'%(MODELDIR),'rb'))
        self.mtype = mtype
        if mtype=='s':
            self.det_s12 = load_det_s12('%s/ds1.3'%(MODELDIR))
            self.ldt_p10_8 = load_ldt_p10_8('%s/ps1.2'%(MODELDIR))
        elif mtype=='m':
            self.det_s12 = load_det_s12('%s/dm1.3'%(MODELDIR))
            self.ldt_p10_8 = load_ldt_p10_8('%s/ps1.2'%(MODELDIR))
        elif mtype=='x':
            self.det_s12 = load_det_s12('%s/dx1.3'%(MODELDIR))
            self.ldt_p10_8 = load_ldt_p10_8('%s/px1.2'%(MODELDIR))

    def rect81(self, img, rect):
        bbx, bby, bbw, bbh = rect[0], rect[1], rect[2]-rect[0], rect[3]-rect[1]

        bby -= bbh * 0.1
        bby = int(bby)
        bbh *= 1.1
        bbh = int(bbh)
        bbx = max(1, bbx)
        bby = max(1, bby)
        bbw = min(img.shape[1]-1-bbx, bbw)
        bbh = min(img.shape[0]-1-bby, bbh)
        rectnew = [bbx, bby, bbw, bbh, 1.0]

        is_color = True if self.mtype=='x' else False
        score, gt8 = align_p10_8(self.ldt_p10_8, img, rectnew, is_color)
        gt81 = np.array(np.dot(gt8.reshape(1,-1), self.mat_align8to81)).reshape(-1,2)
        pack = {}
        pack['score'] = score
        pack['gt8'] = gt8
        pack['gt81'] = gt81
        return pack

    def detect(self, img):
        rects = detect_s12(self.det_s12, img)
        packs = []
        for rect in rects:
            score, gt8 = align_p10_8(self.ldt_p10_8, img, rect)
            gt81 = np.array(np.dot(gt8.reshape(1,-1), self.mat_align8to81)).reshape(-1,2)
            pack = {'rect':rect, 'gt8':gt8, 'gt81':gt81, 'score':score}
            packs.append(pack)
        return packs

def load_det_s12(model_path):
    network = load_network(model_path)
    pred_func_nchw = inf.Function(inf.get_fprop_env(fast_run=False)).compile(network.outputs)
    detector = inf.detection.DenseDetector(pred_func_nchw)
    return detector

def load_ldt_p10_8(model_path):
    env = FpropEnv()
    net = io.load_network(model_path)
    print(net.outputs_names)
    _ = []
    for i in ['cls_softmax', 'bb_brainfc_bbout']:
        fc = net.find_opr_by_name(i)
        _.append(env.get_mgbvar(fc))
    fprop = env.comp_graph.compile_outonly(_)
    return fprop

def detect_s12(detector, img, detetc_thres=0.96):
    gray = img
    rst = detector.predict(gray, detetc_thres)
    outrects = []
    for bb,s in zip(rst['objects'],rst['scores']):
        bb.y -= bb.h * 0.1
        bb.y = int(bb.y)
        bb.h *= 1.1
        bb.h = int(bb.h)
        bb.x = max(1, bb.x)
        bb.y = max(1, bb.y)
        bb.w = min(img.shape[1]-1-bb.x, bb.w)
        bb.h = min(img.shape[0]-1-bb.y, bb.h)
        if bb.w > 0 and bb.h > 0 and  bb.x > 0 and bb.y > 0 and bb.x+bb.w < img.shape[1] and bb.y+bb.h < img.shape[0]:
            outrects.append([bb.x, bb.y, bb.w, bb.h, s])
    return outrects

def align_p10_8(aligner, img, rect, is_color=False):
    [x,y,w,h,s]=rect
    img_crop = img[max(y,0):min(y+h,img.shape[0]),max(x,0):min(x+w,img.shape[1])]
    img_crop = cv2.resize(img_crop, (INPSIZE, INPSIZE))
    if is_color is False:
        img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY).reshape(INPSIZE,INPSIZE,1)
    img_crop = img_crop.transpose((2,0,1)).astype('float32')
    crops = []
    crops.append(img_crop)

    pred = aligner(data=crops)
    score = pred[0][0][1]
    pred = pred[1][0]
    preds = []
    for l in range(8):
        ptx = pred[2*l]*w/float(INPSIZE)+x
        pty = pred[2*l+1]*h/float(INPSIZE)+y
        preds.append((ptx,pty))
    preds = np.array(preds).reshape(-1, 2)
    return score, preds


if __name__ == '__main__':
    det = mdetect('gpu0', 'm')
    img_file = '/tmp/xxx.jpg'
    img = cv2.imread(img_file)
    res = det.detect(img)

