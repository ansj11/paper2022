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
from landstack.utils import misc

INPSIZE = 112
mean_shape = pkl.load(open('/unsullied/sharefs/_research_facelm/Isilon-datashare/baseimg/extra/mean_face_81.pkl', 'rb'))

def rot_scale_align(src, dst):
    src_x, src_y = src[:, 0], src[:, 1]
    dst_x, dst_y = dst[:, 0], dst[:, 1]
    d = (src**2).sum()
    a = sum(src_x*dst_x + src_y*dst_y) / d
    b = sum(src_x*dst_y - src_y*dst_x) / d
    mat = np.array([[a, -b], [b, a]])
    return mat

def alignto(lm, mean_face, output_size):
    lm = np.array(lm).reshape(-1, 2)
    mean = lm.mean(axis=0)
    mat1 = rot_scale_align(lm - mean, mean_face).T
    mat1 *= output_size
    mat2 = np.float64([[mat1[0][0], mat1[1][0], -mean[0] * mat1[0][0] - mean[1] * mat1[1][0] + output_size // 2],
                       [mat1[0][1], mat1[1][1], -mean[0] * mat1[0][1] - mean[1] * mat1[1][1] + output_size // 2]])
    return mat2

def ld_affine(lm, mat):
    lm = np.array(lm).reshape(-1, 2)
    return np.dot(np.concatenate([lm, np.ones((lm.shape[0], 1))], axis=1), mat.T)

def inv_mat(mat):
    return np.linalg.inv(mat.tolist() + [[0, 0, 1]])[:2]

def loadmodel(modelpath, oprs):
    env = FpropEnv(verbose_fprop=False)
    net = io.load_network(modelpath)
    print(net.outputs)
    l = oprs.split(',')
    if len(l)==2:
        opr_ld, opr_cls = l[0],l[1]
        try:
            opr_ld = net.find_opr_by_name(opr_ld)
            opr_cls = net.find_opr_by_name(opr_cls)
        except:
            opr_ld = net.outputs_visitor.all_oprs_dict[opr_ld]
            opr_cls = net.outputs_visitor.all_oprs_dict[opr_cls]
        opr_ld = env.get_mgbvar(opr_ld)
        opr_cls = env.get_mgbvar(opr_cls)
        fprop = env.comp_graph.compile_outonly([opr_ld, opr_cls])
    else:
        opr_ld = l[0]
        try:
            opr_ld = net.find_opr_by_name(opr_ld)
        except:
            opr_ld = net.outputs_visitor.all_oprs_dict[opr_ld]
        opr_ld = env.get_mgbvar(opr_ld)
        fprop = env.comp_graph.compile_outonly([opr_ld])
    return fprop

def getlm(imgt, ld, fprop, iscolor=False):
    mat = alignto(ld, mean_shape, INPSIZE)
    img = cv2.warpAffine(imgt, mat, (INPSIZE, INPSIZE), borderMode=cv2.BORDER_REPLICATE)
    if iscolor is False:
        img = misc.ensure_hwc(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape((INPSIZE,INPSIZE,1))
    img = img.transpose((2,0,1)).astype('float32')
    data_img = img[np.newaxis,:,:,:]
    pred = fprop(img=data_img)
    pred = pred[0] * INPSIZE
    invmat = inv_mat(mat)
    ld = ld_affine(pred, invmat).reshape(-1)
    return ld

def getattr(imgt, ld, fprop, iscolor=False):
    mat = alignto(ld, mean_shape, INPSIZE)
    img = cv2.warpAffine(imgt, mat, (INPSIZE, INPSIZE), borderMode=cv2.BORDER_REPLICATE)
    if iscolor is False:
        img = misc.ensure_hwc(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape((INPSIZE,INPSIZE,1))
    img = img.transpose((2,0,1)).astype('float32')
    data_img = img[np.newaxis,:,:,:]
    pred = fprop(img=data_img)
    return pred[0]




