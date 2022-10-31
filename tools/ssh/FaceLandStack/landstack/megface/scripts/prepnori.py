#!/usr/bin/env python3

import json
import sys
import os
import argparse
import cv2
from tqdm import tqdm
import nori2 as nori
import pickle as pkl
import numpy as np
from shutil import copyfile
from neupeak.utils.misc import set_mgb_default_device
from detect.detxp import detect

def rotate_image(img, angle):
    h,w = img.shape[:2]
    if angle!=180:
        angle %= 360
        M_rotate = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        img_rotated = cv2.warpAffine(img, M_rotate, (w, h))
        return img_rotated
    else:
        res = []
        for i in range(h):
            res.append(img[h-i-1,:]) 
            res[i,:] = img[h-i-1,:]     
        res = np.array(res)
        return res
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--indir', required=True, help='test video')
    parser.add_argument('-or', '--outres', required=True, help='test video')
    parser.add_argument('-od', '--outdir', required=True, help='test video')
    args = parser.parse_args()
    set_mgb_default_device(0)

    facedir = os.path.join(args.outdir, 'face')
    nondir = os.path.join(args.outdir, 'none')
    if not os.path.exists(facedir):
        os.makedirs(facedir)
    if not os.path.exists(nondir):
        os.makedirs(nondir)

    #files = [ os.path.join(args.indir,f) for f in os.listdir(args.indir) if os.path.isfile(os.path.join(args.indir,f)) ]
    inlist = os.path.join(args.indir, 'path')
    files = open(inlist, 'r')
    facenum=0
    nonnum =0
    packs = []
    for path in files:
        path = os.path.join(args.indir, path.strip())
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h,w = img.shape[0], img.shape[1]
        dets = detect(img)
        bface = False
        for det in dets:
            rect = det['rect']
            gt81 = det['gt81']
            gt8 = det['gt8']
            #cy = math.fabs(rect[0]+rect[2]*0.5-h*0.5)
            #cx = math.fabs(rect[1]+rect[3]*0.5-w*0.5)
            #if cx>h*0.4 or rect[2]<w*0.2 or rect[3]<h*0.2:
            #    #print(rect)
            #    continue

            x0,y0,x1,y1 = gt8[:,0].min(),gt8[:,1].min(),gt8[:,0].max(),gt8[:,1].max()
            gw, gh = x1-x0, y1-y0
            x0 = int(max(x0-gw, 0))
            x1 = int(min(x1+gw, w))
            y0 = int(max(y0-gh, 0))
            y1 = int(min(y1+gh, h))
            if x1-x0<100 or y1-y0<100:
                continue
            outimg = img[y0:y1,x0:x1,:]
            gt8[:,0] -= x0
            gt8[:,1] -= y0
            gt81[:,0] -= x0
            gt81[:,1] -= y0

            outdir2 = '%s/%05d'%(facedir, facenum/1000)
            if not os.path.exists(outdir2):
                os.makedirs(outdir2)
            outfile = '%s/%05d.jpg'%(outdir2, facenum) 
            cv2.imwrite(outfile, outimg)
            
            pack = {}
            pack['path'] = outfile
            pack['gt8'] = gt8
            pack['gt81'] = gt81
            packs.append(pack)
            facenum += 1
            bface = True

        if not bface:
            outdir2 = '%s/%05d'%(nondir, nonnum/1000)
            if not os.path.exists(outdir2):
                os.makedirs(outdir2)
            outfile = '%s/%05d.jpg'%(outdir2, nonnum) 
            copyfile(path, outfile)
            nonnum += 1

    detout = open(args.outres, 'wb')
    pkl.dump(packs, detout)
    detout.close()


