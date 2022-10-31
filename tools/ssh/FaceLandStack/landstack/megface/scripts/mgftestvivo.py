#!/usr/bin/env python3
from meghair.utils import io
import megbrain as mgb
import megskull
import megskull.opr.all as O
from megskull.graph import FpropEnv
from tqdm import tqdm
import nori2 as nori

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
sys.path.append('../')
from landstack.megface.detectcore import loadyuv, detect, MegFaceAPI

def detectimgrot(mgf, img, dtype):
    outpacks = {}
    rotlist = [0, 90, 180, 270]
    for rot in rotlist:
        rot = int(rot)
        maxrot = 'rot%d'%(rot%360)
        outpacks[maxrot] = None
   
        imgrot = misc.rotatecv(img, maxrot)
        #outpath = 'testrot_%d.jpg'%(rot)
        #cv2.imwrite(outpath, imgrot) 

        faces = detect(mgf, imgrot, stage=dtype, thres=0.2, maxface=False, gt=None, options={'orient':mgfpy.Orient.MGF_UP})
        maxscore = -1
        maxface = None
        for face in faces:
            score = face['conf']
            if maxscore < score:
                maxscore = score
                maxface = face
            if maxscore >= 0.7:
                break
        if maxscore > 0:
            gt81 = maxface['pts']
            gt81 = misc.rotateld(gt81, maxrot, imgrot.shape[0], imgrot.shape[1])
            rect = misc.rotateld(maxface['rect'], maxrot, imgrot.shape[0], imgrot.shape[1]) if 'rect' in maxface else []
            outpacks[maxrot] = {'score':maxscore, 'gt81':gt81, 'rect':rect}
    return outpacks

def detectimgmeg(mgf, img, dtype):
    outpacks = {}
    rotlist = [0, 90, 180, 270]
    for rot in rotlist:
        rot = int(rot)
        maxrot = 'rot%d'%(rot%360)
        outpacks[maxrot] = None

        megfacetype = mgfpy.Orient.MGF_UP
        if rot == 0:
            megfacetype = mgfpy.Orient.MGF_UP
        elif rot == 90:
            megfacetype = mgfpy.Orient.MGF_RIGHT
        elif rot == 180:
            megfacetype = mgfpy.Orient.MGF_DOWN
        elif rot == 270:
            megfacetype = mgfpy.Orient.MGF_LEFT
        print(megfacetype)

        imgrot = img.copy()

        faces = detect(mgf, imgrot, stage=dtype, thres=0.2, maxface=False, gt=None, options={'orient':megfacetype})
        maxscore = -1
        maxface = None
        for face in faces:
            score = face['conf']
            if maxscore < score:
                maxscore = score
                maxface = face
            if maxscore >= 0.7:
                break
        if maxscore > 0:
            gt81 = maxface['pts']
            rect = maxface['rect'] if 'rect' in maxface else []
            outpacks[maxrot] = {'score':maxscore, 'gt81':gt81, 'rect':rect}
    return outpacks
   
##########################################################################
def saveresult(packs, img, dtype):
    for rot in packs:
        imgrot = img.copy()
        if packs[rot] is None:
            outpath = '%s_%s_none.jpg'%(dtype, rot)
            cv2.imwrite(outpath, imgrot)
        else:
            gt81 = packs[rot]['gt81']
            rect = packs[rot]['rect']
            score = packs[rot]['score']
            for i in range(len(gt81)):
                cv2.circle(imgrot, (int(gt81[i,0]), int(gt81[i,1])), 3, (255,0,0), 3)
                if rect != []:
                    cv2.rectangle(imgrot, (int(rect[0,0]),int(rect[0,1])), (int(rect[1,0]), int(rect[1,1])), (0,0,255), 3)
                    cv2.putText(imgrot, '00', (int(rect[0,0]), int(rect[1,1])), 0, 0.8, (0,0,255), 3)
            outpath = '%s_%s_%f.jpg'%(dtype, rot, score)
            cv2.imwrite(outpath, imgrot)

def testimg():
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

    imgpath = '/unsullied/sharefs/_research_facelm/Isilon-datashare/tmp/vivo/nv21/tmp/beijing1_ARN_1.jpg_720_1280'
    img = loadyuv(imgpath)

    res0 = detectimgrot(mgf, img, 'lm_81')
    res1 = detectimgmeg(mgf, img, 'det_81')
    saveresult(res0, img, 'rotout')
    saveresult(res1, img, 'align')


##########################################################################
def testall():
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
    
    outdir = '171007_augbig'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    #imgdir = '/unsullied/sharefs/_research_facelm/Isilon-datashare/tmp/vivo/nv21/'
    imgdir = '/unsullied/sharefs/_research_facelm/Isilon-datashare/tmp/vivo/compare/'
    imglist = os.listdir(imgdir)
    for path in imglist:
        imgpath = '%s/%s'%(imgdir, path)
        #if imgpath.find('.jpg')<0:
        #    continue
        img = loadyuv(imgpath)
        packs = detectimgmeg(mgf, img, 'lm_81')

        if packs['rot0'] is None:
            err, err90, err180, err270 = 1,1,1,1
        else:
            err90 = 1 if packs['rot90'] is None else ((((packs['rot0']['gt81']-packs['rot90']['gt81'])**2).sum(axis=1))**0.5).mean()
            err180 = 1 if packs['rot180'] is None else ((((packs['rot0']['gt81']-packs['rot180']['gt81'])**2).sum(axis=1))**0.5).mean()
            err270 = 1 if packs['rot270'] is None else ((((packs['rot0']['gt81']-packs['rot270']['gt81'])**2).sum(axis=1))**0.5).mean()
            err = max(err90, err180, err270)

        imgout = img.copy()
        for rot in packs:
            if rot == 'rot0':
                c = (0,0,0); loc = (30,30)
            elif rot == 'rot90':
                c = (255,0,0); loc = (30,60)
            elif rot == 'rot180':
                c = (0,255,0); loc = (30,90)
            elif rot == 'rot270':
                c = (0,0,255); loc = (30,120)
            if packs[rot] is None:
                continue
            gt81 = packs[rot]['gt81']
            if outdir.find('65')>0:
                for i in range(19,81,1):
                    cv2.circle(imgout, (int(gt81[i,0]), int(gt81[i,1])), 3, c, 3)
            elif outdir.find('16')>0:
                index = [28, 32, 73, 77, 20, 24, 65, 69, 61, 63, 62, 36, 45, 53, 37]
                for i in index:
                    cv2.circle(imgout, (int(gt81[i,0]), int(gt81[i,1])), 3, c, 3)
            else:
                for i in range(len(gt81)):
                    cv2.circle(imgout, (int(gt81[i,0]), int(gt81[i,1])), 3, c, 3)
            cv2.putText(imgout, '%s %.2f'%(rot, packs[rot]['score']), loc, 0, 1.0, c, 3)
        outpath = '%s/%f_%f_%f_%f_%s.jpg'%(outdir, err, err90, err180, err270, path)
        cv2.imwrite(outpath, imgout)

if __name__ == '__main__':
    testimg()
    #testall()



