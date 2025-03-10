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
from landstack.megface.lmkcore import loadmodel, getlm
from landstack.megface.detectcore import detect, MegFaceAPI

INPSIZE = 112
nori_path = '/unsullied/sharefs/_research_facelm/Isilon-datashare/baseimg/landmark_test0831.info'
nori_path0930 = '/unsullied/sharefs/_research_facelm/Isilon-datashare/baseimg/landmark_test0930.info'
nori_path1030 = '/unsullied/sharefs/_research_facelm/Isilon-datashare/baseimg/landmark_test1030.info'
# test_nori = pkl.load(open(nori_path, 'rb'))
f = nori.Fetcher()
#test_set = ['validation-valid','combine-valid','mobile-valid','security-valid']
#test_setall = test_set + ['down-valid','up-valid','profile-valid','blackglasses-valid','sunglasses-valid','blur-valid','illumination-valid','moustache-valid','occlusion-valid','mask-valid','hair-valid','extremeexpre-valid','bigmouth-valid','black-valid']
test_setall = ['validation-valid']
ld_comps = ['contour', 'eye', 'eyebrow', 'mouth', 'nose', 'organs', 'keypoints']

def split(label):
    label = label.reshape(-1,162)
    comps = dict()
    comps['contour'] = label[:, :19*2]
    comps['eye'] = np.concatenate([label[:, 19*2:28*2], label[:, 64*2:73*2]], axis=1)
    comps['eyebrow'] = np.concatenate([label[:, 28*2:36*2], label[:, 73*2:]], axis=1)
    comps['mouth'] = label[:, 36*2:54*2]
    comps['nose'] = label[:, 54*2:64*2]
    comps['organs'] = label[:, 19*2:]
    index = [28,32,73,77,20,24,23,65,69,68,61,63,62,36,45,53,37]
    comps['keypoints'] = label.reshape(len(label), -1, 2)[:, index, :].reshape(len(label), -1)
    return comps

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ldname', default='ld')
    parser.add_argument('--modelpath', required=True)
    parser.add_argument('--oprnames', required=True)
    parser.add_argument('--dataset', default="170831")
    parser.add_argument('--keys', default="validation-valid")
    parser.add_argument('--iscolor', default=False)
    args = parser.parse_args()
    test_setall = args.keys.split(',')
    nori_path = '/unsullied/sharefs/_research_facelm/Isilon-datashare/baseimg/landmark_test{}.info'.format(args.dataset)
    if not os.path.exists(nori_path):
        print("[ERROR]{} is not exists!".format(nori_path))
        os._exit(-1)
    test_nori = pkl.load(open(nori_path, 'rb'))

    fprop = loadmodel(args.modelpath, args.oprnames)

    if args.ldname.find('megface')>=0:
        device = 'gpu0'
        mgf = MegFaceAPI(
            # /home/xiongpengfei/megface-v2/lib/libmegface.so
            # megface_lib_path='/unsullied/sharefs/weixin/wxdataset/megface-v2/lib/libmegface.so',
            megface_lib_path='/unsullied/sharefs/chenxi/isilon-share/shares/public/megface-v2/lib/libmegface.so',
            face_model_root='/unsullied/sharefs/xiongpengfei/Isilon-alignmentModel/3rdparty/Model24/FaceModel/',
            version='2.5',
            device=device,
        )
        if args.ldname == 'megface_middle':
            mgf.register_det_rect_config('detector_rect.densebox.middle.v1.3.conf')
            mgf.register_det_81_config('lmk.postfilter.small.v1.2.conf')
        elif args.ldname == 'megface.mobile':
            mgf.register_det_rect_config('detector_rect.densebox.small.v1.3.conf')
            mgf.register_det_81_config('lmk.postfilter.small.v1.2.conf')
        elif args.ldname == 'megface.xlarge':
            mgf.register_det_rect_config('detector_rect.densebox.xlarge.v1.3.conf')
            mgf.register_det_81_config('lmk.postfilter.xlarge.v1.2.conf')

    # load allsample
    print('begin')
    tasks_seperator, data_gt, data_res, data_norm = [], [], [], []
    for task in test_setall:
        nori_id_list = test_nori[task]
        st = len(data_gt)
        for nori_id in tqdm(nori_id_list):
            pack = pkl.loads(f.get(nori_id))
            img = pack['img']
            gt = pack['ld'].reshape((-1,2))
            img = cv2.imdecode(np.fromstring(img, np.uint8), cv2.IMREAD_COLOR)
            res = None
            if args.ldname.find('megface')>=0:
                init = detect(mgf, img, stage='det_81', thres=0.9, gt=gt)
                if init != None:
                    res = getlm(img, init['pts'], fprop, iscolor=args.iscolor)
            else:
                init = pack[args.ldname].reshape(-1,2)
                res = getlm(img, init, fprop, iscolor=args.iscolor)
            if res is not None:
                data_res.append(res.reshape(-1))
                data_gt.append(gt.reshape(-1))
                data_norm.append(np.sum((gt[23]-gt[68])**2)**0.5)
        ed = len(data_gt)
        tasks_seperator.append((st,ed))
    data_res, data_gt, data_norm = map(np.array, [data_res, data_gt, data_norm])

    #fp = open(args.logtxt, 'w')

    # get err
    #fp.write('task:\t\terr(contour,eye,eyebrow,nose,mouth)\n')
    for tid, task in enumerate(test_setall):
        st, ed = tasks_seperator[tid]
        pred_t = data_res[st:ed]
        gt_t = data_gt[st:ed]
        norm_t = data_norm[st:ed]
        print(st,ed,pred_t.shape,gt_t.shape,norm_t.shape)
        n = len(pred_t)
        err = np.mean(np.sum(((pred_t-gt_t)**2).reshape((n,-1,2)), axis=2)**0.5/norm_t.reshape((-1,1)))

        pred_t_comps = split(pred_t)
        gt_t_comps = split(gt_t)
        err_comps = {}
        for comp in sorted(gt_t_comps.keys()):
            err_comps[comp] = np.mean(np.sum(((pred_t_comps[comp]-gt_t_comps[comp])**2).reshape((n,-1,2)), axis=2)**0.5/norm_t.reshape((-1,1)))
        print("%s:\t%.5f(%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f)\n"%(task,err,err_comps['contour'],err_comps['eye'],err_comps['eyebrow'],err_comps['nose'],err_comps['mouth'],err_comps['organs'],err_comps['keypoints']))
        #fp.write("%s:\t%.5f(%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f)\n"%(task,err,err_comps['contour'],err_comps['eye'],err_comps['eyebrow'],err_comps['nose'],err_comps['mouth'],err_comps['organs'],err_comps['keypoints']))

    #fp.close()

