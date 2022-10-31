#!/usr/bin/env python3

import megbrain
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
from neupeak.utils.misc import set_mgb_default_device
from detect.det import mdetect

def rotate_image(img, angle):
    h,w = img.shape[:2]
    #if angle!=180:
    #    angle %= 360
    #    M_rotate = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    #    img_rotated = cv2.warpAffine(img, M_rotate, (w, h))
    #    return img_rotated
    if angle == 90:
        res = img.copy()
        res = res.reshape(w,h,img.shape[2])
        for y in range(w):
            for x in range(h):
                res[y,x,:] = img[h-x-1,y,:]
        return res
    elif angle == 270:
        #img_rotated = np.zeros((w, h, img.shape[2]), 'uint8')
        res = img.copy()
        res = res.reshape(w,h,img.shape[2])
        for y in range(w):
            for x in range(h):
                res[y,x,:] = img[x,w-y-1,:]
        return res
    else:
        res = []
        for i in range(h):
            res.append(img[h-i-1,:])
            #res[i,:] = img[h-i-1,:]
        res = np.array(res)
        return res
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', required=True, help='test video')
    parser.add_argument('-do', '--result', required=True, help='test video')
    parser.add_argument('-vo', '--videoout', required=True, help='test video')
    parser.add_argument('-f', '--flip', required=True, help='test video')
    parser.add_argument('-g', '--gpu', required=True, help='test video')
    args = parser.parse_args()
    #set_mgb_default_device((args.gpu))
    #megbrain.config.set_default_device(args.gpu)

    video_name = args.video
    outdet_name = args.result
    outvideo_name = args.videoout
    flip = int(args.flip)
    tmp = os.path.abspath(os.path.join(outvideo_name, os.pardir))
    if not os.path.exists(tmp):
        os.mkdir(tmp)
    print(video_name, outvideo_name, outdet_name, tmp)

    mdet = mdetect(args.gpu, 'm')

    faces = ()
    cap = cv2.VideoCapture(video_name)
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    #fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    fps = cap.get(cv2.CAP_PROP_FPS)
    videoout = None
    MAXFRAME = 1000

    bsaveimage = False
    outimg_dir = outvideo_name
    outimg_dir = '%s_frame'%(outimg_dir[:-4])
    if not os.path.exists(outimg_dir):
        os.mkdir(outimg_dir)

    frame_idx = 0
    while True:
        ret,frame = cap.read()
        if not ret:
            break
        #print("curr faces %d"%(len(tracks)))
        if flip>0 and frame.shape[1] > frame.shape[0]:
            #frame = rotate_image(frame, 90)
            frame = np.transpose(frame, (1,0,2))
            #frame = rotate_image(frame, 180)
        print(frame.shape)
        #if frame_idx==0:
        #    cv2.namedWindow("Image")
        #    cv2.imshow("Image", frame)
        #    cv2.waitKey(0)
        if videoout is None:
            shape = frame.shape
            videoout = cv2.VideoWriter(outvideo_name, fourcc, int(fps), (shape[1], shape[0]))
        videoout.write(frame)
        if bsaveimage:
            out_path = '%s/%d.jpg'%(outimg_dir, frame_idx)
            cv2.imwrite(out_path, frame)

        rects, shapes, scores = [],[],[]
        dets = mdet.detect(frame)
    
        if dets == []:
            faces += (rects, shapes, scores), 
            continue
        for det in dets:
            rect = det['rect']
            gt81 = det['gt81']
            score = det['score']
            #cy = math.fabs(rect[0]+rect[2]*0.5-frame.shape[0]*0.5)
            #cx = math.fabs(rect[1]+rect[3]*0.5-frame.shape[1]*0.5)
            #if cx>frame.shape[1]*0.4 or rect[2]<frame.shape[0]*0.2 or rect[3]<frame.shape[1]*0.2:
                #print(rect)
                #continue
            rects.append(rect)
            shapes.append(gt81.reshape(-1,2))
            scores.append(score)
        faces += (rects, shapes, scores),
        print('%d %d'%(len(faces), frame_idx))
        frame_idx += 1
        if frame_idx > MAXFRAME:
            break
    videoout.release()

    detout = open(outdet_name, 'wb')
    pkl.dump(faces, detout, 0)
    detout.close()

