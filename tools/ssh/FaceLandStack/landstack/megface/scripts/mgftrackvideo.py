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

if __name__ == '__main__':
    device = 'gpu0'
    mgf = MegFaceAPI(
        megface_lib_path='/home/xiongpengfei/megface-v2/lib/libmegface.so',
        face_model_root='/unsullied/sharefs/xiongpengfei/Isilon-alignmentModel/3rdparty/Model2/Model/',
        version='2.4',
        device=device,
    )
    colors = np.random.randint(0, 256, (10000, 3))

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--invideo', required=True, help='test video')
    parser.add_argument('-ot', '--outtrack', required=True)
    parser.add_argument('-ov', '--outvideo', required=True)
    parser.add_argument('-g', '--gpu', required=False, default='gpu0')
    #parser.add_argument('-cf', '--conf', required=False, default='tracker.mobile.v3.fast.conf')
    parser.add_argument('-cf', '--conf', required=False, default='tracker.mobile.v4.pose.conf')
    args = parser.parse_args()

    mgf.register_track_81_config(args.conf)

    # load det result
    video_input = args.invideo
    video_output = args.outvideo
    print(video_input, video_output)

    cv2.namedWindow('img', 0)

    # begin track
    maxframe = 10000
    cap = cv2.VideoCapture(video_input)
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    fps = cap.get(cv2.CAP_PROP_FPS)
    outvideo = None
    frame_idx = 0
    tracks = {}
    trackids = []
    while True:
        if frame_idx >= maxframe:
            break
        ret,frame = cap.read()
        if not ret:
            break
        tracks[frame_idx] = []
        if outvideo is None:
            shape = frame.shape
            outvideo = cv2.VideoWriter(video_output, fourcc, int(fps), (shape[1], shape[0]))

        faces = track(mgf, frame)
        tracks[frame_idx].append(faces)
        for face in faces['items']:
            rect = face['rect']
            fid = face['track_id']
            if fid in trackids:
                ths = 0.5
            else:
                ths = 0.9
                trackids.append(fid)
            conf = face['confidence']
            ld = face['landmark']
            print(conf, ths)
            if conf < ths:
                continue
            cv2.rectangle(frame, (rect['left'], rect['top']), (rect['right'], rect['bottom']), colors[int(fid)].tolist(), 3)
            cv2.putText(frame, str(fid), (rect['left'], rect['top']), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            for pt in ld['points']:
                cv2.circle(frame, (int(pt['x']), int(pt['y'])), 0, (255, 255, 128), -1)

        cv2.imshow("img", frame)
        cv2.waitKey()
        outvideo.write(frame)
        frame_idx += 1

    pkl.dump(tracks, open(args.outtrack, 'wb'))

