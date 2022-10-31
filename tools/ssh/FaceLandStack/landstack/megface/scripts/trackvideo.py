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
from detect import det

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

def transpkl(json_faces):
    faces = ()
    idx = 0
    for face in json_faces:
        frame_id = face['frame_id']
        dtboxes = face['dtboxes']
        #print(frame_id, len(dtboxes))

        rects, shapes, scores = [],[],[]
        if idx < frame_id-1:
            for i in (idx, frame_id):
                faces += (rects, shapes, scores),

        for i in range(len(dtboxes)):
            ld = dtboxes[i]
            ld = np.array(ld).reshape(-1,2)
            #print(ld.shape)
            shapes.append(ld)
        faces += (rects, shapes, scores),
        idx = frame_id
    return faces

def loadmodel(model, oprs):
    env = FpropEnv()
    net = io.load_network(os.path.join(args.model))
    print(net.outputs)
    opr_ld, opr_cls = args.opr.split(',')
    print(opr_ld, opr_cls)
    try:
         opr_ld = net.find_opr_by_name(opr_ld)
         opr_cls = net.find_opr_by_name(opr_cls)
    except:
        opr_ld = net.outputs_visitor.all_oprs_dict[opr_ld]
        opr_cls = net.outputs_visitor.all_oprs_dict[opr_cls]
    opr_ld = env.get_mgbvar(opr_ld)
    opr_cls = env.get_mgbvar(opr_cls)
    fprop = env.comp_graph.compile_outonly([opr_ld, opr_cls])
    return fprop

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--invideo', required=True, help='test video')
    parser.add_argument('-t', '--indet', required=True, help='test video det')
    parser.add_argument('-ot', '--outtrack', required=True)
    parser.add_argument('-oi', '--outframe', required=False, default='')
    parser.add_argument('-oj', '--outres', required=False, default='')
    parser.add_argument('-ov', '--outvideo', required=False, default='')
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-f', '--opr', required=True, default='s-pred,s-prob')
    parser.add_argument('-g', '--gpu', required=False, default='gpu0')
    parser.add_argument('-l', '--log', required=False, default='logtxt')
    parser.add_argument('-c', '--color', required=False, default='True')
    args = parser.parse_args()

    # load det result
    video_input = args.invideo
    det_input = args.indet
    print(video_input, det_input)
    faces = ()
    if det_input[-4:] == 'json':
        json_faces = json.load(open(det_input,'r'))
        faces = transpkl(json_faces)
    else:
        faces = pkl.load(open(det_input, 'rb'))
    print(len(faces))
    #print(faces[0])

    # load model
    iscolor = eval(args.color)
    mgb.config.set_default_device(args.gpu)
    fprop = loadmodel(args.model, args.opr)

    # begin track
    maxframe = 10000
    detthres = 0.8
    lmkthres = 0.3
    cap = cv2.VideoCapture(video_input)
    frame_idx = 0
    tracks = []
    while True:
        if frame_idx >= len(faces):
            break
        if frame_idx >= maxframe:
            break
        ret,frame = cap.read()
        if not ret:
            break

        # track
        for track in tracks:
            if track['is_end']:
                continue
            pld = track['ld'][-1]
            mat = alignto(pld, mean_shape, INPSIZE)
            img = cv2.warpAffine(frame, mat, (INPSIZE, INPSIZE), borderMode=cv2.BORDER_REPLICATE)
            if not iscolor:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape((INPSIZE,INPSIZE,1))
            img = img.transpose((2,0,1)).astype('float32')
            data_img = img[np.newaxis,:,:,:]
            pred, score = fprop(img=data_img)
            pred = pred[0] * INPSIZE
            invmat = inv_mat(mat)
            ld = ld_affine(pred, invmat).reshape(-1)
            track['ld'].append(ld)
            track['score'].append(score)
            if score<lmkthres:
                track['is_end'] = True

        # detect is appended when no track face overlapped
        for i in range(len(faces[frame_idx][1])):
            detface = np.array(faces[frame_idx][1][i]).reshape((-1,2))
            detscore = faces[frame_idx][2][i]
            if detscore<=detthres:
                continue
            x0,y0,x1,y1 = detface[:,0].min(),detface[:,1].min(),detface[:,0].max(),detface[:,1].max()
            flag = 1
            for track in tracks:
                if track['is_end']:
                    continue
                trackface = track['ld'][-1].reshape((-1,2))
                x2,y2,x3,y3 = trackface[:,0].min(), trackface[:,1].min(), trackface[:,0].max(), trackface[:,1].max()
                xx = min(x1,x3)-max(x0,x2)
                yy = min(y1,y3)-max(y0,y2)
                if (xx>0)and(yy>0)and(xx*yy>(x3-x1)*(y3-y2)*0.5):
                    flag = 0
                    break
            if flag:
                print('detect is append')
                trackid = len(tracks)
                tracks.append({'trackid':trackid,'ld':[detface.reshape(-1)],'score':[1.],'st_frame':frame_idx,'is_end':False})
        frame_idx += 1
    print("all faces %d"%(len(tracks)))

    # save
    cap = cv2.VideoCapture(video_input)
    issavevideo = (len(args.outvideo)>0)
    issaveframe = len(args.outframe)>0
    issaveres = (len(args.outres)>0)
    if issaveframe:
        outframedir = args.outframe
        if not os.path.exists(outframedir):
            os.mkdir(outframedir)
    if issaveres:
        outresdir='%s'%(args.outvideo[:-4])
        if not os.path.exists(outresdir):
            os.mkdir(outresdir)   
    if issavevideo:
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        fps = cap.get(cv2.CAP_PROP_FPS)
        outvideo = None
    frame_idx = 0
    outpacks = {}
    while True:
        if frame_idx >= len(faces):
            break
        if frame_idx >= maxframe:
            break
        ret,frame = cap.read()
        if not ret:
            break

        outpacks[frame_idx] = []
        for track in tracks:
            if (frame_idx >= track['st_frame']) and (frame_idx < track['st_frame'] + len(track['ld'])):
                idx = frame_idx - track['st_frame']
                outpacks[frame_idx].append({'trackid':track['trackid'], 'idx':idx, 'ld':track['ld'][idx], 'score':track['score'][idx]})

        if issaveframe:
            outpath = '%s/%d.jpg'%(outframedir, frame_idx)
            cv2.imwrite(outpath, frame)
            outpath = '%s/%d.txt'%(outframedir, frame_idx)
            ld = outpacks[frame_idx][0]['ld'].reshape(-1,2)
            f = open(outpath, 'w')
            for i in range(len(ld)):
                f.write('%f %f\n'%(ld[i,0], ld[i,1]))
            f.close()
        if issaveres or issavevideo:
            if issavevideo and outvideo is None:   
                shape = frame.shape
                #outpath = '%s_detect.avi'%(args.outvideo[:-4])
                outvideo = cv2.VideoWriter(args.outvideo, fourcc, int(fps), (shape[1], shape[0]))
            for pack in outpacks[frame_idx]:
                idx = pack['idx']     
                ld = pack['ld'].reshape(-1,2)
                trackid = pack['trackid']
                score = pack['score']
                color = (255,0,0)
                if idx == 0:
                    color = (0,255,255)
                x0,x1,y0,y1 = ld[:,0].min(), ld[:,1].min(), ld[:,0].max(), ld[:,1].max()
                circle_r = 1
                if (x1-x0)**2+(y1-y0)**2 > 200**2:
                    circle_r = 2
                if (x1-x0)**2+(y1-y0)**2 > 400**2:
                    circle_r = 3
                for j in range(len(ld)):
                    cv2.circle(frame, (int(ld[j,0]), int(ld[j,1])), circle_r, color, -1)
                cv2.putText(frame, '%d %.1f'%(trackid, score), (int(ld[0,0]),int(ld[0,1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            if issaveres:
                outpath = '%s/%d.jpg'%(outresdir, frame_idx)
                cv2.imwrite(outpath, frame)
            if issavevideo:
                outvideo.write(frame)

        frame_idx += 1
    pkl.dump(outpacks, open(args.outtrack, 'wb'))

