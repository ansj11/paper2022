import time
from collections import OrderedDict

from benchmark.tester import TestUnitBase
from landstack.utils.visualization import draw
from landstack.utils import misc, augment, transforms, geom
from landstack.train import env
import os
import numpy as np
import cv2
import tqdm
from IPython import embed

class TestUnit(TestUnitBase):
    def __init__(self, caches):
        name = "Gaze Video Tester ({}-{})".format(time.strftime("%x"), time.strftime("%X"))
        super(TestUnit, self).__init__(name, caches)

    def gen_kv_dict_all(self, data_root, video_names):
        if self._kv_dict_all is None:
            self._kv_dict_all = OrderedDict()
            for video_name in video_names.split(','):
                video_file = os.path.join(data_root, video_name+'.mp4')
                det_file = os.path.join(data_root, video_name+'.det')
                assert os.path.exists(det_file), '{} not exist'.format(det_file)
                video_data, fps = misc.load_videos(video_file)
                det_data = misc.load_pickle(det_file)
                assert len(video_data) == len(det_data)
                self._kv_dict_all[video_name] = dict(
                    video_data = video_data,
                    det_data = det_data,
                    fps = fps
                )
        else:
            print("Using cached data")
        return self._kv_dict_all

    def gen_inf_outputs_data(self, net, devices, lm_name, out_root):
        assert self._kv_dict_all, 'Please generate kv_dict_all first'
        mean_face = misc.load_pickle('/unsullied/sharefs/_research_facelm/Isilon-datashare/baseimg/extra/mean_face_81.pkl')
        width_height = np.float32([48, 32])
        misc.ensure_dir(out_root, erase=True)
        aug = augment.AugmentBase(
            mean_face=mean_face, image_shape=(112, 112),
            transforms_post=transforms.Compose([
                transforms.ConvertToGray(),
                transforms.HWC2CHW(),
            ])
        )
        lm_name, ec_name, gv_name, iris_name, eye_lm_name, mat_name = lm_name.split(',')
        inf_func = env.compile_inf_func(net.outputs)
        res_dict = OrderedDict()
        for k, value in self._kv_dict_all.items():
            frames = value['video_data']
            dets = value['det_data']
            fps = value['fps']

            frames_out = []
            lm_pred = dets[0][1]
            for frame, det in tqdm.tqdm(zip(frames, dets), total=len(frames)):
                img_out, _, param = aug(frame, lm_pred, lm_pred.copy())
                mat0 = np.linalg.inv(param['mat'])
                outputs_data = env.forward(inf_func, {'img':np.uint8(img_out).reshape(1, 1, 112, 112)})
                lm_new = outputs_data[lm_name]
                lm_pred = geom.lm_affine_general(112 * lm_new, mat0)
                mat_left, mat_right = outputs_data[mat_name]

                # eye center
                ec_left, ec_right = outputs_data[ec_name]
                ec_left[0] = 1 - ec_left[0]
                ec_left_raw = geom.lm_affine_general(geom.lm_affine_general(ec_left*width_height, mat_left), mat0)
                ec_right_raw = geom.lm_affine_general(geom.lm_affine_general(ec_right*width_height, mat_right), mat0)
                ec_raw = np.concatenate([ec_left_raw, ec_right_raw]).reshape(-1, 2)

                # gaze vector
                gv_left, gv_right = outputs_data[gv_name]
                gv_left[0] = -gv_left[0]

                s = np.sqrt((np.linalg.inv(mat_left)[0, :2] ** 2).sum())
                gp_left = ec_left*width_height + s * gv_left[:2] * 10
                gp_left_raw = geom.lm_affine_general(geom.lm_affine_general(gp_left, mat_left), mat0)
                s = np.sqrt((np.linalg.inv(mat_right)[0, :2] ** 2).sum())
                gp_right = ec_right*width_height + s * gv_right[:2] * 10
                gp_right_raw = geom.lm_affine_general(geom.lm_affine_general(gp_right, mat_right), mat0)
                gp_raw = np.concatenate([gp_left_raw, gp_right_raw]).reshape(-1, 2)

                # iris
                iris_left = outputs_data[iris_name][0].reshape(-1, 2)
                iris_left[:, 0] = 1 - iris_left[:, 0]
                iris_right = outputs_data[iris_name][1].reshape(-1, 2)
                iris_left_raw = geom.lm_affine_general(geom.lm_affine_general(iris_left*width_height, mat_left), mat0)
                iris_right_raw = geom.lm_affine_general(geom.lm_affine_general(iris_right*width_height, mat_right), mat0)
                iris_raw = np.concatenate([iris_left_raw, iris_right_raw]).reshape(-1, 2)

                # eye lm
                eye_lm_left = outputs_data[eye_lm_name][0].reshape(-1, 2)
                eye_lm_left[:, 0] = 1- eye_lm_left[:, 0]
                eye_lm_right = outputs_data[eye_lm_name][1].reshape(-1, 2)
                eye_lm_left_raw = geom.lm_affine_general(geom.lm_affine_general(eye_lm_left*width_height, mat_left), mat0)
                eye_lm_right_raw = geom.lm_affine_general(geom.lm_affine_general(eye_lm_right*width_height, mat_right), mat0)
                eye_lm_raw = np.concatenate([eye_lm_left_raw, eye_lm_right_raw]).reshape(-1, 2)

                img_copy = draw(frame, ec_raw, gp_raw, iris_raw, eye_lm_raw, show=False, wait=False, line=True)
                frames_out.append(img_copy)
            save_file = os.path.join(out_root, k+'.avi')
            misc.merge_to_video(save_file, frames_out, fps=fps)

            res_dict0 = OrderedDict()
            res_dict0['#frame'] = len(frames)
            res_dict0['fps'] = fps
            res_dict[k] = res_dict0
        return res_dict


def main(model, devices, args_input, caches):
    """args_input is a list of rest arguments"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--lm_name', default='pred,s-pred-eye-center,s-pred-gaze-vector,s-pred-iris,s-pred-eye-lm,s-pred-mat')
    parser.add_argument('--data_root', default='/unsullied/sharefs/_research_facelm/Isilon-datashare/testvideos/gaze')
    parser.add_argument('--video_names', default='gaze,pose')
    parser.add_argument('--out_root', default='/tmp/gaze.video')

    args, unknownargs = parser.parse_known_args(args_input)

    assert len(args.lm_name.split(',')) == 6, args.lm_name
    net = misc.load_network_and_extract(model, args.lm_name)

    t = TestUnit(caches=caches)
    t.gen_kv_dict_all(
        data_root=args.data_root,
        video_names=args.video_names,
    )

    return t.gen_inf_outputs_data(
        net=net,
        devices=devices,
        lm_name=args.lm_name,
        out_root=args.out_root,
    )



