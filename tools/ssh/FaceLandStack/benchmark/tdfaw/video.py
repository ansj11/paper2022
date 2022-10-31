import time
from collections import OrderedDict

from benchmark.tester import TestUnitBase
from landstack.utils.visualization import draw
from landstack.utils import misc, augment, transforms
from landstack.train import env
import os
import numpy as np
import cv2
import tqdm

class TestUnit(TestUnitBase):
    def __init__(self, caches):
        name = "occ.video Tester ({}-{})".format(time.strftime("%x"), time.strftime("%X"))
        super(TestUnit, self).__init__(name, caches)
        self.n_points = 66

    def gen_kv_dict_all(self, **kwargs):
        if self._kv_dict_all is None:
            data_root = kwargs['data_root']
            video_suffix = ['.{}'.format(v) for v in kwargs['video_suffix'].split(',')] # add dot
            det_suffix = '.{}'.format(kwargs['det_suffix']) # add dot
            times = kwargs['times']
            n_video = kwargs['n_video']
            video_files, det_files, det_full_files = [], [], []
            self._kv_dict_all = OrderedDict()
            if any([data_root.endswith(v) for v in video_suffix]):
                video_files.append(data_root)
            else:
                for root, dirs, files in os.walk(data_root):
                    for fname in files:
                        full_path = os.path.join(root, fname)
                        if any([fname.endswith(v) for v in video_suffix]):
                            video_files.append(full_path)
                        elif fname.endswith(det_suffix):
                            det_files.append(fname)
                            det_full_files.append(full_path)
            for video_file in video_files:
                if len(self._kv_dict_all) >= n_video >= 0:
                    break
                raw_file = '.'.join(os.path.basename(video_file).split('.')[:-1])
                det_file = '{}{}'.format(raw_file, det_suffix)
                if det_file in det_files:
                    idx = det_files.index(det_file)
                    det = misc.load_pickle(det_full_files[idx])
                    det = det[0:len(det):times]
                    frames, fps = misc.load_videos(video_file, times)
                    assert len(frames) == len(det), '{} vs {}'.format(len(frames), len(det))
                    self._kv_dict_all.setdefault(video_file, [frames, det, fps])
                else:
                    print("WRN: Could not found det file of {}".format(video_file))
        else:
            print("Using cached data")
        return self._kv_dict_all

    def gen_inf_outputs_data(self, **kwargs):
        assert self._kv_dict_all, 'Please generate kv_dict_all first'
        devices = kwargs.get('devices', 'gpu0')
        thr = kwargs['thr']
        net = kwargs['net']
        mean_face = misc.load_pickle(kwargs['mean_face_file'])
        img_opr = net.outputs_visitor.all_oprs_dict['img']
        image_height, image_width= img_opr.partial_shape[2:]
        inf_func = env.compile_inf_func(net.outputs, devices=devices)
        self._inf_outputs_data = OrderedDict()
        image_shape = (image_height, image_width)
        aug = augment.AugmentBase(
            mean_face=mean_face, image_shape=image_shape,
            transforms_post=transforms.Compose([
                transforms.ConvertToGray(),
                transforms.HWC2CHW(),
                transforms.NormalizeLandmark(image_shape=image_shape),
            ])
        )
        for k, v in tqdm.tqdm(self._kv_dict_all.items()):
            frames, dets, fps = v
            largest_face_in_first_frame = dets[0][0]
            lm_raw = largest_face_in_first_frame['gt81']

            frames_out = []
            for frame in frames:
                img_out, _, param_out_dict = aug(frame, lm_raw, lm_raw.copy())
                mat_out = param_out_dict['mat']
                mat_inv = np.linalg.inv(mat_out)
                feed_dict = dict(
                    img = img_out[np.newaxis, :, :, :],
                )
                output_data = env.forward(inf_func, feed_dict)
                eyeoc_pred_sig = output_data[net.outputs[0].name][0]
                eyeoc_pred = (eyeoc_pred_sig >= thr).astype(np.float32)
                lm_rel = output_data[net.outputs[1].name][0].reshape(-1, 2) * np.float32([image_width, image_height])
                lm_raw = np.concatenate([lm_rel, np.ones((len(lm_rel), 1))], axis=1).dot(mat_inv.T)[:, :2]
                text = '{:.5f}-{}'.format(float(eyeoc_pred_sig), int(eyeoc_pred))
                img_copy = draw(frame, lm_raw, text=text, wait=False, show=False, point_size=1, font_scale=2)
                frames_out.append(img_copy)


            self._inf_outputs_data.setdefault(k, [frames_out, fps])
        return self._inf_outputs_data

    def compute_metric(self, **kwargs):
        assert self._kv_dict_all and self._inf_outputs_data, 'Please gen kv_dict_all and inf_outputs_data first'
        out_root = kwargs.get('out_root')
        misc.ensure_dir(out_root)
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        res_dict = OrderedDict()
        for k, v in self._inf_outputs_data.items():
            frames_out, fps = v
            height, width = frames_out[0].shape[:2]
            out_file = os.path.join(out_root, os.path.basename(k))
            out = cv2.VideoWriter(out_file, fourcc, int(fps), (width, height))
            print("Saving to {}".format(out_file))
            for frame in frames_out:
                out.write(frame)
            out.release()
            res_dict.setdefault(os.path.basename(k), OrderedDict([('#frames', len(frames_out))]))
        return res_dict


def main(model, devices, args_input, caches):
    """args_input is a list of rest arguments"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--lm_name', default='s-pred-eyeoc-sig,s-pred')
    parser.add_argument('--data_root', default='/unsullied/sharefs/_research_facelm/Isilon-datashare/baseimg/select_eyeoc')
    parser.add_argument('--thr', default=0.5, type=float)
    parser.add_argument('--out-root', default='/tmp/eyeoc.video')
    parser.add_argument('--video-suffix', default='mp4,avi')
    parser.add_argument('--det-suffix', default='det')
    parser.add_argument('--times', default=1, type=int)
    parser.add_argument('--n-video', default=-1, type=int)

    args, unknownargs = parser.parse_known_args(args_input)

    net = misc.load_network_and_extract(model, args.lm_name.split(','))

    # init TestUnit
    t = TestUnit(caches=caches)
    t.gen_kv_dict_all(
        data_root=args.data_root,
        video_suffix=args.video_suffix,
        det_suffix=args.det_suffix,
        times=args.times,
        n_video=args.n_video,
    )

    t.gen_inf_outputs_data(
        mean_face_file='/unsullied/sharefs/_research_facelm/Isilon-datashare/baseimg/extra/mean_face_81.pkl',
        net=net,
        devices=devices,
        thr=args.thr,
    )
    return t.compute_metric(
        out_root=args.out_root,
    )



