import time
from collections import OrderedDict

from megskull.network.raw import RawNetworkBuilder

from benchmark.dataset import static
from benchmark.tester import TestUnitBase
from landstack.utils.visualization import draw
from landstack.utils import misc, geom
from landstack.train import env
from IPython import embed
import os
import numpy as np
import cv2
import tqdm

class TestUnit(TestUnitBase):
    def __init__(self, caches):
        name = "track.stn.video Tester ({}-{})".format(time.strftime("%x"), time.strftime("%X"))
        super(TestUnit, self).__init__(name, caches)
        self.splits = dict(
            all = list(range(81)),
            contour = list(range(19)),
            eye =  list(range(19, 28)) + list(range(64, 73)),
            eyebrow = list(range(28, 36)) + list(range(73, 81)),
            mouth = list(range(36, 54)),
            nose = list(range(54, 64))
        )
        self.n_points = 81

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
        net = kwargs['net']
        mean_face = misc.load_pickle(kwargs['mean_face_file'])
        img_opr = net.outputs_visitor.all_oprs_dict['img']
        image_height, image_width= img_opr.partial_shape[2:]
        img_type = img_opr.dtype
        inf_func = env.compile_inf_func(net.outputs, devices=devices)
        self._inf_outputs_data = OrderedDict()
        for k, v in tqdm.tqdm(self._kv_dict_all.items()):
            frames, dets, fps = v
            faces = sorted(dets[0][0], key=lambda x:x[2]*x[3])
            left, top, width, height= faces[-1][:4]
            # left, top, width, height= dets[0][0][0][:4]
            left -= int(width * 0.1)
            top -= int(height * 0.1)
            width = int(1.2 * width)
            height = int(1.2 * height)
            right = left + width
            bottom = top + height
            src = np.float32([[left, top], [right, top], [left, bottom], [right, bottom]])
            dst = np.float32([[0, 0], [image_width, 0], [0, image_height], [image_width, image_height]])
            mat = cv2.getPerspectiveTransform(src, dst)
            mat_inv = np.linalg.inv(mat)
            img_raw = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)[np.newaxis, :, :]
            img = cv2.warpPerspective(frames[0], mat, (image_width, image_height))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[np.newaxis, :, :]

            if not img_type == np.uint8:
                img =  img.astype(np.float32)

            feed_dict = dict(
                img = img[np.newaxis, :, :, :],
            )
            output_data = env.forward(inf_func, feed_dict)
            s_pred_val = output_data[net.outputs[0].name][0].reshape(-1, 2) * np.float32([image_width, image_height])
            s_pred_val_raw = np.concatenate([s_pred_val, np.ones((len(s_pred_val), 1))], axis=1).dot(mat_inv.T)[:, :2]
            frames_out = []
            img_copy = draw(img_raw, s_pred_val_raw, rects=[left, top, right, bottom], wait=False, show=False)
            frames_out.append(img_copy)
            for frame in frames[1:]:
                img_raw = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)[np.newaxis, :, :]
                mat = geom.align_to(s_pred_val_raw, mean_face, (image_height, image_width), extend=True).T
                mat_inv = np.linalg.inv(mat)
                img = cv2.warpPerspective(frame, mat, (image_width, image_height))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[np.newaxis, :, :]
                if not img_type == np.uint8:
                    img =  img.astype(np.float32)
                feed_dict = dict(
                    img = img[np.newaxis, :, :, :],
                )
                output_data = env.forward(inf_func, feed_dict)
                s_pred_val = output_data[net.outputs[0].name][0].reshape(-1, 2) * np.float32([image_width, image_height])
                s_pred_val_raw = np.concatenate([s_pred_val, np.ones((len(s_pred_val), 1))], axis=1).dot(mat_inv.T)[:, :2]
                img_copy = draw(img_raw, s_pred_val_raw, wait=False, show=False)
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
    parser.add_argument('--lm_name', default='s-pred')
    parser.add_argument('--data_root', default='/unsullied/sharefs/_research_facelm/Isilon-datashare/baseimg/select_track')
    parser.add_argument('--out_root', default='/tmp/out')
    parser.add_argument('--video_suffix', default='mp4,avi')
    parser.add_argument('--det_suffix', default='det')
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
    )
    return t.compute_metric(
        out_root=args.out_root,
    )



