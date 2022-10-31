import time
from collections import OrderedDict
import numpy as np
import tqdm
import cv2

from benchmark.dataset import dense
from benchmark.tester import TestUnitBase
from landstack.utils import misc, geom
from landstack.train import env
from landstack.utils.visualization import draw
import os
from IPython import embed


class TestUnit(TestUnitBase):
    def __init__(self, caches):
        name = "Stability Tester ({}-{})".format(time.strftime("%x"), time.strftime("%X"))
        super(TestUnit, self).__init__(name, caches)
        self.splits = dict(
            all = list(range(124)),
            contour = list(range(21)) + list(range(90, 108)),
            eyebrow = list(range(21, 39)) + list(range(108, 124)),
            eye =  list(range(39, 57)),
            nose = list(range(57, 72)),
            mouth = list(range(72, 90)),
        )
        self.n_points = 124

    def gen_kv_dict_all(self, **kwargs):
        if self._kv_dict_all is None:
            n_samples = kwargs['n_samples']
            val_nori_file = kwargs['val_nori_file']
            val_keys = kwargs['val_keys']
            is_color = kwargs['is_color']
            val_ds = dense.DenseImage(val_nori_file, val_keys)
            self._kv_dict_all = val_ds.load(
                n_samples=n_samples, 
                is_color=is_color,
            )
            self._caches.setdefault('kv_dict_all', self._kv_dict_all)
        else:
            print("Using cached data")
        return self._kv_dict_all

    def gen_inf_outputs_data(self, **kwargs):
        assert self._kv_dict_all, 'Please generate kv_dict_all first'
        devices = kwargs.get('devices', 'gpu0')
        net = kwargs['net']
        n_loop = kwargs.get('n_loop', 150)
        sta_out_root = kwargs.get('sta_out_root')
        mean_face = misc.load_pickle(kwargs['mean_face_file'])
        img_opr = net.outputs_visitor.all_oprs_dict['img']
        img_type = img_opr.dtype
        image_height, image_width = img_opr.partial_shape[2:]
        inf_func = env.compile_inf_func(net.outputs, devices=devices)

        # loop
        self._inf_outputs_data = OrderedDict()
        for ds_name, data_dict_subset in self._kv_dict_all.items():
            trace, frames_out = [], []
            img_raw = data_dict_subset['img']
            lm = data_dict_subset['label']
            lm = lm[:,:248]
            n = len(img_raw)
            for _ in tqdm.tqdm(range(n_loop)):
                img, mat_inv = [], []
                for img_raw0, lm0 in zip(img_raw, lm):
                    mat0 = geom.align_to(lm0, mean_face, (image_height, image_width), extend=True).T
                    img0 = cv2.warpPerspective(img_raw0, mat0, (image_width, image_height))[:, :, np.newaxis]
                    img.append(img0)
                    mat_inv.append(np.linalg.inv(mat0))
                if img_type == np.uint8:
                    feed_dict = dict(
                        img = np.uint8(img).transpose(0, 3, 1, 2),
                    )
                else:
                    feed_dict = dict(
                        img = np.float32(img).transpose(0, 3, 1, 2),
                    )

                output_data = env.forward(inf_func, feed_dict)
                lm_pred = output_data[net.outputs[0].name].reshape(-1, 2) * np.float32([image_width, image_height])
                lm_pred = lm_pred.reshape(n, -1, 2)
                mat_inv = np.float32(mat_inv).transpose(0, 2, 1)
                lm_pred_raw = np.matmul(np.concatenate([lm_pred, np.ones((n, -1, 1))], axis=2), mat_inv)[:, :, :2]

                merges = []
                if sta_out_root is not None:
                    for i in range(n):
                        img_copy = draw(img_raw[i], lm_pred_raw[i], wait=False, show=False)
                        merges.append(img_copy)
                frames_out.append(merges)

                trace.append(lm_pred_raw)
                lm = lm_pred_raw

            trace = np.float32(trace)
            frames_out = np.uint8(frames_out)
            self._inf_outputs_data.setdefault(ds_name, [trace, frames_out])
        return self._inf_outputs_data

    def compute_metric(self, norm_type, components, sta_out_root=None):
        assert self._kv_dict_all and self._inf_outputs_data, 'Please gen kv_dict_all and inf_outputs_data first'
        res_dict = OrderedDict()
        for ds_name, (trace, frames) in self._inf_outputs_data.items():
            lm_gt = self._kv_dict_all[ds_name]['label']
            norm = geom.compute_lm_norm(lm_gt, norm_type=norm_type, n_points=124)
            trace = trace.transpose(1, 0, 2, 3) # [trace, samples, points, 2] to [samples, trace, points, 2]
            mean = trace.mean(axis=1, keepdims=True)
            norm_dist = ((trace - mean) ** 2).sum(axis=3) ** 0.5
            # std_norm_dist = norm_dist.std(axis=1)
            std_norm_dist = norm_dist.mean(axis=1) / norm[:, np.newaxis]
            split_std_dict = OrderedDict()
            for c in components:
                split_std_dict.setdefault(c, std_norm_dist[:, self.splits[c]].mean())
            res_dict.setdefault(ds_name, split_std_dict)

            if sta_out_root is not None:
                misc.ensure_dir(sta_out_root)
                fourcc = cv2.VideoWriter_fourcc(*'H264')
                frames = frames.transpose(1, 0, 2, 3, 4) #[trace, sample, img] to [sample, trace, img]
                for idx, frame in enumerate(frames):
                    height, width = frame[0].shape[:2]
                    out_file = os.path.join(sta_out_root, '{}-{:.4f}.mp4'.format(idx, std_norm_dist[idx].mean()))
                    print("Saving to {}".format(out_file))
                    out = cv2.VideoWriter(out_file, fourcc, 40, (width, height))
                    for frame0 in frame:
                        out.write(frame0)
                    out.release()
        return res_dict

def main(model, devices, args_input, caches):
    """args_input is a list of rest arguments"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--keys', type=str, default=None)
    parser.add_argument('--components', type=str, default=None)
    parser.add_argument('--lm_name', default='s-pred')
    parser.add_argument('--norm', choices=['geom', 'pupil', 'area'], default='geom')
    parser.add_argument('--is_color', action='store_true')
    parser.add_argument('--n_loop', type=int, default=50)
    parser.add_argument('--n_samples', type=int, default=150)
    parser.add_argument('--sta_out_root', default=None)
    args, unknownargs = parser.parse_known_args(args_input)

    # params
    val_keys = args.keys and args.keys.split(',') or [
        'validation-valid',
    ]
    components = args.components or 'all.contour.eye.eyebrow.nose.mouth'
    components = components.split('.')

    # load model
    net = misc.load_network_and_extract(model, args.lm_name.split(','))

    # init TestUnit
    t = TestUnit(caches=caches)
    t.gen_kv_dict_all(
        val_nori_file='/unsullied/sharefs/_research_facelm/Isilon-datashare/baseimg/alljobs/jobdense/landmark_dense.info',
        val_keys=val_keys,
        n_samples=args.n_samples,
        is_color=args.is_color,
    )
    t.gen_inf_outputs_data(
        mean_face_file='/unsullied/sharefs/_research_facelm/Isilon-datashare/baseimg/alljobs/jobdense/extra/mean_dense.pkl',
        n_loop=args.n_loop,
        net=net,
        devices=devices,
        sta_out_root=args.sta_out_root,
    )
    return t.compute_metric(
        components=components,
        norm_type=args.norm,
        sta_out_root=args.sta_out_root,
    )



