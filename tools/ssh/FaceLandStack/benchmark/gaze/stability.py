import time
from collections import OrderedDict
import numpy as np

from megskull.network.raw import RawNetworkBuilder

from benchmark.dataset import gaze
from benchmark.tester import TestUnitBase
from landstack.utils import misc, transforms
from landstack.utils import augment, geom
from landstack.utils.visualization import draw
from landstack.train import env
from IPython import embed


class TestUnit(TestUnitBase):
    def __init__(self, caches):
        name = "Gaze Stability Tester ({}-{})".format(time.strftime("%x"), time.strftime("%X"))
        super(TestUnit, self).__init__(name, caches)
        self.splits = dict(
            all = list(range(14)),
            iris = list(range(4)),
            gaze = list(range(4,6)),
            eye = list(range(6,14))
        )
        self.n_points = 14

    def gen_kv_dict_all(self, **kwargs):
        if self._kv_dict_all is None:
            n_samples = kwargs['n_samples']
            val_nori_file = kwargs['val_nori_file']
            val_keys = kwargs['val_keys']
            val_ds = gaze.GazeImage(val_nori_file, val_keys)
            self._kv_dict_all = val_ds.load(
                n_samples=n_samples, 
            )
            self._caches.setdefault('kv_dict_all', self._kv_dict_all)
        else:
            print("Using cached data")
        return self._kv_dict_all

    def gen_inf_outputs_data(self, **kwargs):
        assert self._kv_dict_all, 'Please generate kv_dict_all first'
        devices = kwargs.get('devices', 'gpu0')
        net = kwargs['net']
        lm_name = kwargs['lm_name']
        image_shape = kwargs['image_shape']
        n_loop = kwargs.get('n_loop', 150)
        mean_face = misc.load_pickle(kwargs['mean_face_file'])

        # build augmenter
        aug = augment.AugmentGaze(
            mean_face=np.array(mean_face), width_height=image_shape,
            transforms_post=transforms.Compose([
                transforms.HWC2CHW(),
                transforms.NormalizeLandmark(image_shape),
            ])
        )

        oprs = net.outputs_visitor.all_oprs_dict
        print(net.outputs)
        assert lm_name in oprs, lm_name
        outputs = {lm_name: oprs[lm_name]}
        inf_func = env.compile_inf_func(outputs, devices=devices)

        # loop
        self._inf_outputs_data = OrderedDict()
        for ds_name, data_dict_subset in self._kv_dict_all.items():
            trace = []
            img = data_dict_subset['img']
            lm = data_dict_subset['label']
            for _ in range(n_loop):
                img_relative, trans_mat = [], []
                for img0, lm0 in zip(img, lm):
                    img_relative0, _, trans_mat0 = aug(img0, lm0, mat=True)
                    img_relative.append(img_relative0)
                    trans_mat.append(trans_mat0)
                feed_data = dict(
                    img = np.array(img_relative).astype('float32')
                )
                lm_pred = env.forward(inf_func, feed_data)[lm_name]
                lm_pred *= image_shape[0]

                lm = geom.lm_affine(lm_pred, geom.invert_mat(trans_mat))
                trace.append(lm)
            trace = np.array(trace)
            self._inf_outputs_data.setdefault(ds_name, trace)

            return self._inf_outputs_data

    def compute_metric(self, components):
        assert self._kv_dict_all and self._inf_outputs_data, 'Please gen kv_dict_all and inf_outputs_data first'
        res_dict = OrderedDict()
        for ds_name, trace in self._inf_outputs_data.items():
            lm_gt = self._kv_dict_all[ds_name]['label']
            trace = trace.transpose(1, 0, 2, 3) # [trace, samples, points, 2] to [samples, trace, points, 2]
            mean = trace.mean(axis=1, keepdims=True)
            norm_dist = ((trace - mean) ** 2).sum(axis=3) ** 0.5
            std_norm_dist = norm_dist.mean(axis=1)
            split_std_dict = OrderedDict()
            for c in components:
                split_std_dict.setdefault(c, std_norm_dist[:, self.splits[c]].mean())
            res_dict.setdefault(ds_name, split_std_dict)
        return res_dict

def main(model, devices, args_input, caches):
    """args_input is a list of rest arguments"""
    import argparse
    from meghair.utils import io

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--keys', type=str, default=None)
    parser.add_argument('--components', type=str, default=None)
    parser.add_argument('--lm_name', default='main:pred_lm')
    parser.add_argument('--n_loop', type=int, default=150)
    parser.add_argument('--n_samples', type=int, default=50)
    args = parser.parse_args(args_input)

    # params
    val_keys = args.keys and args.keys.split(',') or [
        'valid',
    ]
    components = args.components or 'all.iris.gaze.eye'
    components = components.split('.')

    # load model
    net = io.load(model)
    net = net['network'] if not isinstance(net, RawNetworkBuilder) else net

    # init TestUnit
    t = TestUnit(caches=caches)
    t.gen_kv_dict_all(
        val_nori_file='/unsullied/sharefs/g:research_facelm/Isilon-datashare/gaze/gaze_nori.info',
        val_keys=val_keys,
        n_samples=args.n_samples,
    )
    t.gen_inf_outputs_data(
        mean_face_file='/unsullied/sharefs/g:research_facelm/Isilon-datashare/gaze/mean_eye.pkl',
        image_shape=(32, 48),
        n_loop=args.n_loop,
        net=net,
        lm_name=args.lm_name,
        devices=devices,
    )
    return t.compute_metric(
        components=components,
    )



