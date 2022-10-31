import time
from collections import OrderedDict

from megskull.network.raw import RawNetworkBuilder

from benchmark.dataset import tdfaw
from benchmark.tester import TestUnitBase
from landstack.utils.visualization import draw
from landstack.utils import misc
from landstack.train import env
from IPython import embed
import os

class TestUnit(TestUnitBase):
    def __init__(self, caches):
        name = "Accuracy Tester ({}-{})".format(time.strftime("%x"), time.strftime("%X"))
        super(TestUnit, self).__init__(name, caches)
        self.n_points = 66

    def gen_kv_dict_all(self, **kwargs):
        if self._kv_dict_all is None:
            mean_face_file = kwargs['mean_face_file']
            image_shape = kwargs['image_shape']
            val_nori_file = kwargs['val_nori_file']
            val_keys = kwargs['val_keys']
            is_color = kwargs['is_color']
            norm_type = kwargs['norm_type']
            val_ds = tdfaw.tdfawImage(val_nori_file, val_keys)
            self._kv_dict_all = val_ds.load(
                align=True,
                norm_type=norm_type,
                mean_face_file=mean_face_file,
                image_shape=image_shape,
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
        inf_func = env.compile_inf_func(net.outputs, devices=devices)
        self._inf_outputs_data = OrderedDict()
        for k, v in self._kv_dict_all.items():
            self._inf_outputs_data.setdefault(k, env.forward(inf_func, v))
        return self._inf_outputs_data

    def compute_metric(self, components=None):
        assert self._kv_dict_all and self._inf_outputs_data, 'Please gen kv_dict_all and inf_outputs_data first'
        res_dict = OrderedDict()
        for k in self._kv_dict_all.keys():
            norm = self._kv_dict_all[k]['norm']
            lm_gt = self._kv_dict_all[k]['label']
            lm_pred = self._inf_outputs_data[k]
            lm_pred = list(lm_pred.values())[0] if isinstance(lm_pred, dict) else lm_pred
            assert len(norm) == len(lm_pred), '{} vs. {}'.format(len(norm), len(lm_pred))
            val_loss = ((lm_gt - lm_pred)**2).reshape(-1, self.n_points, 3)
            val_loss = val_loss.sum(axis=2)**0.5/norm
            res_dict.setdefault(k, {'all':val_loss.mean()})
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
    parser.add_argument('--lowbit', action='store_true')
    parser.add_argument('--on-mobile', action='store_true', help='test acc on mobile')
    parser.add_argument('--profile', default=None)
    parser.add_argument('--debug', action='store_true', help='enable -c dbg')
    parser.add_argument('--network-outputs', default='/tmp/network_outputs.pkl')
    parser.add_argument('--sfa', default=7)
    parser.add_argument('--bit_midout', default=16)

    args, unknownargs = parser.parse_known_args(args_input)

    # params
    val_keys = args.keys and args.keys.split(',') or [
        'valid'
    ]
    components = args.components or 'all.contour.eye.eyebrow.nose.mouth'
    components = components.split('.')

    # load model
    net = misc.load_network_and_extract(model, args.lm_name)

    # init TestUnit
    t = TestUnit(caches=caches)
    t.gen_kv_dict_all(
        mean_face_file='/unsullied/sharefs/_research_facelm/Isilon-datashare/3dlandmark/3dfaw/extra/mean_3dfaw.pkl',
        val_nori_file='/unsullied/sharefs/_research_facelm/Isilon-datashare/3dlandmark/3dfaw/landmark3d.info',
        val_keys=val_keys,
        image_shape=(112, 112),
        norm_type=args.norm,
        is_color=args.is_color,
    )

    t.gen_inf_outputs_data(
        net=net,
        devices=devices,
    )
    return t.compute_metric()



