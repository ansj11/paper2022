import time
from collections import OrderedDict
import numpy as np

from megskull.network.raw import RawNetworkBuilder

from benchmark.dataset import static
from benchmark.tester import TestUnitBase
from landstack.utils import misc
from landstack.utils.visualization import draw
from landstack.train import env
import cv2
from IPython import embed
import os
import tqdm

class TestUnit(TestUnitBase):
    def __init__(self, caches):
        name = "Accuracy Show Result Tester ({}-{})".format(time.strftime("%x"), time.strftime("%X"))
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
            val_ds = static.StaticImage81(val_nori_file, val_keys)
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
        lm_name = kwargs['lm_name']

        oprs = net.outputs_visitor.all_oprs_dict
        assert lm_name in oprs, lm_name
        outputs = {lm_name: oprs[lm_name]}
        inf_func = env.compile_inf_func(outputs, devices=devices)
        self._inf_outputs_data = OrderedDict()
        for k, v in self._kv_dict_all.items():
            self._inf_outputs_data.setdefault(k, env.forward(inf_func, v))
        return self._inf_outputs_data

    def compute_metric(self, img_dir, img_num, is_sort=False):
        assert self._kv_dict_all and self._inf_outputs_data, 'Please gen kv_dict_all and inf_outputs_data first'
        res_dict = OrderedDict()
        for k in self._kv_dict_all.keys():
            img = self._kv_dict_all[k]['img']
            norm = self._kv_dict_all[k]['norm']
            lm_gt = self._kv_dict_all[k]['label']
            lm_pred = list(self._inf_outputs_data[k].values())[0]
            assert len(norm) == len(lm_pred), '{} vs. {}'.format(len(norm), len(lm_pred))
            val_loss = ((lm_gt - lm_pred)**2).reshape(-1, self.n_points, 2)
            val_loss = val_loss.sum(axis=2)**0.5/norm
            val_loss = val_loss.mean(axis=1)

            index = list(range(img_num))
            if is_sort:
                index = np.argsort(val_loss*(-1))[:img_num]
            out_dir = os.path.join(img_dir, k)
            misc.ensure_dir(out_dir)
            for i in tqdm.tqdm(range(len(index))):
                idx = index[i]
                img_show = draw(img[idx], lm_pred[idx], lm_gt[idx], wait=False, show=False)
                cv2.imwrite('{}/{:3d}-{:.5g}.jpg'.format(out_dir, i, val_loss[idx]), img_show)

def main(model, devices, args_input, caches):
    """args_input is a list of rest arguments"""
    import argparse
    from meghair.utils import io

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--keys', type=str, default=None)
    parser.add_argument('--lm_name', default='s-pred')
    parser.add_argument('--norm', help='geom|pupil|area', default='geom')
    parser.add_argument('--is_color', action='store_true')
    parser.add_argument('--img_dir', default='show')
    parser.add_argument('--img_num', default=100)
    parser.add_argument('--is_sort', action='store_true')
    args = parser.parse_args(args_input)

    # params
    val_keys = args.keys and args.keys.split(',') or [
        'validation-valid',
    ]

    # load model
    net = io.load(model)
    net = net['network'] if not isinstance(net, RawNetworkBuilder) else net

    # init TestUnit
    t = TestUnit(caches=caches)
    t.gen_kv_dict_all(
        mean_face_file='/unsullied/sharefs/_research_facelm/Isilon-datashare/baseimg/extra/mean_face_81.pkl',
        val_nori_file='/unsullied/sharefs/_research_facelm/Isilon-datashare/baseimg/landmark_test.info',
        val_keys=val_keys,
        image_shape=(112, 112),
        norm_type=args.norm,
        is_color=args.is_color,
    )
    t.gen_inf_outputs_data(
        net=net,
        lm_name=args.lm_name,
        devices=devices,
    )
    t.compute_metric(
        img_dir=args.img_dir,
        img_num=args.img_num,
        is_sort=args.is_sort
    )



