import time
from collections import OrderedDict

from megskull.network.raw import RawNetworkBuilder

from benchmark.dataset import dense
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
            mean_face_file = kwargs['mean_face_file']
            image_shape = kwargs['image_shape']
            val_nori_file = kwargs['val_nori_file']
            val_keys = kwargs['val_keys']
            is_color = kwargs['is_color']
            norm_type = kwargs['norm_type']
            val_ds = dense.DenseImage(val_nori_file, val_keys)
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

    def compute_metric(self, components=None):
        assert self._kv_dict_all and self._inf_outputs_data, 'Please gen kv_dict_all and inf_outputs_data first'
        res_dict = OrderedDict()
        for k in self._kv_dict_all.keys():
            norm = self._kv_dict_all[k]['norm']
            lm_gt = self._kv_dict_all[k]['label']
            lm_gt = lm_gt[:,:248]
            lm_pred = self._inf_outputs_data[k]
            lm_pred = list(lm_pred.values())[0] if isinstance(lm_pred, dict) else lm_pred
            assert len(norm) == len(lm_pred), '{} vs. {}'.format(len(norm), len(lm_pred))
            val_loss = ((lm_gt - lm_pred)**2).reshape(-1, self.n_points, 2)
            val_loss = val_loss.sum(axis=2)**0.5/norm
            split_loss_dict = OrderedDict()
            for c in components:
                split_loss_dict.setdefault(c, val_loss[:, self.splits[c]].mean())
            res_dict.setdefault(k, split_loss_dict)
        return res_dict

def main(model, devices, args_input, caches):
    """args_input is a list of rest arguments"""
    import argparse
    from meghair.utils import io

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
        'validation-valid',
    ]
    components = args.components or 'all.contour.eye.eyebrow.nose.mouth'
    components = components.split('.')

    # load model
    net = io.load(model)
    net = net['network'] if not isinstance(net, RawNetworkBuilder) else net
    print(net.outputs)

    # init TestUnit
    t = TestUnit(caches=caches)
    t.gen_kv_dict_all(
        mean_face_file='/unsullied/sharefs/_research_facelm/Isilon-datashare/landmarkextend/dense/extra/mean_dense.pkl',
        val_nori_file='/unsullied/sharefs/_research_facelm/Isilon-datashare/landmarkextend/dense/landmarkdense_170830.info',
        val_keys=val_keys,
        image_shape=(112, 112),
        norm_type=args.norm,
        is_color=args.is_color,
    )

    if args.lowbit or args.on_mobile:
        command = "lsk-test gen.speed {} --lm_name {} --keys {} --no_info --network-outputs {} --sfa {} --bit_midout {}".format(
            model, args.lm_name, args.keys, args.network_outputs, args.sfa, args.bit_midout)
        command = command + ' --lowbit' if args.lowbit else command
        command = command + ' --debug' if args.debug else command
        command = command + ' --profile {}'.format(args.profile) if args.profile is not None else command
        print("Invoking sys command {}".format(command))
        res = os.system(command)
        if res:
            print("Error exit")
        else:
            inf_outputs_data = misc.load_pickle(args.network_outputs)
            t._inf_outputs_data = inf_outputs_data
    else:
        t.gen_inf_outputs_data(
            net=net,
            lm_name=args.lm_name,
            devices=devices,
        )
    return t.compute_metric(
        components=components,
    )



