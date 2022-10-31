import time
from collections import OrderedDict
import numpy as np
from benchmark.dataset import GazeImage
from benchmark.tester import TestUnitBase
from landstack.utils import misc
from landstack.train import env
from benchmark.dataset.gaze import decode

from IPython import embed

class TestUnit(TestUnitBase):
    def __init__(self, caches):
        name = "Gaze Accuracy Tester ({}-{})".format(time.strftime("%x"), time.strftime("%X"))
        super(TestUnit, self).__init__(name, caches)

    def gen_kv_dict_all(self, width_height, val_keys, is_color):
        if self._kv_dict_all is None:
            val_ds = GazeImage(keys=val_keys)
            self._kv_dict_all = val_ds.load(
                width_height=width_height,
                is_color=is_color,
            )
            all_in_kv_dict = OrderedDict()
            for k, v in self._kv_dict_all.items():
                for kk, vv in v.items():
                    all_in_kv_dict.setdefault(kk, []).append(vv)
            for k, v in all_in_kv_dict.items():
                self._kv_dict_all.setdefault('all-in', OrderedDict())[k] = np.concatenate(v)
            self._caches.setdefault('kv_dict_all', self._kv_dict_all)

        else:
            print("Using cached data")
        return self._kv_dict_all

    def gen_inf_outputs_data(self, net, devices):
        assert self._kv_dict_all, 'Please generate kv_dict_all first'
        inf_func = env.compile_inf_func(list(net.outputs), devices=devices)
        self._inf_outputs_data = OrderedDict()
        for k, v in self._kv_dict_all.items():
            self._inf_outputs_data.setdefault(k, env.forward(inf_func, v))
        return self._inf_outputs_data

    def compute_metric(self, lm_name, badcase, out_root='/tmp/gaze.accuracy'):
        assert self._kv_dict_all and self._inf_outputs_data, 'Please gen kv_dict_all and inf_outputs_data first'
        res_dict = OrderedDict()
        # if badcase:
        #     misc.ensure_dir(out_root, erase=True)

        loss_all = []
        for key in self._kv_dict_all.keys():
            r = decode(self._kv_dict_all[key]['label'])
            ec_gt = r['eye_center']
            gp_gt = r['gaze_point']
            gaze_pred = self._inf_outputs_data[key][lm_name]
            ec_pred = gaze_pred[:, :2]
            gp_pred = gaze_pred[:, 2:]

            ec_loss_l1 =  np.abs(ec_gt - ec_pred).mean()
            ec_loss_l2 = np.sqrt(((ec_gt - ec_pred).reshape(-1, 2) ** 2).sum(axis=1)).mean()
            gp_loss_l1 =  np.abs(gp_gt - gp_pred).mean()
            gp_loss_l2_no_reduce = np.sqrt(((gp_gt - gp_pred).reshape(-1, 2) ** 2).sum(axis=1))
            gp_loss_l2 = gp_loss_l2_no_reduce.mean()
            loss_l1 = (ec_loss_l1 + gp_loss_l1) / 2.0
            loss_l2 = (ec_loss_l2 + gp_loss_l2) / 2.0
            res_dict0 = OrderedDict()
            res_dict0['eye_center(l1/l2)'] = '{:.3f}/{:.3f}'.format(ec_loss_l1, ec_loss_l2)
            res_dict0['gaze_point(l1/l2)'] = '{:.3f}/{:.3f}'.format(gp_loss_l1, gp_loss_l2)
            res_dict0['l1'] = '{:.3f}'.format(loss_l1)
            res_dict0['l2'] = '{:.3f}'.format(loss_l2)
            res_dict[key] = res_dict0
            if key != 'all-in':
                loss_all.append(gp_loss_l1)

            # if badcase and 'img_raw' in self._kv_dict_all[key]:
            #     nori_id = self._kv_dict_all[key]['nori_id']
            #     indices = np.argsort(gp_loss_l2_no_reduce.flatten())[::-1]
            #     for idx in indices:
            #     path = os.path.join(out_root, key)
            #     v=self._kv_dict_all[key]
            #     for idx,nori_id in enumerate(v['nori_id']):
            #         img_out = draw(v['img_gaze'][idx],lm_gt[idx][8:12],lm_pred[idx][8:12],1-lm_pred[idx][8:12],canvas_height_width=(160,240))
            #         img_path = os.path.join(path,'{}.jpg'.format(nori_id))
            #         cv2.imwrite(img_path,img_out)

        print('avg gaze_point l2 loss: {:.3f}'.format(np.mean(loss_all)))
        return res_dict

def main(model, devices, args_input, caches):
    """args_input is a list of rest arguments"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--keys', type=str, default=None)
    parser.add_argument('--lm_name', default='s-pred-gaze-patch')
    parser.add_argument('--is_color', action='store_true')
    parser.add_argument('--badcase', action='store_true', help='whether to log ')
    args, unknownargs = parser.parse_known_args(args_input)

    # params
    val_keys = args.keys and args.keys.split(',')

    # load model
    net = misc.load_network_and_extract(model, args.lm_name)

    # init TestUnit
    t = TestUnit(caches=caches)
    t.gen_kv_dict_all(
        val_keys=val_keys,
        width_height=(48, 32),
        is_color=args.is_color,
    )
    t.gen_inf_outputs_data(
        net=net,
        devices=devices,
    )
    return t.compute_metric(
        lm_name=args.lm_name,
        badcase=args.badcase,
    )



