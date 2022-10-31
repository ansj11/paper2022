import time
from collections import OrderedDict

from benchmark.dataset import EyeOccImage
from benchmark.tester import TestUnitBase
from landstack.utils.visualization import draw
from landstack.utils import misc
from landstack.train import env
from IPython import embed
import numpy as np
import os
import cv2
from sklearn.metrics import confusion_matrix


def compute_cfm_acc(pred, gt):
    pred = pred.reshape(-1, 4)
    gt = gt.reshape(-1, 4)
    mask = gt.sum(axis=1).flatten() == 1
    pred = pred[mask]
    gt = gt[mask]
    y_pred = np.argmax(pred, axis=1)
    y_true = np.argmax(gt, axis=1)
    acc = (y_pred == y_true).mean()
    cfm = confusion_matrix(y_true, y_pred)
    return cfm, acc


def m_cross_entropy(pred, label):
    loss_no_reduce = - label * np.log(pred + 1e-5)
    loss = loss_no_reduce.sum() / label.sum()
    return loss

class TestUnit(TestUnitBase):
    def __init__(self, caches):
        name = "eyeocc.accuracy Tester ({}-{})".format(time.strftime("%x"), time.strftime("%X"))
        super(TestUnit, self).__init__(name, caches)
        self.n_points = 81
    def gen_kv_dict_all(self, image_shape, val_keys, is_color, norm_type, badcase):
        if self._kv_dict_all is None:
            val_ds = EyeOccImage(keys=val_keys)
            self._kv_dict_all = val_ds.load(
                norm_type=norm_type,
                image_shape=image_shape,
                is_color=is_color
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

    def compute_metric(self, lm_name, badcase, out_root='/tmp/eyeocc.accuracy'):
        assert self._kv_dict_all and self._inf_outputs_data, 'Please gen kv_dict_all and inf_outputs_data first'
        res_dict = OrderedDict()
        if badcase:
            misc.ensure_dir(out_root, erase=True)

        loss_all = []
        for key in self._kv_dict_all.keys():
            prob_key, pred_key = lm_name.split(',')
            lm_pred = self._inf_outputs_data[key][pred_key]
            gt = self._kv_dict_all[key]['eyeocc'].reshape(-1, 2, 4)
            probs =  self._inf_outputs_data[key][prob_key].reshape(-1, 2, 4)
            left_prob = probs[:, 0, :]
            right_prob = probs[:, 1, :]
            left_gt = gt[:, 0, :]
            right_gt = gt[:, 1, :]

            cfm_left, acc_left = compute_cfm_acc(left_prob, left_gt)
            cfm_right, acc_right = compute_cfm_acc(right_prob, right_gt)
            val_loss_left = m_cross_entropy(left_prob, left_gt)
            val_loss_right = m_cross_entropy(right_prob, right_gt)
            val_loss = (val_loss_left + val_loss_right) / 2.0

            res_dict0 = OrderedDict()
            res_dict0['loss'] = val_loss
            res_dict0['loss_left'] = val_loss_left
            res_dict0['loss_right'] = val_loss_right
            res_dict0['acc_left'] = '{:.5f}'.format(acc_left)
            res_dict0['acc_right'] = '{:.5f}'.format(acc_right)
            res_dict[key] = res_dict0

            if key != 'all-in':
                loss_all.append(val_loss)
            if badcase:
                nori_id = self._kv_dict_all[key]['nori_id']
                img = self._kv_dict_all[key]['img']
                ds_save_root = os.path.join(out_root, key)
                left_ds_save_root = os.path.join(ds_save_root, 'left')
                right_ds_save_root = os.path.join(ds_save_root, 'right')
                misc.ensure_dir(left_ds_save_root, right_ds_save_root)
                np.savetxt(os.path.join(left_ds_save_root, 'cfm.txt'), cfm_left, fmt='%d')
                np.savetxt(os.path.join(right_ds_save_root, 'cfm.txt'), cfm_right, fmt='%d')
                for j in range(len(img)):
                    left_gt0 = np.argmax(left_gt[j])
                    left_pred0 = np.argmax(left_prob[j])
                    right_gt0 = np.argmax(right_gt[j])
                    right_pred0 = np.argmax(right_prob[j])
                    if left_gt[j].sum() != 0 and left_gt0 != left_pred0:
                        save_file = os.path.join(left_ds_save_root, 'mis_{}_as_{}'.format(left_gt0, left_pred0), '{}-{:.2f}-{:.2f}-{:.2f}-{:.2f}.jpg'.format(nori_id[j], *list(left_prob[j])))
                        img_copy = draw(img[j], lm_pred[j], wait=False, show=False, point_size=1)
                        misc.ensure_dir(os.path.dirname(save_file))
                        cv2.imwrite(save_file, img_copy)
                    if right_gt[j].sum() != 0 and right_gt0 != right_pred0:
                        save_file = os.path.join(right_ds_save_root, 'mis_{}_as_{}'.format(right_gt0, right_pred0), '{}-{:.2f}-{:.2f}-{:.2f}-{:.2f}.jpg'.format(nori_id[j], *list(right_prob[j])))
                        img_copy = draw(img[j], lm_pred[j], wait=False, show=False, point_size=1)
                        misc.ensure_dir(os.path.dirname(save_file))
                        cv2.imwrite(save_file, img_copy)
        print('avg loss: {:.3f}'.format(np.mean(loss_all)))
        return res_dict

def main(model, devices, args_input, caches):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--keys', type=str, default=None)
    parser.add_argument('--lm_name', default='s-pred-eyeocc,s-pred')
    parser.add_argument('--badcase', action='store_true', help='whether to log ')
    parser.add_argument('--norm', choices=['geom', 'pupil', 'area'], default='geom')
    parser.add_argument('--is_color', action='store_true')
    parser.add_argument('--lowbit', action='store_true')
    parser.add_argument('--on-mobile', action='store_true', help='test acc on mobile')
    parser.add_argument('--profile', default=None)
    parser.add_argument('--debug', action='store_true', help='enable -c dbg')
    parser.add_argument('--network-outputs', default='/tmp/network_outputs.pkl')
    parser.add_argument('--sfa', default=7, type=int)
    parser.add_argument('--bit_midout', default=16, type=int)

    args, unknownargs = parser.parse_known_args(args_input)

    # params
    val_keys = args.keys and args.keys.split(',')

    # load model
    assert len(args.lm_name.split(',')) == 2, args.lm_name
    net = misc.load_network_and_extract(model, args.lm_name)

    # init TestUnit
    t = TestUnit(caches=caches)
    t.gen_kv_dict_all(
        val_keys=val_keys,
        image_shape=(112, 112),
        norm_type=args.norm,
        is_color=args.is_color,	#load img data
        badcase=args.badcase,
    )

    if args.lowbit or args.on_mobile:
        command = "lsk-test gen.speed {} --lm_name {} --keys {} --no_info --network-outputs {} --sfa {} --bit_midout {} --dataset mouthocc.accuracy".format(
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
            devices=devices,
        )

    return t.compute_metric(
        lm_name=args.lm_name,
        badcase=args.badcase,
    )
