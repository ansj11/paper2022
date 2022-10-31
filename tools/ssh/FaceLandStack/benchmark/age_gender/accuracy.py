import time
from collections import OrderedDict

from benchmark.dataset import AgeGenderImage
from benchmark.tester import TestUnitBase
from landstack.utils.visualization import draw, quick_draw, montage
from landstack.utils import misc
from landstack.train import env
from IPython import embed
import numpy as np
import os
import cv2

def load_legacy(image_shape, is_color):
    import json
    import tqdm
    hetao_file = '/unsullied/sharefs/_research_facelm/Isilon-datashare/proj/age_gender/source/mobile/0-hetao/hetao.val.json'
    faceid3_file = '/unsullied/sharefs/_research_facelm/Isilon-datashare/proj/age_gender/source/mobile/1-faceid3/faceid3.val.json'
    with open(hetao_file) as f:
        hetao_data = json.load(f)
    with open(faceid3_file) as f:
        faceid_data = json.load(f)
    val_kv_dict = OrderedDict()

    img_batch = []
    age_batch = []
    gender_batch = []
    for k in tqdm.tqdm(hetao_data):
        img = misc.unpack_img(k['dataid'], is_color=True)
        img = cv2.resize(img, image_shape)
        if is_color:
            img = img.transpose(2, 0, 1)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[np.newaxis, :, :]
        img_batch.append(img)
        age_batch.append(k['age'])
        gender_batch.append(k['gender'] == 0)
    val_kv_dict['hetao_legacy'] = dict(
        img=np.uint8(img_batch),
        age=np.float32(age_batch),
        gender=np.float32(gender_batch),
    )

    img_batch = []
    age_batch = []
    gender_batch = []
    for k in tqdm.tqdm(faceid_data):
        img = misc.unpack_lm(k['dataid'], is_color=True)[0]
        img = cv2.resize(img, image_shape)
        if is_color:
            img = img.transpose(2, 0, 1)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[np.newaxis, :, :]
        img_batch.append(img)
        age_batch.append(k['age'])
        gender_batch.append(k['gender'] == 0)
    val_kv_dict['faceid3_legacy'] = dict(
        img=np.uint8(img_batch),
        age=np.float32(age_batch),
        gender=np.float32(gender_batch),
    )
    return val_kv_dict

class TestUnit(TestUnitBase):
    def __init__(self, caches):
        name = "age.gender.accuracy Tester ({}-{})".format(time.strftime("%x"), time.strftime("%X"))
        super(TestUnit, self).__init__(name, caches)
        self.n_points = 81

    def gen_kv_dict_all(self, image_shape, val_keys, is_color, norm_type, badcase):
        if self._kv_dict_all is None:
            val_ds = AgeGenderImage(keys=val_keys)
            self._kv_dict_all = val_ds.load(
                align=True,
                norm_type=norm_type,
                image_shape=image_shape,
                is_color=is_color,
            )
            # self._kv_dict_all.update(load_legacy(image_shape, is_color))

            self._caches.setdefault('kv_dict_all', self._kv_dict_all)
        else:
            print("Using cached data")
        return self._kv_dict_all

    def gen_inf_outputs_data(self, net, devices):
        assert self._kv_dict_all, 'Please generate kv_dict_all first'
        inf_func = env.compile_inf_func(net.outputs, devices=devices)
        self._inf_outputs_data = OrderedDict()
        for k, v in self._kv_dict_all.items():
            self._inf_outputs_data.setdefault(k, env.forward(inf_func, v, batch_size=2000))
        return self._inf_outputs_data

    def compute_metric(self, lm_name, badcase, age_toleration, gender_thr, out_root='/tmp/age.gender.accuracy'):
        assert self._kv_dict_all and self._inf_outputs_data, 'Please gen kv_dict_all and inf_outputs_data first'
        res_dict = OrderedDict()
        if badcase:
            misc.ensure_dir(out_root, erase=True)

        for key in self._kv_dict_all.keys():
            age_gt = self._kv_dict_all[key]['age'].flatten()
            gender_gt = self._kv_dict_all[key]['gender'].flatten()
            age_key, gender_key = lm_name.split(',')
            age_pred = self._inf_outputs_data[key][age_key].flatten()
            gender_prob =  self._inf_outputs_data[key][gender_key].flatten()

            # age metric
            age_loss_no_reduct = np.abs(age_pred - age_gt)
            age_loss = age_loss_no_reduct.mean()
            age_distance = (np.maximum(age_loss_no_reduct - age_toleration, 0) ** 2).mean()
            age_accuracy = np.mean(age_loss_no_reduct <= age_toleration)

            # gender metric
            gender_loss = (-(gender_gt* np.log(gender_prob+ 1e-5) + (1-gender_gt)*np.log(1-gender_prob+1e-5))).mean()
            gender_pred = gender_prob >= gender_thr
            fp = gender_pred * (1 - gender_gt)
            fn = (1 - gender_pred) * gender_gt
            gender_acc = ((gender_prob >= gender_thr) == gender_gt).mean()

            res_dict0 = OrderedDict()
            res_dict0['age_distance'] = age_distance
            res_dict0['age_accuracy'] = age_accuracy
            res_dict0['age_loss'] = age_loss
            res_dict0['gender_loss'] = '{:.4g}'.format(gender_loss)
            res_dict0['gender_acc'.format(gender_thr)] = '{:.3g}'.format(gender_acc)
            res_dict0['thr'] = gender_thr
            res_dict0['#male'] = np.sum(gender_gt)
            res_dict0['#female'] = len(gender_gt) - np.sum(gender_gt)
            res_dict0['#total'] = len(gender_gt)
            res_dict[key] = res_dict0

            if badcase:
                nori_id = self._kv_dict_all[key]['nori_id']
                lm_out = self._kv_dict_all[key]['label']
                img_out = self._kv_dict_all[key]['img']
                ds_save_root = os.path.join(out_root, key)

                # age
                ds_save_root_age = os.path.join(ds_save_root, 'age')
                misc.ensure_dir(ds_save_root_age)
                indices = np.argsort(age_loss_no_reduct)[::-1][:100]
                for ii, idx in enumerate(indices):
                    img0 = img_out[idx]
                    lm0 = lm_out[idx]
                    nori_id0 = nori_id[idx]
                    save_file = os.path.join(ds_save_root_age, '{}-{}.jpg'.format(ii, nori_id0))
                    img_copy = draw(img0, lm0, canvas_height_width=(512, 512),
                                    font_scale=2, text='{:.1f}/real{:.1f}'.format(age_pred[idx], age_gt[idx]), show=False, wait=False)
                    cv2.imwrite(save_file, img_copy)

                # gender
                ds_save_root_gender = os.path.join(ds_save_root, 'gender')
                fp_root = os.path.join(ds_save_root_gender, 'fp')
                fn_root = os.path.join(ds_save_root_gender, 'fn')
                misc.ensure_dir(fp_root)
                misc.ensure_dir(fn_root)

                for j in range(len(img_out)):
                    if fp[j]:
                        img_copy = draw(img_out[j], lm_out[j], text='female',
                                        font_scale=2, canvas_height_width=(512, 512), show=False, wait=False)
                        save_file = os.path.join(fp_root, '{}-{}-{:.5g}.jpg'.format(j, nori_id[j], gender_prob[j]))
                        cv2.imwrite(save_file, img_copy)
                    if fn[j]:
                        img_copy = draw(img_out[j], lm_out[j], text='male',
                                        font_scale=2, canvas_height_width=(512, 512), show=False, wait=False)
                        save_file = os.path.join(fn_root, '{}-{}-{:.5g}.jpg'.format(j, nori_id[j], gender_prob[j]))
                        cv2.imwrite(save_file, img_copy)
        return res_dict

def main(model, devices, args_input, caches):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--keys', type=str, default=None)
    parser.add_argument('--lm_name', default='s-age_s,s-gender_s')
    parser.add_argument('--badcase', action='store_true', help='whether to log ')
    parser.add_argument('--age_toleration', type=int, default=10)
    parser.add_argument('-s', '--image_shape', default='112,112')
    parser.add_argument('--gender_thr', type=float, default=0.5)
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
    args.image_shape = tuple(map(int, args.image_shape.split(',')))
    assert len(args.image_shape) == 2, args.image_shape

    # load model
    assert len(args.lm_name.split(',')) == 2, args.lm_name
    net = misc.load_network_and_extract(model, args.lm_name)

    # init TestUnit
    t = TestUnit(caches=caches)
    t.gen_kv_dict_all(
        val_keys=val_keys,
        image_shape=args.image_shape,
        norm_type=args.norm,
        is_color=args.is_color,
        badcase=args.badcase,
    )

    if args.lowbit or args.on_mobile:
        command = "lsk-test gen.speed {} --lm_name {} --keys {} --no_info --network-outputs {} --sfa {} --bit_midout {} --dataset eyeoc.accuracy".format(
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
        age_toleration=args.age_toleration,
        gender_thr=args.gender_thr,
    )



