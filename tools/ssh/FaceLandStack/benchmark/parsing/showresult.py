import time
from collections import OrderedDict
import numpy as np

from megskull.network.raw import RawNetworkBuilder

from benchmark.dataset import parsing
from benchmark.tester import TestUnitBase
from landstack.utils import misc
from landstack.utils.visualization import draw
from landstack.train import env
import cv2
from IPython import embed
import os
import tqdm
from matplotlib import pyplot as plt

class TestUnit():
    def __init__(self):
        self._kv_dict_all = None
        self._inf_outputs_data = None
        return 

    def gen_kv_dict_all(self, mean_face_file, part_shape, val_nori_file, val_keys):
        if self._kv_dict_all is None:
            val_ds = parsing.ParsingImage(val_nori_file, val_keys)
            self._kv_dict_all = val_ds.load(
                mean_face_file=mean_face_file,
                part_shape=part_shape,
            )
        return self._kv_dict_all

    def gen_part_outputs_data(self, net, lm_name, devices):
        assert self._kv_dict_all, 'Please generate kv_dict_all first'
        oprs = net.outputs_visitor.all_oprs_dict
        assert lm_name in oprs, lm_name
        outputs = {lm_name: oprs[lm_name]}
        inf_func = env.compile_inf_func(outputs, devices=devices)
        outputs_data = OrderedDict()
        for k, v in self._kv_dict_all.items():
            outputs_data.setdefault(k, env.forward(inf_func, v))
        return outputs_data

    def gen_inf_outputs_data(self, nets, oprs, devices):
        self._inf_outputs_data = {}
        for name in nets.keys():
            self._inf_outputs_data[name] = self.gen_part_outputs_data(nets[name], oprs[name], devices)
            #print(self._inf_outputs_data[name]['helen_valid'][oprs[name]].shape)
        return self._inf_outputs_data

    def compute_metric(self, img_dir, img_num):
        assert self._kv_dict_all and self._inf_outputs_data, 'Please gen kv_dict_all and inf_outputs_data first'
        for k in self._kv_dict_all.keys():
            out_dir = os.path.join(img_dir, k)
            misc.ensure_dir(out_dir)
            part_shape={'face':(128,128), 'leye':(32,32), 'reye':(32,32), 'nose':(32,32), 'mouth':(32,32)}
            part_type = ['leye', 'reye', 'nose', 'mouth']
            for img_num in range(self._kv_dict_all[k]['imgface'].shape[0]):
                img = self._kv_dict_all[k]['imgface'][img_num, 0, :, :]
                lm = self._kv_dict_all[k]['lm'][img_num, :, :]
                mask_gt = np.zeros([11, img.shape[0], img.shape[1]])
                mask_pred = np.zeros([11, img.shape[0], img.shape[1]])
                mask_gt[:3, :, :] = self._kv_dict_all[k]['maskface'][img_num, :, :, :]
                mask_pred[:3, :, :] = self._inf_outputs_data['face'][k]['s-pred'][img_num, :, :, :]
                start_channel = 3
                stop_channel = 3
                for _type in part_type:
                    outw, outh = part_shape[_type]
                    if _type == 'leye':
                        lmp = np.concatenate((lm[134:154,:],lm[174:194,:]), axis=0).reshape(-1,2)
                    elif _type == 'reye':
                        lmp = np.concatenate((lm[114:134,:],lm[154:174,:]), axis=0).reshape(-1,2)
                    elif _type == 'nose':
                        lmp = np.concatenate((lm[114:115,:],lm[134:135,:],lm[41:58,:]), axis=0).reshape(-1,2)
                    elif _type == 'mouth':
                        lmp = lm[58:114,:]
                    
                    x0,x1,y0,y1 = lmp[:,0].min(), lmp[:,0].max(), lmp[:,1].min(), lmp[:,1].max()
                    xc, yc = (x0+x1)/2, (y0+y1)/2
                    x0,x1,y0,y1 = int(xc-outw/2),int(xc+outw/2),int(yc-outh/2),int(yc+outh/2)
                    stop_channel += self._kv_dict_all[k]['mask' + _type].shape[1]
                    mask_gt[start_channel: stop_channel, y0: y1, x0: x1] = self._kv_dict_all[k]['mask' + _type][img_num, : , :, :]
                    mask_pred[start_channel: stop_channel, y0: y1, x0: x1] = self._inf_outputs_data[_type][k]['s-pred'][img_num, :, :, :]
                    start_channel = stop_channel
                mask_gt = np.round(mask_gt)
                mask_pred = np.round(mask_pred)
                mask_gt = np.argmax(mask_gt, axis=0)
                mask_pred = np.argmax(mask_pred, axis=0)
                img_out = np.concatenate((img, mask_gt * 255.0 / 10.0, mask_pred * 255.0 / 10.0), axis=1)
                
                time_str = time.strftime("%Y-%m-%d-%X", time.localtime())
                path = os.path.join(out_dir, time_str)
                print(path + '-%d.jpg saved!'%img_num)
                cv2.imwrite(path + '-%d.jpg'%img_num, img_out)
        return

if __name__ == "__main__":
    import argparse
    from meghair.utils import io

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True, default='face:model, eye:model,')
    parser.add_argument('--opr_name', required=True, default='face:name, eye:name')
    parser.add_argument('--img_dir', default='show')
    parser.add_argument('--img_num', default=100)
    args = parser.parse_args()

    # load model
    model_names = args.model_name.split(',')
    opr_names = args.opr_name.split(',')
    nets, oprs = {}, {}
    for name in model_names:
        names = name.split(':')
        nets[names[0]] = io.load_network(names[1])
    for name in opr_names:
        names = name.split(':')
        oprs[names[0]] = names[1]

    # init TestUnit
    t = TestUnit()
    t.gen_kv_dict_all(
        mean_face_file='/unsullied/sharefs/_research_facelm/Isilon-datashare/parsing/extra/mean_helen.pkl',
        val_nori_file='/unsullied/sharefs/_research_facelm/Isilon-datashare/parsing/parsing_helen.info',
        val_keys=['helen_valid'],
        part_shape={'face':(128,128), 'leye':(32,32), 'reye':(32,32), 'nose':(32,32), 'mouth':(32,32)},
    )
    t.gen_inf_outputs_data(
        nets=nets,
        oprs=oprs,
        devices='gpu0',
    )

    t.compute_metric(
        img_dir=args.img_dir,
        img_num=args.img_num,
    )

