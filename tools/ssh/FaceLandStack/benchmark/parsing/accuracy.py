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

    @staticmethod
    def p_r_f_a_compute(mask_gt, mask_pred):
        mask_gt = np.round(mask_gt)
        mask_pred = np.round(mask_pred)
        # plt.subplot(121)
        # plt.imshow(mask_gt)
        # plt.title('gt')
        # plt.subplot(122)
        # plt.imshow(mask_pred)
        # plt.title('pred')
        # plt.show()
        l = mask_gt * 10
        total = l + mask_pred
        TP = np.size(np.where(total == 11)[0], axis=0)
        FP = np.size(np.where(total == 1)[0], axis=0)
        FN = np.size(np.where(total == 10)[0], axis=0)
        TN = np.size(np.where(total == 0)[0], axis=0)
        print('TP: %2.2f | FP: %2.2f | FN: %2.2f | TN: %2.2f | ALL: %2.2f'%(TP, FP, FN, TN, TP + FP + FN + TN))

        p = TP / (TP + FP + 1e-10)
        r = TP / (TP + FN + 1e-10)
        f = p * r * 2 / (p + r + 1e-10)
        return p, r, f

    def compute_metric(self, img_dir, img_num):
        def mean(l):
            return sum(l) / len(l)

        assert self._kv_dict_all and self._inf_outputs_data, 'Please gen kv_dict_all and inf_outputs_data first'
        precision = {}
        recall = {}
        F_value = {}
        accuracy = {}
        accuracy_3 = {}
        accuracy_11 = {}
        for k in self._kv_dict_all.keys():
            out_dir = os.path.join(img_dir, k)
            misc.ensure_dir(out_dir)
            part_shape={'face':(128,128), 'leye':(32,32), 'reye':(32,32), 'nose':(32,32), 'mouth':(32,32)}
            part_type = ['leye', 'reye', 'nose', 'mouth']
            all_num = self._kv_dict_all[k]['imgface'].shape[0]
            precision[k] = {} 
            precision[k]['face'] = [0] * 100
            precision[k]['hair'] = [0] * 100
            precision[k]['back'] = [0] * 100
            precision[k]['leye'] = [0] * 100
            precision[k]['leyebrow'] = [0] * 100
            precision[k]['reye'] = [0] * 100
            precision[k]['reyebrow'] = [0] * 100
            precision[k]['nose'] = [0] * 100
            precision[k]['mouthup'] = [0] * 100
            precision[k]['mouthdown'] = [0] * 100
            precision[k]['mouthin'] = [0] * 100
            precision[k]['3_type'] = [0] * 100
            recall[k] = {}
            recall[k]['face'] = [0] * 100
            recall[k]['hair'] = [0] * 100
            recall[k]['back'] = [0] * 100
            recall[k]['leye'] = [0] * 100
            recall[k]['reye'] = [0] * 100
            recall[k]['leyebrow'] = [0] * 100
            recall[k]['reyebrow'] = [0] * 100
            recall[k]['nose'] = [0] * 100
            recall[k]['mouthup'] = [0] * 100
            recall[k]['mouthin'] = [0] * 100
            recall[k]['mouthdown'] = [0] * 100
            recall[k]['3_type'] = [0] * 100
            F_value[k] = {}
            F_value[k]['face'] = [0] * 100
            F_value[k]['hair'] = [0] * 100
            F_value[k]['back'] = [0] * 100
            F_value[k]['leye'] = [0] * 100
            F_value[k]['reye'] = [0] * 100
            F_value[k]['leyebrow'] = [0] * 100
            F_value[k]['reyebrow'] = [0] * 100
            F_value[k]['nose'] = [0] * 100
            F_value[k]['mouthup'] = [0] * 100
            F_value[k]['mouthin'] = [0] * 100
            F_value[k]['mouthdown'] = [0] * 100
            F_value[k]['3_type'] = [0] * 100

            for img_num in range(self._kv_dict_all[k]['imgface'].shape[0]):
                img = self._kv_dict_all[k]['imgface'][img_num, :, :, :]
                lm = self._kv_dict_all[k]['lm'][img_num, :, :]
                mask_gt = np.zeros([11, img.shape[1], img.shape[2]])
                mask_pred = np.zeros([11, img.shape[1], img.shape[2]])
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
                precision[k]['back'][img_num], recall[k]['back'][img_num], F_value[k]['back'][img_num] = self.p_r_f_a_compute(mask_gt[0, :, :], mask_pred[0, :, :])
                precision[k]['face'][img_num], recall[k]['face'][img_num], F_value[k]['face'][img_num] = self.p_r_f_a_compute(mask_gt[1, :, :], mask_pred[1, :, :])
                precision[k]['hair'][img_num], recall[k]['hair'][img_num], F_value[k]['hair'][img_num] = self.p_r_f_a_compute(mask_gt[2, :, :], mask_pred[2, :, :])
                precision[k]['leyebrow'][img_num], recall[k]['leyebrow'][img_num], F_value[k]['leyebrow'][img_num] = self.p_r_f_a_compute(mask_gt[3, :, :], mask_pred[3, :, :])
                precision[k]['leye'][img_num], recall[k]['leye'][img_num], F_value[k]['leye'][img_num] = self.p_r_f_a_compute(mask_gt[4, :, :], mask_pred[4, :, :])
                precision[k]['reyebrow'][img_num], recall[k]['reyebrow'][img_num], F_value[k]['reyebrow'][img_num] = self.p_r_f_a_compute(mask_gt[5, :, :], mask_pred[5, :, :])
                precision[k]['reye'][img_num], recall[k]['reye'][img_num], F_value[k]['reye'][img_num] = self.p_r_f_a_compute(mask_gt[6, :, :], mask_pred[6, :, :])
                precision[k]['nose'][img_num], recall[k]['nose'][img_num], F_value[k]['nose'][img_num] = self.p_r_f_a_compute(mask_gt[7, :, :], mask_pred[7, :, :])
                precision[k]['mouthup'][img_num], recall[k]['mouthup'][img_num], F_value[k]['mouthup'][img_num] = self.p_r_f_a_compute(mask_gt[8, :, :], mask_pred[8, :, :])
                precision[k]['mouthin'][img_num], recall[k]['mouthin'][img_num], F_value[k]['mouthin'][img_num] = self.p_r_f_a_compute(mask_gt[9, :, :], mask_pred[9, :, :])
                precision[k]['mouthdown'][img_num], recall[k]['mouthdown'][img_num], F_value[k]['mouthdown'][img_num] = self.p_r_f_a_compute(mask_gt[10, :, :], mask_pred[10, :, :])
                precision[k]['3_type'][img_num], recall[k]['3_type'][img_num], F_value[k]['3_type'][img_num] = self.p_r_f_a_compute(np.concatenate((mask_gt[0, :, :], mask_gt[1, :, :], mask_gt[2, :, :]), axis=0), np.concatenate((mask_pred[0, :, :], mask_pred[1, :, :], mask_pred[2, :, :]), axis=0))
                mask_gt = np.round(mask_gt)
                mask_pred = np.round(mask_pred)
                mask_gt = np.argmax(mask_gt, axis=0)
                mask_pred = np.argmax(mask_pred, axis=0)

            print('===================== %s ==================='%(k))
            print('name  |  back  |  hair  |  face  |leyebrow|  leye  |reyebrow|  reye  |  nose  |mouthup |mouthin |mouthdown|  3_type')  
            print('%4s  |  %2.1f  |  %2.1f  |  %2.1f  |  %2.1f  |  %2.1f  |  %2.1f  |  %2.1f  |  %2.1f  |  %2.1f  |  %2.1f  |  %2.1f  |  %2.1f'%('prec', 100.0 * mean(precision[k]['back']), 100.0 * mean(precision[k]['hair']), 100.0 * mean(precision[k]['face']), 100.0 * mean(precision[k]['leyebrow']), 100.0 * mean(precision[k]['leye']), 100.0 * mean(precision[k]['reyebrow']), 100.0 * mean(precision[k]['reye']), 100.0 * mean(precision[k]['nose']), 100.0 * mean(precision[k]['mouthup']), 100.0 * mean(precision[k]['mouthin']), 100.0 * mean(precision[k]['mouthdown']), 100.0 * mean(precision[k]['3_type'])))
            print('%4s  |  %2.1f  |  %2.1f  |  %2.1f  |  %2.1f  |  %2.1f  |  %2.1f  |  %2.1f  |  %2.1f  |  %2.1f  |  %2.1f  |  %2.1f  |  %2.1f'%('reca', 100.0 * mean(recall[k]['back']), 100.0 * mean(recall[k]['hair']), 100.0 * mean(recall[k]['face']), 100.0 * mean(recall[k]['leyebrow']), 100.0 * mean(recall[k]['leye']), 100.0 * mean(recall[k]['reyebrow']), 100.0 * mean(recall[k]['reye']), 100.0 * mean(recall[k]['nose']), 100.0 * mean(recall[k]['mouthup']), 100.0 * mean(recall[k]['mouthin']), 100.0 * mean(recall[k]['mouthdown']), 100.0 * mean(recall[k]['3_type'])))
            print('%4s  |  %2.1f  |  %2.1f  |  %2.1f  |  %2.1f  |  %2.1f  |  %2.1f  |  %2.1f  |  %2.1f  |  %2.1f  |  %2.1f  |  %2.1f  |  %2.1f'%('f_va', 100.0 * mean(F_value[k]['back']), 100.0 * mean(F_value[k]['hair']), 100.0 * mean(F_value[k]['face']), 100.0 * mean(F_value[k]['leyebrow']), 100.0 * mean(F_value[k]['leye']), 100.0 * mean(F_value[k]['reyebrow']), 100.0 * mean(F_value[k]['reye']), 100.0 * mean(F_value[k]['nose']), 100.0 * mean(F_value[k]['mouthup']), 100.0 * mean(F_value[k]['mouthin']), 100.0 * mean(F_value[k]['mouthdown']), 100.0 * mean(F_value[k]['3_type'])))
                # img_out = np.concatenate((img, mask_gt * 255.0 / 10.0, mask_pred * 255.0 / 10.0), axis=1)
                
                # time_str = time.strftime("%Y-%m-%d-%X", time.localtime())
                # path = os.path.join(out_dir, time_str)
                # print(path + '-%d.jpg saved!'%img_num)
                # cv2.imwrite(path + '-%d.jpg'%img_num, img_out)
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

