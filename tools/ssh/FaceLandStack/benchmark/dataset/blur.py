from collections import OrderedDict
import numpy as np
from landstack.utils import misc
import tqdm
import cv2
from IPython import embed


class BlurrinessImage(object):
    """blurriness"""
    def __init__(self, nori_file=None, keys=None):
        self.nori_file = nori_file or '/unsullied/sharefs/_research_facelm/Isilon-datashare/proj/blur/landmark_blur_val_171028.info'
        self.keys = keys

    def load(self, image_shape=(112, 112), is_color=False, n_samples=None, legacy=False):
        ds_kv_dict = misc.load_pickle(self.nori_file)
        self.keys  = self.keys or list(ds_kv_dict.keys())
        assert isinstance(self.keys, (tuple, list))
        for key in self.keys:
            assert key in ds_kv_dict, key
        print("Loading data from {}\nkeys: {}".format(self.nori_file, ','.join(self.keys)))
        kv_dict_all = OrderedDict()
        for key in self.keys:
            tuple5_list = ds_kv_dict[key]
            n_samples0 = min(n_samples or len(tuple5_list), len(tuple5_list))
            print("\tFound {}/{} samples to load from {}".format(n_samples0, len(tuple5_list), key))
            img_kv_dict, img_kv_dict_legacy = OrderedDict(), OrderedDict()
            tuple5_encode_list_all, tuple5_encode_list_all_legacy = [], []
            pair_encode_list_all, pair_encode_list_all_legacy = [], []
            for tuple5 in tqdm.tqdm(tuple5_list[:n_samples0]):
                pair_encode_list, pair_encode_list_legacy = [], []
                tuple5_encode_list, tuple5_encode_list_legacy = [], []
                levels = [v['level'] for v in tuple5]
                for i in range(1, len(levels)):
                    assert levels[i-1] <= levels[i], '{} vs {}'.format(levels[i-1], levels[i])
                for i in range(len(tuple5)-1):
                    for j in range(i+1, len(tuple5)):
                        if tuple5[i]['level'] == tuple5[j]['level']:
                            continue
                        elif tuple5[i]['category'] == 1 or tuple5[j]['category'] == 1:
                            continue
                        else:
                            new_pair = []
                            for data_id in (tuple5[i]['data_id'], tuple5[j]['data_id']):
                                if data_id not in img_kv_dict:
                                    img = misc.unpack_lm(data_id, is_color=True)[0]
                                    img = cv2.resize(img, image_shape)
                                    if not is_color:
                                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
                                    img_kv_dict[data_id] = [len(img_kv_dict), img] # [idx, img]
                                new_pair.append(img_kv_dict[data_id][0])
                            pair_encode_list.append(new_pair)

                            if legacy:
                                new_pair_legacy = []
                                for data_id in (tuple5[i]['data_id_legacy'], tuple5[j]['data_id_legacy']):
                                    if data_id not in img_kv_dict_legacy:
                                        img = cv2.resize(misc.unpack_img(data_id, is_color=True), image_shape)
                                        if not is_color:
                                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
                                        img_kv_dict_legacy[data_id] = [len(img_kv_dict_legacy), img] # [idx, img]
                                    new_pair_legacy.append(img_kv_dict_legacy[data_id][0])
                                pair_encode_list_legacy.append(new_pair_legacy)

                for i in range(len(tuple5)):
                    tuple5_encode_list.append(img_kv_dict[tuple5[i]['data_id']][0])
                pair_encode_list_all.append(pair_encode_list)
                tuple5_encode_list_all.append(tuple5_encode_list)

                if legacy:
                    for i in range(len(tuple5)):
                        tuple5_encode_list_legacy.append(img_kv_dict_legacy[tuple5[i]['data_id_legacy']][0])
                    pair_encode_list_all_legacy.append(pair_encode_list_legacy)
                    tuple5_encode_list_all_legacy.append(tuple5_encode_list_legacy)

            img_batch = np.uint8([v[1] for v in img_kv_dict.values()])
            kv_dict_all[key] = dict(
                img = img_batch.transpose(0, 3, 1, 2),
                pairs = pair_encode_list_all,
                tuple5 = tuple5_encode_list_all,
            )
            if legacy:
                img_batch_legacy = np.uint8([v[1] for v in img_kv_dict_legacy.values()])
                kv_dict_all[key+'-legacy'] = dict(
                    img = img_batch_legacy.transpose(0, 3, 1, 2),
                    pairs = pair_encode_list_all_legacy,
                    tuple5 = tuple5_encode_list_all_legacy,
                )

        return kv_dict_all


if __name__ == '__main__':
    from IPython import embed
    from landstack.utils.visualization import draw
    ds = BlurrinessImage()
    kv_dict_all = ds.load(n_samples=100)
    for ds_name, kv_dict in kv_dict_all.items():
        img = kv_dict['img']
        pair_list = kv_dict['pairs']
        for tuple5 in pair_list:
            for idx1, idx2 in tuple5:
                img1 = img[idx1]
                img2 = img[idx2]
                draw(img1, name='img1', wait=False)
                draw(img2, name='img2')


