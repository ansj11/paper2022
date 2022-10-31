from collections import OrderedDict
from landstack.utils import misc
from landstack.utils import transforms
import numpy as np
import cv2
def encode(postfilter):
    if(postfilter=='face'):
        return 1.0
    else:
        return 0.0

class PostImage(object):
    def __init__(self, nori_file=None, keys=None):
        self.nori_file = nori_file or '/unsullied/sharefs/_research_facelm/Isilon-datashare/proj/postfilter/landmark_postfilter_val_171204.info'
        self.keys = keys

    def load(self, is_color=False, n_samples=None):
        ds_kv_dict = misc.load_pickle(self.nori_file)
        self.keys  = self.keys or list(ds_kv_dict.keys())
        assert isinstance(self.keys, (tuple, list))
        for key in self.keys:
            assert key in ds_kv_dict, key

        print("Loading data from {}\nKeys: {}".format(self.nori_file, ','.join(self.keys)))
        kv_dict_all = OrderedDict()
        aug = transforms.HWC2CHW()
        for key in self.keys:
            nori_kv_dict = ds_kv_dict[key]
            n_samples0 = min(n_samples or len(nori_kv_dict), len(nori_kv_dict))
            print("\tFound {}/{} samples to load from {}".format(n_samples0, len(nori_kv_dict), key))
            img_out_batch, postfilter_batch, nori_id_batch = [], [], []
            for nori_id, postfilter in list(nori_kv_dict.items())[:n_samples0]:
                img_raw, _, _ = misc.unpack_lm(nori_id, is_color=is_color)
                img_out = aug(img_raw, None)[0][0]
                #print(nori_id, postfilter, img_out.shape)
                img_out_batch.append(img_out)
                postfilter_batch.append(encode(postfilter))
                nori_id_batch.append(nori_id)
            kv_dict = dict(
                img = np.uint8(img_out_batch),
                postfilter = np.float32(postfilter_batch).flatten(),
                nori_id =  nori_id_batch,
            )
            kv_dict_all.setdefault(key, kv_dict)

        return kv_dict_all


if __name__ == '__main__':
    from landstack.utils.visualization import draw
    from IPython import embed
    ds = PostImage(keys=['black-valid', 'down-valid'])
    kv_dict_all = ds.load()
    for ds_name, kv_dict in kv_dict_all.items():
        img = kv_dict['img']
        postfilter = kv_dict['postfilter']
        for img0, mask0 in zip(img, postfilter):
            draw(img0, text=str(int(mask0)))

