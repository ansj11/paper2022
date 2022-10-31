from collections import OrderedDict
import numpy as np
import cv2

import pickle as pkl
import nori2 as nori
from landstack.utils import misc, transforms, augment

class ParsingImage(object):
    def __init__(self, nori_file, keys):
        self.nori_file = nori_file
        self.keys = [keys] if not isinstance(keys, (tuple, list)) else keys

    def load(self, mean_face_file, part_shape):
        print("Loading data from {}\nkeys: {}".format(self.nori_file, ','.join(self.keys)))
        
        mean_helen = misc.load_pickle(mean_face_file)
        mean_face = mean_helen['helen5']*1.1
        aug = augment.AugmentParsing(
            mean_face=mean_face, part_shape=part_shape,
            transforms_post=transforms.Compose([
                transforms.ConvertToGray(),
                transforms.HWC2CHW(),
            ])
        )
        
        fn = nori.Fetcher()
        nori_dict = misc.load_pickle(self.nori_file)
        
        kv_dict_all = OrderedDict()
        for key in self.keys:
            assert key in nori_dict, key
        for key in self.keys:
            selected_nori_id_list = nori_dict[key]
            imgs, masks, lms = {}, {}, []
            for nori_id in selected_nori_id_list:
                r = pkl.loads(fn.get(nori_id))
                lm_raw = np.array(r['ld']).reshape(-1,2)
                img_raw = cv2.imdecode(np.fromstring(r['img'], np.uint8), cv2.IMREAD_COLOR)
                masks_raw = []
                for i in range(len(r['mask'])):
                    masks_raw.append( cv2.imdecode(np.fromstring(r['mask'][i], np.uint8), cv2.IMREAD_GRAYSCALE) )
            
                imgs_aug, masks_aug, lm_aug = aug(img_raw, lm_raw, masks_raw)
   
                for name in imgs_aug:
                    if not name in imgs:
                        imgs[name] = []
                    if not name in masks:
                        masks[name] = []
                    imgs[name].append( imgs_aug[name].astype('uint8') )
                    masks[name].append( masks_aug[name].astype('float32') / 255 )
                lms.append(lm_aug)

            for name in imgs:
                imgs[name] = np.array(imgs[name]).astype('uint8')
                masks[name] = np.array(masks[name]).astype('float32')
            lms = np.array(lms).astype('float32')

            kv_dict = dict(
                imgface=imgs['face'],maskface=masks['face'],
                imgleye=imgs['leye'],maskleye=masks['leye'],
                imgreye=imgs['reye'],maskreye=masks['reye'],
                imgnose=imgs['nose'],masknose=masks['nose'],
                imgmouth=imgs['mouth'],maskmouth=masks['mouth'],
                lm=lms,
            )
            kv_dict_all.setdefault(key, kv_dict)
            
        return kv_dict_all

