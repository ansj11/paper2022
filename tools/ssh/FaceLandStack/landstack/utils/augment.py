import cv2
import numpy as np

from landstack.utils import misc, geom
from IPython import embed


class AugmentBase(object):
    def __init__(self, mean_face, image_shape=(112, 112), transforms_pre=None, transforms_post=None, rng=None):
        """
        :param mean_face:
        :param image_shape:
        :param transforms_pre: for landmark only
        :param transforms_post: for both image and landmark
        :param rng:
        """
        if isinstance(mean_face, str):  # try to load it
            self.mean_face = misc.load_pickle(mean_face).reshape(-1, 2)
        else:
            self.mean_face = mean_face.reshape(-1, 2)
        self.image_shape = image_shape
        self.transforms_pre = transforms_pre
        self.transforms_post = transforms_post
        self.interpolation_methods = [
            cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC]
        self.rng = rng

    def __call__(self, img, lm_ref, lm_gt, **kwargs):
        # pre-process for landmark
        if self.transforms_pre is not None:
            v, _ = self.transforms_pre(img, lm_ref)
            assert len(v) == 2, len(v)
            lm_ref = v[1]

        # align to mean face
        align_func = kwargs.get('align_func', geom.align_to)
        trans_mat_extend = align_func(
            lm_ref, self.mean_face, self.image_shape, extend=True)
        trans_mat = trans_mat_extend[:, :2]
        img = geom.warp_affine(img, trans_mat, self.image_shape)
        # for gray scale image
        img = img[:, :, np.newaxis] if img.ndim == 2 else img
        # please pay attention that this will result in landmark in image_shape scale
        lm_gt = geom.lm_affine(lm_gt, trans_mat)

        # post-process for image and landmark
        v = [img, lm_gt]
        params_dict = dict(
            mat=trans_mat_extend.T
        )
        params_dict.update(kwargs)
        if self.transforms_post is not None:
            v, params_dict = self.transforms_post(*v, **params_dict)
            assert len(v) == 2

        return v[0], v[1], params_dict

class AugmentBaseGeneral(object):
    def __init__(self, mean_face, width_height, transforms_pre=None, transforms_post=None, rng=None):
        assert isinstance(mean_face, np.ndarray), type(mean_face)
        self.mean_face = mean_face.reshape(-1, 2)
        self.width_height = width_height
        self.transforms_pre = transforms_pre
        self.transforms_post = transforms_post
        self.interpolation_methods = [cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC]
        self.rng = rng

    def __call__(self, img, lm_ref, lm_gt, **kwargs):
        img = misc.ensure_hwc(img)

        # pre-process for landmark
        if self.transforms_pre is not None:
            v, _ = self.transforms_pre(img, lm_ref)
            assert len(v) == 2, len(v)
            lm_ref = v[1]

       # align to mean face
        mat = geom.align_to_general(lm_ref, self.mean_face)
        img = cv2.warpPerspective(img, mat, self.width_height)
        lm_gt = geom.lm_affine_general(lm_gt, mat)

        # post-process for image and landmark
        v = [img, lm_gt]
        params_dict = dict(
            mat=mat
        )
        params_dict.update(kwargs)
        if self.transforms_post is not None:
            v, params_dict = self.transforms_post(*v, **params_dict)
            assert len(v) == 2

        return v[0], v[1], params_dict


class Augment3dfawWithNorm(AugmentBase):
    def __init__(self, mean_face, image_shape=(112, 112), transforms_pre=None, transforms_post=None, rng=None):
        super(Augment3dfawWithNorm, self).__init__(
            mean_face, image_shape, transforms_pre, transforms_post, rng)

    def __call__(self, img, lm_ref, lm_gt=None, mat=False, norm_type='pupil'):
        lm_ref = np.array(lm_ref).reshape(-1, 3)
        lm_gt = lm_gt or lm_ref.copy()
        lm_ref = lm_ref[:, :2]

        # pre-process for landmark
        if self.transforms_pre is not None:
            l, _ = self.transforms_pre(img, lm_ref)
            assert len(l) == 2, len(l)
            lm_ref = l[1]

        # align to mean face
        trans_mat_extend = geom.align_to(
            lm_ref, self.mean_face, self.image_shape, extend=True)
        trans_mat = trans_mat_extend[:, :2]
        img = geom.warp_affine(img, trans_mat, self.image_shape)
        # for gray scale image
        img = img[:, :, np.newaxis] if img.ndim == 2 else img
        # please pay attention that this will result in landmark in image_shape scale
        lm_gt = geom.lm_affine_fake3d(lm_gt, trans_mat)

        # post-process for image and landmark
        v = [img, lm_gt]
        params_dict = dict(
            mat=trans_mat_extend.T
        )

        if self.transforms_post is not None:
            v, params_dict = self.transforms_post(*v, **params_dict)

        img_out, lm_out = v
        trans_mat = params_dict['mat']
        #print('img_out', img_out.shape, 'lm_out', lm_out.shape)
        norm = geom.compute_lm_norm(lm_out[:, :2], norm_type, 66)
        #print('img_out', img_out.shape, 'lm_out', lm_out.shape, 'norm', norm)
        res = [img_out, lm_out, norm, trans_mat] if mat else [
            img_out, lm_out, norm]
        return res


class AugmentParsing():
    def __init__(self, mean_face, part_shape, transforms_pre=None, transforms_post=None, rng=None):
        self.mean_face = mean_face.reshape(-1, 2)
        self.part_shape = part_shape
        self.transforms_pre = transforms_pre
        self.transforms_post = transforms_post
        self.rng = rng

    def gethelenpt5(self, pld):
        pld = pld.reshape(-1, 2)
        le = pld[134:154, :].mean(axis=0)
        re = pld[114:134, :].mean(axis=0)
        ne = pld[41:58, :].mean(axis=0)
        lm = pld[58, :]
        rm = pld[71, :]
        return np.concatenate((le, re, ne, lm, rm), axis=0).reshape(-1, 2)

    def croppart(self, img, lm, masks, partshape, parttype):
        outw, outh = partshape[parttype]
        lm = lm.reshape(-1, 2)
        if parttype == 'leye':
            lmp = np.concatenate(
                (lm[134:154, :], lm[174:194, :]), axis=0).reshape(-1, 2)
        elif parttype == 'reye':
            lmp = np.concatenate(
                (lm[114:134, :], lm[154:174, :]), axis=0).reshape(-1, 2)
        elif parttype == 'reye':
            lmp = np.concatenate(
                (lm[114:134, :], lm[154:174, :]), axis=0).reshape(-1, 2)
        elif parttype == 'nose':
            lmp = np.concatenate(
                (lm[114:115, :], lm[134:135, :], lm[41:58, :]), axis=0).reshape(-1, 2)
        elif parttype == 'mouth':
            lmp = lm[58:114, :]

        x0, x1, y0, y1 = lmp[:, 0].min(), lmp[:, 0].max(), lmp[:, 1].min(), lmp[:, 1].max()
        xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
        x0, x1, y0, y1 = int(xc - outw / 2), int(xc + outw / 2), int(yc - outh / 2), int(yc + outh / 2)
        img_out = img[:, y0:y1, x0:x1]

        if parttype == 'leye':
            mask_out = np.concatenate(
                (masks[2][y0:y1, x0:x1][np.newaxis, :, :], masks[4][y0:y1, x0:x1][np.newaxis, :, :]), axis=0)
        elif parttype == 'reye':
            mask_out = np.concatenate(
                (masks[3][y0:y1, x0:x1][np.newaxis, :, :], masks[5][y0:y1, x0:x1][np.newaxis, :, :]), axis=0)
        elif parttype == 'nose':
            mask_out = masks[6][y0:y1, x0:x1][np.newaxis, :, :]
        elif parttype == 'mouth':
            mask_out = np.concatenate((masks[7][y0:y1, x0:x1][np.newaxis, :, :], masks[8][y0:y1, x0:x1][np.newaxis, :, :], masks[9][y0:y1, x0:x1][np.newaxis, :, :]), axis=0)

        return img_out, mask_out

    def __call__(self, img, lm, masks):
        lm = np.array(lm).reshape(-1, 2)
        lm_ref = self.gethelenpt5(lm)

        # pre-process for landmark
        if self.transforms_pre is not None:
            l, _ = self.transforms_pre(img, lm_ref)
            assert len(l) == 2, len(l)
            lm_ref = l[1]

        # align to mean face
        image_shape = self.part_shape['face']
        trans_mat_extend = geom.align_to(
            lm_ref, self.mean_face, image_shape, extend=True)
        trans_mat = trans_mat_extend[:, :2]
        img_out = geom.warp_affine(
            img, trans_mat, image_shape, cv2.BORDER_CONSTANT)
        # for gray scale image
        img_out = img_out[:, :, np.newaxis] if img_out.ndim == 2 else img_out
        # please pay attention that this will result in landmark in image_shape scale
        lm_out = geom.lm_affine(lm, trans_mat)
        masks_out = []
        for i in range(len(masks)):
            masks_out.append(geom.warp_affine(
                masks[i], trans_mat, image_shape, cv2.BORDER_CONSTANT))
        masks_out[0] = geom.warp_affine(
            masks[0], trans_mat, image_shape, cv2.BORDER_CONSTANT, (255, 255, 255))

        if self.transforms_post is not None:
            l, _ = self.transforms_post(img_out, lm_out)
            img_out, lm_out = l[0], l[1]
        #print(img_out.shape, lm_out.shape)

        # crop each part
        imgs, masks = {}, {}
        imgs['face'] = img_out.copy()
        masks['face'] = np.concatenate(
            (masks_out[0][np.newaxis, :, :], masks_out[1][np.newaxis, :, :], masks_out[10][np.newaxis, :, :]), axis=0)
        for name in ['leye', 'reye', 'nose', 'mouth']:
            imgs[name], masks[name] = self.croppart(
                img_out, lm_out, masks_out, self.part_shape, name)

        return imgs, masks, lm_out


class Augment81WithNormCen(AugmentBase):
    def __init__(self, mean_face, image_shape=(112, 112), transforms_pre=None, transforms_post=None, rng=None):
        super(Augment81WithNormCen, self).__init__(
            mean_face, image_shape, transforms_pre, transforms_post, rng)

    def __call__(self, img, lm_ref, lm_gt=None, mat=False, norm_type='geom'):
        lm_gt = lm_gt or lm_ref.copy()
        lm_ref = np.array(lm_ref).reshape(-1, 2)
        lm_gt = np.array(lm_gt).reshape(-1, 2)

        # pre-process for landmark
        if self.transforms_pre is not None:
            lm_ref = self.transforms_pre(lm_ref)

        # align to mean face
        trans_mat_extend = geom.align_to_cen(
            lm_ref, self.mean_face, self.image_shape, extend=True)
        trans_mat = trans_mat_extend[:, :2]
        interpolation_method = self.rng.choice(
            self.interpolation_methods) if self.rng else self.interpolation_methods[0]
        img = geom.warp_affine(img, trans_mat, self.image_shape)
        #img = cv2.warpAffine(img, trans_mat.T, self.image_shape, borderMode=cv2.BORDER_REPLICATE, flags=interpolation_method)
        # for gray scale image
        img = img[:, :, np.newaxis] if img.ndim == 2 else img
        # please pay attention that this will result in landmark in image_shape scale
        lm_gt = geom.lm_affine(lm_gt, trans_mat)

        # post-process for image and landmark
        v = [img, lm_gt]
        if self.transforms_post is not None:
            v = self.transforms_post(*v)

        v = list(v)
        v.append(trans_mat_extend)
        img_out, lm_out, trans_mat = v
        norm = geom.compute_lm_norm(lm_out, norm_type)
        lm_out = lm_out.reshape(-1)
        res = [img_out, lm_out, norm, trans_mat] if mat else [
            img_out, lm_out, norm]
        return res


class AugmentDenseWithNorm(AugmentBase):
    def __init__(self, mean_face, image_shape=(112, 112), transforms_pre=None, transforms_post=None, rng=None):
        super(AugmentDenseWithNorm, self).__init__(
            mean_face, image_shape, transforms_pre, transforms_post, rng)

    def extend(self, ld):
        ld = ld.reshape(-1, 2)
        ld0 = ld[:124, :]
        leye = np.concatenate((ld[39:48, :], ld[124:148, :]), axis=0)
        reye = np.concatenate((ld[48:57, :], ld[148:172, :]), axis=0)
        mouth = np.concatenate((ld[72:90, :], ld[172:228, :]), axis=0)
        ldout = np.concatenate((ld0, leye, reye, mouth), axis=0)
        return ldout.reshape(-1)

    def __call__(self, img, lm_ref, lm_gt=None, mat=False, norm_type='geom'):
        lm_ref = np.array(lm_ref).reshape(-1, 2)
        lm_gt = np.array(lm_gt).reshape(-1, 2)

        # pre-process for landmark
        if self.transforms_pre is not None:
            l, _ = self.transforms_pre(img, lm_ref)
            assert len(l) == 2, len(l)
            lm_ref = l[1]

        # align to mean face
        trans_mat_extend = geom.align_to(
            lm_ref, self.mean_face, self.image_shape, extend=True)
        trans_mat = trans_mat_extend[:, :2]
        img = geom.warp_affine(img, trans_mat, self.image_shape)
        # for gray scale image
        img = img[:, :, np.newaxis] if img.ndim == 2 else img
        # please pay attention that this will result in landmark in image_shape scale
        lm_gt = geom.lm_affine(lm_gt, trans_mat)

        # post-process for image and landmark
        v = [img, lm_gt]
        params_dict = dict(
            mat=trans_mat_extend.T
        )
        if self.transforms_post is not None:
            v, params_dict = self.transforms_post(*v, **params_dict)

        img_out, lm_out = v
        #print('img_out', img_out.shape, 'lm_out', lm_out.shape)
        trans_mat = params_dict['mat']
        norm = geom.compute_lm_norm(lm_out, norm_type, 228)
        lm_out = self.extend(lm_out)
        #print('img_out', img_out.shape, 'lm_out', lm_out.shape, 'norm', norm)
        res = [img_out, lm_out, norm, trans_mat] if mat else [
            img_out, lm_out, norm]
        return res


class AugmentGaze(object):
    def __init__(self, mean_face, width_height=(48, 32), scale=1.0, transforms_pre=None, transforms_post=None, rng=None):
        assert not isinstance(mean_face, str)
        self.mean_eye = self.gen_mean_eye(mean_face, scale, width_height)
        self.width_height= width_height
        self.transforms_pre = transforms_pre
        self.transforms_post = transforms_post
        self.interpolation_methods = [cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC]
        self.rng = rng

    def __call__(self, img, lm_ref, lm_gt, ratio0=16, **kwargs):
        from benchmark.dataset.gaze import decode, encode
        img = misc.ensure_hwc(img)
        interior_margin = decode(lm_ref)['interior_margin']

        # pre-process for landmark
        if self.transforms_pre is not None:
            v, _ = self.transforms_pre(img, interior_margin)
            assert len(v) == 2, len(v)
            interior_margin = v[1]

        # align to mean face
        mat = geom.align_to_general(interior_margin, self.mean_eye)
        img = cv2.warpPerspective(img, mat, self.width_height)
        lm_gt = geom.lm_affine_general(lm_gt, mat)
        r = decode(lm_gt)
        eye_center = r['eye_center']
        gaze_point = r['gaze_point']

        # modify gaze point
        s = np.sum(mat[0, :2]**2) # modified scale
        r0 = ratio0 / 80 / s # accommodate vis
        t0 = (1 - r0) * eye_center
        m0 = np.float32([
            [r0, 0,  t0[0]],
            [0,  r0, t0[1]],
            [0,  0,  1]
        ])
        r['gaze_point'] = geom.lm_affine_general(gaze_point, m0) # gp_new = ec + (gp_old - ec) * r0
        mat_gp = np.dot(m0, mat)

        lm_gt = encode(r)
        v = [img, lm_gt]
        params_dict = dict(
            mat=mat,
            mat_gp=mat_gp
        )
        params_dict.update(**kwargs)

        if self.transforms_post is not None:
            v, params_dict = self.transforms_post(*v, **params_dict)
            assert len(v) == 2
        return v[0], v[1], params_dict


    @staticmethod
    def gen_mean_eye(lm_ref, scale, width_height):
        lm_ref = lm_ref.reshape(-1, 2)
        mi = lm_ref.min(axis = 0)
        ma = lm_ref.max(axis = 0)
        mid = (mi + ma)/2
        s = (ma - mi).max() * scale
        a = mid - s/2.
        lm_ref = (lm_ref - a)/s * np.float32(width_height).reshape(-1, 2)
        return lm_ref
