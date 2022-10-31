import cv2
import numpy as np

from landstack.utils import misc, geom
from IPython import embed
import itertools

class Compose(object):
    """
    Compose several transforms together
    Case:
        transforms.Compose([
            transforms.RandomHorizontalFlip(flip_idx=flip_idx, rng=rng),
            transforms.Gaussian(rng=rng),
            transforms.ConvertToGray(),
            transforms.HWC2CHW(),
            transforms.NormalizeLandmark(image_shape=config.image_shape),
        ])
    """
    def __init__(self, transforms):
        assert isinstance(transforms, (tuple, list)), type(transforms)
        self.transforms = transforms

    def __call__(self, *args, **kwargs):
        for t in self.transforms:
            v = t(*args, **kwargs)
            assert len(v) == 2, len(v)
            assert isinstance(v[0], (tuple, list)), type(args)
            assert isinstance(v[1], dict), type(kwargs)
            args = v[0]
            kwargs = v[1]
        return list(args), kwargs

class JitterPts(object):
    def __init__(self, trans, rng, isnorm=False):
        self.trans = np.float32(trans)
        self.rng = rng or np.random
        self.isnorm = isnorm

    def __call__(self, img, lm, **kwargs):
        lm_out = lm.reshape(-1,2)
        n_points = lm_out.shape[0]
        if self.isnorm:
            eyedis = geom.compute_lm_norm(lm_out, norm_type='pupil', n_points=n_points)
            scale = eyedis / (0.35*112)
        else:
            scale = 1
        select_idx = self.rng.randint(len(self.trans), size=n_points*2)
        trans_range = np.array(self.trans[select_idx]).reshape(-1, 2)
        trans = self.rng.uniform(low=trans_range[:, 0], high=trans_range[:, 1]).reshape(n_points, 2) * scale
        lm_out += trans
        return [img, lm_out], kwargs

class JitterNormal(object):
    def __init__(self, trans, rng, isnorm=False):
        self.trans = np.float32(trans)
        self.rng = rng or np.random
        self.isnorm = isnorm
        self.neighbours = np.int32([misc.getneib81(i) for i in range(81)]).flatten()

    def __call__(self, img, lm, **kwargs):
        lm_out = lm.reshape(-1,2)
        n_points = lm_out.shape[0]
        assert n_points == 81, "{} vs. 81".format(n_points)
        if self.isnorm:
            eyedis = geom.compute_lm_norm(lm_out, norm_type='pupil', n_points=n_points)
            scale = eyedis / (0.35*112)
        else:
            scale = 1
        pts = lm_out[self.neighbours].reshape(-1, 2, 2)
        dxy = pts[:, 1, :] - pts[:, 0, :]
        t = np.sqrt((dxy**2).sum(axis=1)).reshape(-1, 1)
        select_idx = self.rng.randint(len(self.trans), size=n_points)
        trans_range = np.array(self.trans[select_idx]).reshape(-1, 2)
        trans = self.rng.uniform(low=trans_range[:, 0], high=trans_range[:, 1]).reshape(-1, 1) * scale
        lm_out += trans * dxy / (t + 1e-7)

        return [img, lm_out], kwargs

class JitterConstant(object):
    def __init__(self, trans, rng, isnorm=False):
        self.trans = np.float32(trans)
        self.rng = rng or np.random
        self.isnorm = isnorm
        # parts = [(1,10), (10,19), (19, 28), (28, 36), (36, 54), (54, 64), (64, 73), (73, 81)]
        self.parts = [1, 9, 9, 9, 8, 18, 10, 9, 8] #l-cont, r-cont, l-eye, l-eyebrow, mouth, nose, r-eye, r-eyebrow
        self.indices = list(itertools.chain.from_iterable([[i] * self.parts[i] for i in range(len(self.parts))]))

    def __call__(self, img, lm, **kwargs):
        lm_out = lm.reshape(-1,2)
        n_points = lm_out.shape[0]
        if self.isnorm:
            eyedis = geom.compute_lm_norm(lm_out, norm_type='pupil', n_points=n_points)
            scale = eyedis / (0.35*112)
        else:
            scale = 1

        select_idx = self.rng.randint(len(self.trans), size=len(self.parts)*2)
        trans_range = np.array(self.trans[select_idx]).reshape(-1, 2)
        trans = self.rng.uniform(low=trans_range[:, 0], high=trans_range[:, 1]).reshape(len(self.parts), 2) * scale
        trans = trans[self.indices]
        lm_out += trans

        return [img, lm_out], kwargs


# for landmark only
class Jitter(object):
    """
    Add affine transform noise to lm
    Case:
    transforms.Jitter(
        scale=[(0.95, 1.05)],
        rotate=[(-5, 5)],
        trans_x=[(-1, -0.15), (0.15, 1)], # ring like
        trans_y=[(-1, -0.15), (0.15, 1)], # ring like
        rng=rng
    )
    """
    def __init__(self, scale, rotate, trans_x, trans_y, rng=None):
        self.scale = scale
        self.rotate = rotate
        self.trans_x = trans_x
        self.trans_y = trans_y
        self.rng = rng

    def __call__(self, img, lm, **kwargs):
        if self.rng is not None:
            scale_range = self.scale[self.rng.randint(len(self.scale))]
            scale = self.rng.uniform(*scale_range)
            rotate_range = self.rotate[self.rng.randint(len(self.rotate))]
            rotate = self.rng.uniform(*rotate_range)
            trans_x_range = self.trans_x[self.rng.randint(len(self.trans_x))]
            trans_x = self.rng.uniform(*trans_x_range)
            trans_y_range = self.trans_y[self.rng.randint(len(self.trans_y))]
            trans_y = self.rng.uniform(*trans_y_range)
        else:
            scale = self.scale
            rotate = self.rotate
            trans_x = self.trans_x
            trans_y = self.trans_y


        lm = lm.reshape(-1, 2)
        transform = cv2.getRotationMatrix2D(tuple(lm.mean(axis=0)), rotate, scale)

        s = lm.max(axis=0) - lm.min(axis=0)  # max length
        translate = np.array([trans_x, trans_y])
        transform[:, 2] += translate * s

        cur_extend = np.concatenate([lm, np.ones((lm.shape[0], 1))], axis=1)
        lm_out = np.dot(cur_extend, transform.T)
        return [img, lm_out], kwargs

class JitterNeg0(object):
    """
    Add affine transform noise to lm
    Case:
    transforms.Jitter(
        scale=[(0.95, 1.05)],
        rotate=[(-5, 5)],
        negths=0.5
        rng=rng
    )
    """
    def __init__(self, scale, rotate, negths, rng):
        self.scale = scale
        self.rotate = rotate
        self.negths = negths
        self.rng = rng or np.random

    def __call__(self, img, lm, **kwargs):
        lm = lm.reshape(-1, 2)
        s = lm.max(axis=0) - lm.min(axis=0)  # max length

        scale_range = self.scale[self.rng.randint(len(self.scale))]
        scale = self.rng.uniform(*scale_range)
        rotate_range = self.rotate[self.rng.randint(len(self.rotate))]
        rotate = self.rng.uniform(*rotate_range)
        transform = cv2.getRotationMatrix2D(tuple(lm.mean(axis=0)), rotate, scale)

        while True:
            base = self.rng.rand(2)
            if np.sum(base) > self.negths:
                break
        translate = (self.rng.rand(2) * 0.5 + base) * (2 * self.rng.randint(2, size=[2]) - 1)

        transform[:, 2] += translate * s

        cur_extend = np.concatenate([lm, np.ones((lm.shape[0], 1))], axis=1)
        lm_out = np.dot(cur_extend, transform.T)
        return [img, lm_out], kwargs

class JitterNeg1(object):
    """
    Add affine transform noise to lm
    Case:
    transforms.Jitter(
        scale0=[(0.95, 1.05)],
        scale1=[(0.2, 0.7), (1.3, 1.8)],
        rotate=[(-5, 5)],
        trans0=[(-1, -0.15), (0.15, 1)],
        trans1=[(-0.1,0.1)],
        negths=0.5,
        rng=rng
    )
    """
    def __init__(self, scale0, scale1, rotate, trans0, trans1, negths, rng):
        self.scale0 = scale0
        self.scale1 = scale1
        self.rotate = rotate
        self.trans0 = trans0
        self.trans1 = trans1
        self.negths = negths
        self.rng = rng or np.random

    def __call__(self, img, lm, **kwargs):
        lm = lm.reshape(-1, 2)
        s = lm.max(axis=0) - lm.min(axis=0)  # max length

        if self.rng.rand()<0.6:                 # small resize
            scale_ = self.scale0
            trans_ = self.trans0

            scale_range = scale_[self.rng.randint(len(scale_))]
            scale = self.rng.uniform(*scale_range)
            rotate_range = self.rotate[self.rng.randint(len(self.rotate))]
            rotate = self.rng.uniform(*rotate_range)
            transform = cv2.getRotationMatrix2D(tuple(lm.mean(axis=0)), rotate, scale)

            if self.rng.rand()<0.3:             # neighbors
                trans_x_range = trans_[self.rng.randint(len(trans_))]
                trans_x = self.rng.uniform(*trans_x_range)
                trans_y_range = trans_[self.rng.randint(len(trans_))]
                trans_y = self.rng.uniform(*trans_y_range)
                translate = np.array([trans_x, trans_y])
            else:                               # bgs
                while True:
                    base = self.rng.rand(2)
                    if np.sum(base) > self.negths:
                        break
                translate = (self.rng.rand(2) * 0.5 + base) * (2 * self.rng.randint(2, size=[2]) - 1)
        else:                                    # big size
            scale_ = self.scale1
            trans_ = self.trans1
   
            scale_range = scale_[self.rng.randint(len(scale_))]
            scale = self.rng.uniform(*scale_range)
            rotate_range = self.rotate[self.rng.randint(len(self.rotate))]
            rotate = self.rng.uniform(*rotate_range)
            transform = cv2.getRotationMatrix2D(tuple(lm.mean(axis=0)), rotate, scale)
   
            trans_x_range = trans_[self.rng.randint(len(trans_))]
            trans_x = self.rng.uniform(*trans_x_range)
            trans_y_range = trans_[self.rng.randint(len(trans_))]
            trans_y = self.rng.uniform(*trans_y_range)
            translate = np.array([trans_x, trans_y])

        transform[:, 2] += translate * s

        cur_extend = np.concatenate([lm, np.ones((lm.shape[0], 1))], axis=1)
        lm_out = np.dot(cur_extend, transform.T)
        return [img, lm_out], kwargs

class JitterNeg2(object):
    """
    Add affine transform noise to lm
    Case:
    transforms.Jitter(
        scale=[(0.5, 1.5)],
        rotate=[(-5, 5)],
        trans=[(-1.5,1.5)],
        overlap=0.5,
        rng=rng
    )
    """

    def __init__(self, scale, rotate, trans, overlap, rng):
        self.overlap = overlap
        self.scale = scale
        self.rotate = rotate
        self.trans = trans
        self.rng = rng or np.random
    
    def __call__(self, img, lm, **kwargs):
        lm = lm.reshape(-1, 2)
        s = lm.max(axis=0) - lm.min(axis=0)  # max length

        x0,x1,y0,y1 = lm[:,0].min(), lm[:,0].max(), lm[:,1].min(), lm[:,1].max()
        ss = (y1-y0) * (x1-x0)

        scale_ = self.scale
        rotate_ = self.rotate
        trans_ = self.trans

        while True:
            scale_range = scale_[self.rng.randint(len(scale_))]
            scale = self.rng.uniform(*scale_range)
            rotate_range = self.rotate[self.rng.randint(len(rotate_))]
            rotate = self.rng.uniform(*rotate_range)
            transform = cv2.getRotationMatrix2D(tuple(lm.mean(axis=0)), rotate, scale)
            
            trans_x_range = trans_[self.rng.randint(len(trans_))]
            trans_x = self.rng.uniform(*trans_x_range)
            trans_y_range = trans_[self.rng.randint(len(trans_))]
            trans_y = self.rng.uniform(*trans_y_range)
            translate = np.array([trans_x, trans_y])
            
            transform[:, 2] += translate * s
            cur_extend = np.concatenate([lm, np.ones((lm.shape[0], 1))], axis=1)
            lm_out = np.dot(cur_extend, transform.T)
            lm_out = lm_out.reshape(-1,2)

            x2,y2,x3,y3 = lm_out[:,0].min(), lm_out[:,0].max(), lm_out[:,1].min(), lm_out[:,1].max()
            bx, by, ex, ey = max(x0,x2), max(y0,y2), min(x1,x3), min(y1,y3)
            if bx>ex or by>ey:
                break
            olp1 = ((ex-bx)*(ey-by)) / ((y1-y0)*(x1-x0))
            olp2 = ((ex-bx)*(ey-by)) / ((y3-y2)*(x3-x2))
            if min(olp1, olp2)<self.overlap:
                break
        return [img, lm_out], kwargs

# for both image and landmark
class RandomHorizontalFlip(object):
    """
    Randomly horizontally flips the given img and lm_gt with a probability of 0.5
    Case:
        transforms.RandomHorizontalFlip(flip_idx=flip_idx, rng=rng)
    """
    def __init__(self, flip_idx, rng):
        if flip_idx is not None and isinstance(flip_idx, str): # try to load it
            self.flip_idx = misc.load_pickle(flip_idx)
        else:
            self.flip_idx = flip_idx
        self.rng = rng # rng is None means flip

    def __call__(self, img, lm_gt, **kwargs):
        assert img.ndim == 3, img.ndim
        is_flip = self.rng is None or self.rng.rand() > 0.5 # rng is None means flip
        if is_flip:
            img = img[:, ::-1, :]
            height, width = img.shape[:2]
            if lm_gt is not None:
                lm_gt[:, 0] = width - lm_gt[:, 0]
                lm_gt = lm_gt[self.flip_idx]
            src = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
            dst = np.float32([[width, 0], [0, 0], [width, height], [0, height]])
            mat = cv2.getPerspectiveTransform(src, dst)
            if 'mat' in kwargs:
                kwargs['mat'] = np.dot(mat, kwargs['mat'])
            if 'mat_gp' in kwargs:
                kwargs['mat_gp'] = np.dot(mat, kwargs['mat_gp'])
            if 'occ' in kwargs:
                kwargs['occ'] = kwargs['occ'][self.flip_idx]
            if 'mask' in kwargs:
                kwargs['mask'] = kwargs['mask'][self.flip_idx]
            if 'pose' in kwargs:
                kwargs['pose'][1] *= -1
            if 'eyeocc' in kwargs:
                eyeocc = kwargs['eyeocc']
                old_typ = type(eyeocc)
                assert len(eyeocc) % 2 == 0, len(eyeocc)
                kwargs['eyeocc'] = old_typ(np.float32(eyeocc).reshape(2, -1)[::-1, :].flatten())

        return [img, lm_gt], kwargs

class Gaussian(object):
    """
    Add gaussian noise
    Case:
        transforms.Gaussian(rng=rng)
    """
    def __init__(self, intensity=5, rng=None):
        self.intensity = intensity
        self.rng = rng or np.random

    def __call__(self, img, lm_gt, **kwargs):
        assert img.ndim == 3, img.ndim
        img = img.astype('float32')
        img = np.clip(img+self.rng.randn(*img.shape)*self.rng.randint(self.intensity), 0, 255)
        img = img.astype('uint8')
        return [img, lm_gt], kwargs

class ColorJittering(object):
    """
    Color jittering
    Case:
        transforms.ColorJittering(rng=rng)
    """
    def __init__(self, ):
        pass

    def __call__(self, img, lm_gt, **kwargs):
        assert img.ndim == 3, img.ndim
        from PIL import Image, ImageEnhance
        img = Image.fromarray(img)
        random_factor = np.random.randint(0, 31) / 10.
        color_image = ImageEnhance.Color(img).enhance(random_factor)
        random_factor = np.random.randint(10, 21) / 10.
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)
        random_factor = np.random.randint(10, 21) / 10.
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
        random_factor = np.random.randint(0, 31) / 10.
        img_out = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)
        return [np.asarray(img_out), lm_gt], kwargs

class NormalizeLandmark(object):
    """
    Normalize given landmark to [0,1]
    Case:
        transforms.NormalizeLandmark(image_shape=config.image_shape)
    """
    def __init__(self, image_shape=None, width_height=None):
        assert image_shape is not None or width_height is not None
        self.width_height = width_height or np.array(image_shape)[::-1]

    def __call__(self, img, lm_gt, **kwargs):
        assert img.ndim == 3, img.ndim
        lm_gt = lm_gt.reshape(-1, 2) * 1.0 / self.width_height # scale label to [0, 1]
        return [img, lm_gt], kwargs

class NormalizeLandmark3D(object):
    """
    Normalize given landmark to [0,1]
    Case:
        transforms.NormalizeLandmark(image_shape=config.image_shape)
    """
    def __init__(self, size, dim):
        self.size = size
        self.dim = dim

    def __call__(self, img, lm_gt, **kwargs):
        assert img.ndim == 3, img.ndim
        lm_gt = lm_gt.reshape(-1, self.dim) * 1.0 / self.size # scale label to [0, 1]
        return [img, lm_gt], kwargs

class ConvertToGray(object):
    """
    Convert image to gray scale
    Case:
        transforms.ConvertToGray()
    """
    def __call__(self, img, lm_gt, **kwargs):
        assert img.ndim == 3, img.ndim
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
        return [img, lm_gt], kwargs

class HWC2CHW(object):
    """
    Convert hwc to chw
    Case:
        transforms.HWC2CHW()
    """
    def __call__(self, img, lm_gt, **kwargs):
        assert img.ndim == 3, img.ndim
        img = np.transpose(img, (2, 0, 1))
        return [img, lm_gt], kwargs

class ResizeBackForth(object):
    """
    Resize to and back
    x_range: a tuple
    y_range: a tuple
    Case:
        transforms.ResizeBackForth(x_range=(0.3, 1.0), y_range=(0.3, 1.0), rng=rng)
    """
    def __init__(self, x_range, y_range, rng=None):
        self.x_range = x_range
        self.y_range = y_range
        self.rng = rng or np.random
        self.interpolation_method = [cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC]

    def __call__(self, img, lm_gt, **kwargs):
        assert img.ndim == 3, img.ndim
        if self.rng.rand() > 0.5:
            width, height = img.shape[:2][::-1]
            new_width = int(width * self.rng.uniform(*self.x_range))
            new_height = int(height * self.rng.uniform(*self.y_range))
            method1, method2 = self.rng.choice(self.interpolation_method, 2)
            new_img = cv2.resize(img, (new_width, new_height), interpolation=method1)
            back_img = cv2.resize(new_img, (width, height), interpolation=method2)
            img = back_img
        return [img, lm_gt], kwargs

class Occlusion(object):
    """
    Add random occlusion to img
    Case:
        transforms.Occlusion(
            occ_data=[misc.load_pickle(os.path.join(occ_root, k)) for k in occ_keys],
            scale=[(0.6, 1.0)],
            rotate=[(-180, 180)],
            trans_x=[(-0.3, 0.3)],
            trans_y=[(-0.3, 0.3)],
            rng=rng
        )
    """
    def __init__(self, occ_data, scale, rotate, trans_x, trans_y, rng):
        self.occ_data = occ_data # list
        self.scale = scale
        self.rotate = rotate
        self.trans_x = trans_x
        self.trans_y = trans_y
        self.rng = rng or np.random

    def __call__(self, img, lm_gt, **kwargs):
        if self.rng.rand() > 0.5:
            scale_range = self.scale[self.rng.randint(len(self.scale))]
            scale = self.rng.uniform(*scale_range)
            rotate_range = self.rotate[self.rng.randint(len(self.rotate))]
            rotate = self.rng.uniform(*rotate_range)
            trans_x_range = self.trans_x[self.rng.randint(len(self.trans_x))]
            trans_x = self.rng.uniform(*trans_x_range)
            trans_y_range = self.trans_y[self.rng.randint(len(self.trans_y))]
            trans_y = self.rng.uniform(*trans_y_range)

            type_key = self.rng.choice(len(self.occ_data))
            img_id = self.rng.choice(len(self.occ_data[type_key]))
            occ, mask = self.occ_data[type_key][img_id]
            mask = np.array(mask > 0.9).astype('float32')

            occ_width, occ_height = occ.shape[:2][::-1]
            img_width, img_height = img.shape[:2][::-1]
            transform = cv2.getRotationMatrix2D((occ_width/2.0, occ_height/2.0), rotate, scale)
            s = np.array([occ_width, occ_width])
            translate = np.array([trans_x, trans_y])
            transform[:, 2] += translate * s

            occ2 = cv2.warpAffine(occ, transform, dsize=(img_width, img_height))
            mask2 = cv2.warpAffine(mask, transform, dsize=(img_width, img_height))[:, :, np.newaxis].astype('uint8')

            img = (img * (1-mask2) + occ2 * mask2).astype('uint8')
        return [img, lm_gt], kwargs

class Occlusion2(object):
    """
    Add random occlusion to img
    Case:
        transforms.Occlusion(
            occ_data=[misc.load_pickle(os.path.join(occ_root, k)) for k in occ_keys],
            scale=[(0.8, 1.0)],
            rotate=[(-180, 180)],
            trans_x=[(-0.3, 0.3)],
            trans_y=[(-0.3, 0.3)],
            rng=rng
        )
    """
    def __init__(self, occ_data, scale, rotate, trans_x, trans_y, rng):
        self.occ_data = occ_data # list
        self.scale = scale
        self.rotate = rotate
        self.trans_x = trans_x
        self.trans_y = trans_y
        self.rng = rng or np.random

    def __call__(self, img, lm_gt, **kwargs):
        if self.rng.rand() > 0.3:
            scale_range = self.scale[self.rng.randint(len(self.scale))]
            scale = self.rng.uniform(*scale_range)
            rotate_range = self.rotate[self.rng.randint(len(self.rotate))]
            rotate = self.rng.uniform(*rotate_range)
            trans_x_range = self.trans_x[self.rng.randint(len(self.trans_x))]
            trans_x = self.rng.uniform(*trans_x_range)
            trans_y_range = self.trans_y[self.rng.randint(len(self.trans_y))]
            trans_y = self.rng.uniform(*trans_y_range)

            type_key = self.rng.choice(len(self.occ_data))
            img_id = self.rng.choice(len(self.occ_data[type_key]))
            occ, mask = self.occ_data[type_key][img_id]
            mask = np.array(mask > 0.9).astype('float32')

            delta = img.shape[0]*1.0/max(occ.shape[0], occ.shape[1])
            scale = scale * delta
            occ_width, occ_height = occ.shape[:2][::-1]
            img_width, img_height = img.shape[:2][::-1]
            transform = cv2.getRotationMatrix2D((occ_width/2.0, occ_height/2.0), rotate, scale)
            s = np.array([img.shape[0], img.shape[0]])
            translate = np.array([trans_x, trans_y])
            transform[:, 2] += translate * s

            occ2 = cv2.warpAffine(occ, transform, dsize=(img_width, img_height))
            mask2 = cv2.warpAffine(mask, transform, dsize=(img_width, img_height))[:, :, np.newaxis].astype('uint8')

            img = (img * (1-mask2) + occ2 * mask2).astype('uint8')
        return [img, lm_gt], kwargs


class AddNorm(object):
    """
    Compute norm
    Case:
        transform.AddNorm(
            norm_type='geom,
            n_points=81
        )
    """
    def __init__(self, norm_type='geom', n_points=81):
        self.norm_type = norm_type
        self.n_points = n_points

    def __call__(self, img, lm_gt, **kwargs):
        norm_out = geom.compute_lm_norm(lm_gt, self.norm_type, self.n_points)
        kwargs['norm'] = norm_out
        return [img, lm_gt], kwargs


class Light(object):
    """
    Add light to img
    Case:
        transforms.Light(
            light_data=[misc.load_pickle(os.path.join(occ_root, k)) for k in occ_keys],
            scale=[(0.5, 0.9)],
            ths=0.7
            rng=rng
        )
    """
    def __init__(self, light_data, scale, ths, rng):
        self.light_data = light_data # list
        self.scale = scale
        self.ths = ths
        self.rng = rng or np.random

    def __call__(self, img, lm_gt, *args):
        num = len(lightpkl)
        if img.shape[0]==56:
            template = np.array(self.light_data[1][self.rng.randint(num)])  #light56
        elif img.shape[0]==112:
            template = np.array(self.light_data[0][self.rng.randint(num)])  #light112
        else:
            template = np.array(self.light_data[0][self.rng.randint(num)])
            template = np.resize(template, (img.shape[0], img.shape[1]))
        
        flag = self.rng.rand()
        scale_range = self.scale[self.rng.randint(len(self.scale))]
        scale = self.rng.uniform(*scale_range)
        minx, maxx = template.min(), template.max()
        template = (template-minx)/(maxx-minx)*scale+(1-scale)

        outimg = img.copy()
        outimg = outimg.astype('float32')
        for j in range(outimg.shape[2]):
            if flag > self.ths:
                outimg[:,:,j] = outimg[:,:,j] * template
            else:
                outimg[:,:,j] = outimg[:,:,j] / template
                for y in range(outimg.shape[0]):
                    for x in range(outimg.shape[1]):
                        outimg[y,x,j] = min(outimg[y,x,j], 255)
            minori, maxori = img[:,:,j].min(), img[:,:,j].max()
            minout, maxout = outimg[:,:,j].min(), outimg[:,:,j].max()
            outimg[:,:,j] = (outimg[:,:,j]-minout)/(maxout-minout)*(maxori-minori)
        outimg = outimg.astype('uint8')
        return [outimg, lm_gt] + list(args)

class Gamma(object):
    """
    rejust gamme to img
    Case:
        transforms.Gamma(
            std = 10,
            #std = rng.ranint(100),
            rng=rng
        )
    """
    def __init__(self, std, rng):
        self.std = std
        self.rng = rng
        gamma_table = np.zeros((256, 256), dtype='uint8')
        for i in range(256):
            gamma = 0.4+(3-0.4)/256*i
            gamma_table[i] = np.array([((j / 255.0) ** gamma) * 255 for j in np.arange(0, 256)]).astype("uint8")
        self.gamma_table_1d = gamma_table.reshape(-1)
        nonuniform_sampler = []
        for i in [0.1,0.5,1,2,3]:
            tmp = []
            acc = 0
            for j in range(100):
                x = -i+i*2/100.*j
                v = 1./np.sqrt(2*np.pi)*np.exp(-x**2/2.)
                acc += v
                tmp.append(acc)
            low,high = tmp[0],tmp[-1]
            for j in range(100):
                tmp[j] = (tmp[j]-low)/(high-low)
            nonuniform_sampler.append(tmp)
        self.nonuniform_sampler = np.array(nonuniform_sampler)

    def __call__(self, img, lm_gt, *args):
        mean = self.rng.randint(256)
        gamma = mean + self.rng.randn(3)*self.std
        sz = img.shape[0]
        x0, y0 = self.rng.randn(2)
        norm = np.sqrt(x0**2+y0**2)
        x0 /= norm
        y0 /= norm
        mind = min(0, x0*(sz-1), y0*(sz-1), x0*(sz-1)+y0*(sz-1))
        maxd = max(0, x0*(sz-1), y0*(sz-1), x0*(sz-1)+y0*(sz-1))+1e-5
        x = np.arange(sz)*x0
        y = np.arange(sz)*y0
        x = x.reshape((1,-1))
        y = y.reshape((-1,1))
        lvl = (((x+y)-mind)/(maxd-mind)*100).astype('int')
        res = img.copy()
        for d in range(3):
            sampler_idx = self.rng.randint(5)
            interval = self.rng.randint(150,256)
            l = max(0,gamma[d] - interval/2)
            r = min(255,gamma[d] + interval/2)
            if l>r:
                l,r=r,l
            l = max(0,l)
            r = max(0,r)
            l = min(255,l)
            r = min(255,r)
            lvl_d = (self.nonuniform_sampler[sampler_idx][lvl]*(r-l)+l).astype('int')
            lvl_d = lvl_d*256+img[:,:,d]
            res[:,:,d] = self.gamma_table_1d[lvl_d]
        return [res, lm_gt] + list(args)

class Motion(object):
    """
    add motion blur to img
    Case:
        transforms.Motion(
            ws = 25,
            rng=rng
        )
    """
    def __init__(self, ws, rng):
        self.rng = rng
        self.ws = ws

    def gen_motion_kernel(self, sz, angle):
        half = sz/2
        alpha = (angle-np.floor(angle/180+0.5)*180)/180*3.1415926
        cosa = np.cos(alpha)
        sina = np.sin(alpha)
        xsign = 1
        if cosa < 0:
            xsign = -1
        elif angle==90:
            xsign = 0

        psf = 1
        sx = int( np.floor(np.fabs(half*cosa + psf*xsign -sz*1e-7) + 1) )
        sy = int( np.floor(np.fabs(half*sina + psf - sz*1e-7) + 1) )
        sx2, sy2 = int(sx*2), int(sy*2)

        pv = np.zeros((sy2, sx2))
        for i in range(sy):
            for j in range(sx):
                t = i*np.fabs(cosa) - j*sina
                rad = np.sqrt(i*i + j*j)
                if rad>=half and np.fabs(t)<=psf:
                    temp = half - np.fabs((j+t*sina)/cosa)
                    t = np.sqrt(t*t + temp*temp)
                t = psf + 1e-7 - np.fabs(t)
                if t<0:
                    t = 0
                pv[i,j] = t
                pv[sy2-i-1, sx2-j-1] = t
                pv[i+sy, j] = 0
                pv[i, j+sx] = 0
        pvsum = np.sum(pv)
        #print(pvsum)
        pv /= pvsum
        if cosa>0:
            #pv = np.flip(pv, axis=1)
            pv = np.fliplr(pv)
            #pv = np.flipud(pv)
        return pv

    def __call__(self, img, lm_gt, *args):
        h,w = img.shape[0],img.shape[1]
        sz = int(rng.rand() * w//self.ws + 1)
        angle = int(rng.rand() * 180)
        pv = self.gen_motion_kernel(sz, angle)
        #print(sz, angle, pv.shape)
        res = cv2.filter2D(img,-1,pv)
        return [res, lm_gt] + list(args)







