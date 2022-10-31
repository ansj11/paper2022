import numpy as np
import cv2
from IPython import embed

def align_to(srcs, target, image_shape=(1.0, 1.0), offset=0.5, extend=False):
    """
    If extend is set, will append extra column
    :param srcs:
    :param target: mean face
    :param image_shape:
    :param offset:
    :param extend:
    :return: 3x2 matrix or 3x3 extended matrix
    """
    target = np.array(target).reshape(-1, 2)
    n_points = target.shape[0]
    width_height = np.array(image_shape)[::-1]
    srcs = np.array(srcs).reshape(-1, n_points, 2)
    n_samples = len(srcs)
    mats = []
    for i in range(n_samples):
        mean = srcs[i].mean(axis=0, keepdims=True)
        mat = width_height * rotate_scale_to_dst(srcs[i] - mean, target) # move to [-0.5, 0.5] * width_height
        translation = - np.dot(mean, mat) + offset * width_height
        mat = np.concatenate([mat, translation], axis=0).astype('float64')
        if extend:
            mat = np.concatenate([mat, np.array([0, 0, 1])[:, np.newaxis]], axis=1)
        mats.append(mat)
    mats = np.array(mats).reshape(n_samples, 3, -1).squeeze()
    return mats

def align_to_general(srcs, target):
    target = np.float32(target).reshape(-1, 2)
    n_points = target.shape[0]
    srcs = np.float32(srcs).reshape(-1, n_points, 2)
    n_samples = len(srcs)
    mats = []
    for i in range(n_samples):
        mat = rotate_scale_to_dst_general(srcs[i], target)
        mats.append(mat)
    mats = np.array(mats).reshape(n_samples, 3, 3).squeeze()
    return mats

def align_to_5pts(srcs, target, image_shape=(1.0, 1.0), offset=0.5, extend=False):
    target = np.array(target).reshape(-1, 2)
    n_points = target.shape[0]
    width_height = np.array(image_shape)[::-1]
    srcs = np.array(srcs).reshape(-1, n_points, 2)
    n_samples = len(srcs)
    index = [23,36,68,63,45]
    dst = target[index,:]
    mats = []
    for i in range(n_samples):
        src = srcs[i][index,:]
        mean = src.mean(axis=0, keepdims=True)
        mat = width_height * rotate_scale_to_dst(src - mean, dst) # move to [-0.5, 0.5] * width_height
        translation = - np.dot(mean, mat) + offset * width_height
        mat = np.concatenate([mat, translation], axis=0).astype('float64')
        if extend:
            mat = np.concatenate([mat, np.array([0, 0, 1])[:, np.newaxis]], axis=1)
        mats.append(mat)
    mats = np.array(mats).reshape(n_samples, 3, -1).squeeze()
    return mats

def align_to_8pts(srcs, target, image_shape=(1.0, 1.0), offset=0.5, extend=False):
    target = np.array(target).reshape(-1, 2)
    n_points = target.shape[0]
    width_height = np.array(image_shape)[::-1]
    srcs = np.array(srcs).reshape(-1, n_points, 2)
    n_samples = len(srcs)
    index = [23,10,0,36,68,63,1,45]
    dst = target[index,:]
    mats = []
    for i in range(n_samples):
        src = srcs[i][index,:]
        mean = src.mean(axis=0, keepdims=True)
        mat = width_height * rotate_scale_to_dst(src - mean, dst) # move to [-0.5, 0.5] * width_height
        translation = - np.dot(mean, mat) + offset * width_height
        mat = np.concatenate([mat, translation], axis=0).astype('float64')
        if extend:
            mat = np.concatenate([mat, np.array([0, 0, 1])[:, np.newaxis]], axis=1)
        mats.append(mat)
    mats = np.array(mats).reshape(n_samples, 3, -1).squeeze()
    return mats

def align_to_rigid(srcs, target, image_shape=(1.0, 1.0), offset=0.5, extend=False):
    target = np.array(target).reshape(-1, 2)
    n_points = target.shape[0]
    width_height = np.array(image_shape)[::-1]
    tars = (target+0.5) * width_height
    srcs = np.array(srcs).reshape(-1, n_points, 2)
    n_samples = len(srcs)
    mats = []
    for i in range(n_samples):
        mat = cv2.estimateRigidTransform(srcs[i], tars, fullAffine=False)
        if mat is None:
            print(srcs[i], tars)
        mat = mat.T
        if extend:
            mat = np.concatenate([mat, np.array([0, 0, 1])[:, np.newaxis]], axis=1)
        mats.append(mat)
    mats = np.array(mats).reshape(n_samples, 3, -1).squeeze()
    return mats

def align_to_affine(srcs, target, image_shape=(1.0, 1.0), offset=0.5, extend=False):
    target = np.array(target).reshape(-1, 2)
    n_points = target.shape[0]
    width_height = np.array(image_shape)[::-1]
    tars = (target+0.5) * width_height
    srcs = np.array(srcs).reshape(-1, n_points, 2)
    n_samples = len(srcs)
    mats = []
    for i in range(n_samples):
        print(srcs[i].shape, tars.shape)
        mat = cv2.getAffineTransform(srcs[i], tars)
        if mat is None:
            print(srcs[i], tars)
        mat = mat.T
        if extend:
            mat = np.concatenate([mat, np.array([0, 0, 1])[:, np.newaxis]], axis=1)
        mats.append(mat)
    mats = np.array(mats).reshape(n_samples, 3, -1).squeeze()
    return mats

def align_to_homography(srcs, target, image_shape=(1.0, 1.0), offset=0.5, extend=False):
    target = np.array(target).reshape(-1, 2)
    n_points = target.shape[0]
    width_height = np.array(image_shape)[::-1]
    srcs = np.array(srcs).reshape(-1, n_points, 2)
    n_samples = len(srcs)
    mats = []
    for i in range(n_samples):
        mean = srcs[i].mean(axis=0)
        mat, _ = cv2.findHomography(srcs[i]-mean, target, cv2.RANSAC)
        m1 = np.float32([[1, 0, -mean[0]], [0, 1, -mean[1]], [0, 0, 1]])
        m2 = np.float32([[1, 0, offset], [0, 1, offset], [0, 0, 1]])
        m3 = np.float32([[width_height[0], 0, 0], [0, width_height[1], 0], [0, 0, 1]])
        m0 = m3.dot(m2).dot(mat).dot(m1)
        if not extend:
            mats.append(m0[:2, :].T)
        else:
            mats.append(m0.T)
    mats = np.array(mats).reshape(n_samples, 3, -1).squeeze()
    return mats

def align_to_affine3d(srcs, target, image_shape=(1.0, 1.0), offset=0.5, extend=False):
    target = np.array(target).reshape(-1,3)
    n_points = target.shape[0]
    width_height = np.array(image_shape)[::-1]
    srcs = np.array(srcs).reshape(-1, n_points, 3)
    n_samples = len(srcs)
    mats = []
    for i in range(n_samples):
        mean = srcs[i].mean(axis=0, keepdims=True)
        mat, _ = cv2.estimateAffine3D(srcs[i]-mean, target)
        mat = mat.T
        if extend:
            mat = np.concatenate([mat, np.array([0, 0, 0, 1])], axis=1)
        mats.append(mat)
    mats = np.array(mats).reshape(n_samples, 4, -1).squeeze()
    return mats

def align_to_cen(srcs, target, image_shape=(1.0, 1.0), offset=0.5, extend=False):
    mats = align_to(srcs, target, image_shape, offset, extend)
    tmp = lm_affine(srcs, mats)
    for i in range(mats.shape[0]) :
        maxx = tmp[i,:,0].max()
        maxy = tmp[i,:,1].max()
        minx = tmp[i,:,0].min()
        miny = tmp[i,:,1].min()
        mats[i,0,2] = mats[i,0,2] - (maxx+minx)*0.5 + image_shape[0]*0.5
        mats[i,1,2] = mats[i,1,2] - (maxy+miny)*0.5 + image_shape[1]*0.5
    return mats

def rotate_scale_to_dst(src, dst, eps=1e-12):
    src_x, src_y = src[:, 0], src[:, 1]
    dst_x, dst_y = dst[:, 0], dst[:, 1]
    d = (src**2).sum()

    a = sum(src_x*dst_x + src_y*dst_y) / (d + eps)
    b = sum(src_x*dst_y - src_y*dst_x) / (d + eps)
    mat = np.array([[a, b], [-b, a]])
    return mat

def rotate_scale_to_dst_general(src0, dst0, eps=1e-12):
    ksrc = src0.mean(axis=0, keepdims=True)
    kdst = dst0.mean(axis=0, keepdims=True)

    src = src0 - ksrc
    dst = dst0 - kdst
    src_x, src_y = src[:, 0], src[:, 1]
    dst_x, dst_y = dst[:, 0], dst[:, 1]
    d = (src**2).sum()

    a = sum(src_x*dst_x + src_y*dst_y) / (d + eps)
    b = sum(src_x*dst_y - src_y*dst_x) / (d + eps)
    mat = np.array([[a, -b, 0], [b, a, 0], [0, 0, 1]])
    # embed()

    mat_src, mat_dst = np.eye(3), np.eye(3)
    mat_src[0, 2] = -ksrc[0, 0]
    mat_src[1, 2] = -ksrc[0, 1]
    mat_dst[0, 2] = kdst[0, 0]
    mat_dst[1, 2] = kdst[0, 1]
    mat = mat_dst.dot(mat).dot(mat_src)
    return mat

def lm_affine(lms, mats):
    """
    :param lms:
    :param mats: 3x2 matrix or 3x3 extended matrix
    :return:
    """
    n_samples = len(mats) if np.array(mats).ndim == 3 else 1
    lms = np.array(lms).reshape(n_samples, -1, 2)
    mats = np.array(mats).reshape(n_samples, 3, -1)
    res = []
    for i in range(n_samples):
        lm_extend = np.concatenate([lms[i], np.ones((len(lms[i]), 1))], axis=1) # extend the lms[i]
        res.append(lm_extend.dot(mats[i]))
    res = np.array(res)[:, :, :2].squeeze()
    return res

def lm_affine_general(lms, mats):
    n_samples = len(mats) if np.array(mats).ndim == 3 else 1
    lms = np.array(lms).reshape(n_samples, -1, 2)
    mats = np.array(mats).reshape(n_samples, 3, 3)
    lms_extend = np.concatenate([lms, np.ones(list(lms.shape[:-1]) + [1])], axis=2).transpose(0, 2, 1)
    res = np.matmul(mats, lms_extend).transpose(0, 2, 1)[:, :, :2].squeeze()
    return res

def lm_affine_3d(lms, mats):
    """
    :param lms:
    :param mats: 4x3 matrix or 4x4 extended matrix
    :return:
    """
    n_samples = len(mats) if np.array(mats).ndim == 4 else 1
    lms = np.array(lms).reshape(n_samples, -1, 3)
    mats = np.array(mats).reshape(n_samples, 4, -1)
    res = []
    for i in range(n_samples):
        lm_extend = np.concatenate([lms[i], np.ones((len(lms[i]), 1))], axis=1) # extend the lms[i]
        res.append(lm_extend.dot(mats[i]))
    res = np.array(res)[:, :, :3].squeeze()
    return res

def lm_affine_fake3d(lms, mats):
    """
    :param lms:
    :param mats: 3x2 matrix or 3x3 extended matrix
    :return:
    """
    n_samples = len(mats) if np.array(mats).ndim == 3 else 1
    lms = np.array(lms).reshape(n_samples, -1, 3)
    mats = np.array(mats).reshape(n_samples, 3, -1)
    res = []
    for i in range(n_samples):
        lm_2d, lm_3d = lms[i,:,:2], lms[i,:,2:]
        lm_extend = np.concatenate([lm_2d, np.ones((len(lm_2d), 1))], axis=1)
        lm_res = lm_extend.dot(mats[i][:,:2])
        s = mats[i][0][0]*mats[i][0][0] + mats[i][0][1]*mats[i][0][1]
        lm_res = np.concatenate([lm_res, lm_3d*s], axis=1)
        res.append(lm_res)
    res = np.array(res)[:, :, :3].squeeze()
    return res

def invert_mat(mats):
    mats = np.array(mats).astype('float64')
    n_samples = len(mats) if mats.ndim == 3 else 1
    assert mats.shape[-1] == mats.shape[-2], 'mats should be square, found {}x{}'.format(mats.shape[-2], mats.shape[-1])
    mats = mats.reshape(n_samples, mats.shape[-1], mats.shape[-1]).astype('float64')
    mats_inv = []
    for i in range(n_samples):
        mats_inv.append(np.linalg.inv(mats[i]))
    mats_inv = np.array(mats_inv).squeeze()
    return mats_inv

def warp_affine(imgs, mats, image_shape, borderMode=cv2.BORDER_REPLICATE, borderValue=(0,0,0)):
    image_shape = tuple(np.array(image_shape).astype('int32'))
    width_height = image_shape[::-1]
    n_samples = len(mats) if np.array(mats).ndim == 3 else 1 # infer n_samples

    mats = np.array(mats).reshape(n_samples, 3, -1)
    mats = mats[:, :, :2] if mats.shape[-1] == 3 else mats
    imgs = [imgs] if n_samples == 1 else imgs

    res = []
    for i in range(n_samples):
        img = imgs[i]
        mat = mats[i]
        if borderMode==cv2.BORDER_CONSTANT:
            img = cv2.warpAffine(img, mat.T, width_height, borderMode=borderMode, borderValue=borderValue)
        else:
            img = cv2.warpAffine(img, mat.T, width_height, borderMode=borderMode)
        res.append(img)
    res = res[0] if n_samples == 1 else res
    return res

def warp_affine_general(imgs, mats, image_shape, borderMode=cv2.BORDER_REPLICATE, borderValue=(0,0,0)):
    image_shape = tuple(np.array(image_shape).astype('int32'))
    width_height = image_shape[::-1]
    n_samples = len(mats) if np.array(mats).ndim == 3 else 1 # infer n_samples

    mats = np.array(mats).reshape(n_samples, 3, -1)
    mats = mats[:, :, :2] if mats.shape[-1] == 3 else mats
    imgs = [imgs] if n_samples == 1 else imgs

    res = []
    for i in range(n_samples):
        img = imgs[i]
        mat = mats[i]
        if borderMode==cv2.BORDER_CONSTANT:
            img = cv2.warpAffine(img, mat, width_height, borderMode=borderMode, borderValue=borderValue)
        else:
            img = cv2.warpAffine(img, mat, width_height, borderMode=borderMode)
        res.append(img)
    res = res[0] if n_samples == 1 else res
    return res


def matmul_batch(mats1, mats2):
    assert len(mats1) == len(mats2), '{} vs. {}'.format(len(mats1), len(mats2))
    n_samples = len(mats1)
    res = np.array([mats1[i].dot(mats2[i]) for i in range(n_samples)])
    return res


def compute_lm_norm(lm, norm_type='geom', n_points=81):
    """
    geom: With extra distance between the pupil of two eyes
    pupil: With extra distance between geometric center of each eye
    area: With extra square root of area of whole face, for profile and large pose
    """
    lm = np.array(lm).reshape(-1, n_points, 2)
    assert norm_type in ['geom', 'pupil', 'area'], norm_type
    norm = []
    for lm0 in lm:
        if norm_type == 'pupil':
            if n_points == 81:
                norm0 = np.array([np.sum((lm0[23] - lm0[68])**2)**0.5]) # pupil
            elif n_points == 90 or n_points == 124 or n_points == 228 or n_points == 264:
                norm0 = np.array([np.sum((lm0[47] - lm0[56])**2)**0.5]) # pupil
            elif n_points == 66:
                norm0 = np.array([np.sum((lm0[36] - lm0[45])**2)**0.5])
        elif norm_type == 'geom':
            left_center = lm0[19:28].mean(axis=0)
            right_center= lm0[64:73].mean(axis=0)
            norm0 = np.array([np.sum((left_center - right_center)**2)**0.5])
        else:
            left_top = np.min(lm0, axis=0)
            right_bottom = np.max(lm0, axis=0)
            norm0 = np.sqrt(np.prod(right_bottom - left_top))
        norm.append(norm0)
    norm = np.array(norm).astype('float64').squeeze()
    return norm

def getyaw(label, inpsize, mean_face):
    gt = label.copy()
    gt = lm_affine(gt, align_to(gt, mean_face, image_shape=(inpsize,inpsize)))
    #dx0 = (gt[12,0]-gt[63,0])*(gt[12,0]-gt[63,0]) + (gt[12,1]-gt[63,1])*(gt[12,1]-gt[63,1])
    #dx1 = (gt[3,0]-gt[63,0])*(gt[3,0]-gt[63,0]) + (gt[3,1]-gt[63,1])*(gt[3,1]-gt[63,1])
    dx0 = (gt[12,0]-gt[53,0])*(gt[12,0]-gt[53,0])
    dx1 = (gt[3,0]-gt[53,0])*(gt[3,0]-gt[53,0])
    yaw = np.sqrt(max(dx0/dx1, dx1/dx0))
    return yaw


