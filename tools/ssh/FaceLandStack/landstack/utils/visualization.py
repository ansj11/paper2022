import cv2
import numpy as np
from landstack.utils import misc
from IPython import embed

def draw_point(img, p, c=(255, 255, 128), point_size=1):
    p = tuple(map(int, p))
    c = tuple(map(int, c))
    cv2.circle(img, p, point_size, c, -1)


def draw_rect(img, rect):
    left, top, right, bottom = list(map(int, rect))
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

def draw_quadrangle(img, rect, c=(255, 0, 0), thickness=1):
    rect = rect.reshape(4, 2)
    tl, tr, bl, br = rect
    draw_line(img, tl, tr, c=c, thickness=thickness)
    draw_line(img, tr, br, c=c, thickness=thickness)
    draw_line(img, br, bl, c=c, thickness=thickness)
    draw_line(img, bl, tl, c=c, thickness=thickness)

def draw_text(img, text, p, font_scale=2, color=(255, 0, 0), thickness=5, **kwargs):
    if len(text) >= 2:
        p = list(np.array(p) + np.array([-7., 5.]))
    else:
        p = list(np.array(p) + np.array([-5., 5.]))
    p = tuple(map(int, p))
    cv2.putText(img, '{}'.format(text), p, cv2.FONT_HERSHEY_PLAIN, font_scale, color, thickness)

def draw_line(img, src, target, c=(255, 0, 0), thickness=1):
    src = tuple(src.astype('int32'))
    target = tuple(target.astype('int32'))
    cv2.line(img, src, target, c, thickness)

def get_color(index):
    colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0)] #Blue, Red, Green
    if index < len(colors):
        return colors[index]
    else:
        return np.random.randint(255, size=3)


def draw(img, *lms, name='img', canvas_height_width=None, rects=None, quards=None, text=None, mark=None, line=False,
         wait=True, show=True, rgb=False, **kwargs):
    if img is None:
        img = np.zeros((512, 512, 3))
    img = img.astype('uint8')

    img = misc.ensure_hwc(img)
    canvas_height_width = canvas_height_width or img.shape[:2]

    # meta info
    old_width_height = np.array(img.shape[:2][::-1])
    cavas_width_height = np.array(canvas_height_width)[::-1]
    ratio =  np.min(cavas_width_height / old_width_height) # longer side bounded
    new_width_height = (ratio * old_width_height).astype('int32')
    ratio2 = new_width_height / old_width_height

    # rescale image
    img_copy = cv2.resize(img, tuple(new_width_height))

    # landmark
    lms_copy = []
    for lm in lms:
        lm = lm.reshape(-1, 2)
        scale = ratio2 if np.mean(lm) > 1 else new_width_height
        lms_copy.append(scale*lm)

    if mark is not None and mark is not True:
        mark = np.array(mark)
        if mark.ndim == 1:
            mark = mark.reshape(1, -1)

    for i, lm in enumerate(lms_copy):
        color = get_color(i)
        for j, l in enumerate(lm):
            if i == 0:
                draw_point(img_copy, l, c=color, point_size=kwargs.get('point_size', 2)+2)
            else:
                draw_point(img_copy, l, c=color, point_size=kwargs.get('point_size', 2))

            if mark is not None:
                if mark is True:
                    marker = str(j)
                    draw_text(img_copy, marker, l+3, font_scale=1, color=(0, 0, 0), thickness=1)
                else:
                    try:
                        marker = str(mark[i][j])
                        draw_text(img_copy, marker, l+3, font_scale=1, color=(0, 0, 0), thickness=1)
                    except Exception as e:
                        pass

            if line and i == 1:
                draw_line(img_copy, lms_copy[i][j], lms_copy[i-1][j])

    # bounding box and text
    if rects is not None:
        rects = np.array(rects).reshape(-1, 2)
        scale = ratio2 if np.mean(rects) > 1 else new_width_height
        rects = rects * scale
        rects = rects.reshape(-1, 4)
        for rect in rects:
            draw_rect(img_copy, rect)

    if text is not None:
        if isinstance(text, str):
            text = [text]
        for t in text:
            assert isinstance(t, str), t
        for idx, t in enumerate(text):
            draw_text(img_copy, t, (50, 50+30*idx), **kwargs)

    if quards is not None:
        quards = np.array(quards).reshape(-1, 2)
        scale = ratio2 if np.mean(quards) > 1 else new_width_height
        quards = quards * scale
        quards = quards.reshape(-1, 4, 2)
        for i, quard in enumerate(quards):
            color = get_color(i)
            if i == 0:
                draw_quadrangle(img_copy, quard, c=color, thickness=5)
            else:
                draw_quadrangle(img_copy, quard, c=color, thickness=3)

    img_copy0 = np.zeros(canvas_height_width + (3,)).astype(np.uint8)
    img_copy0[:img_copy.shape[0], :img_copy.shape[1], :] = img_copy

    if show:
        cv2.imshow(name, img_copy0)

    if wait:
        cv2.waitKey(0)
    if rgb:
        img_copy0 = cv2.cvtColor(img_copy0, cv2.COLOR_BGR2RGB)
    return img_copy0

def disable_axis(axarr):
    axarr = np.array([axarr]).flatten()
    for ax in axarr:
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    return axarr

# def montage(imgs, nrow, ncol, canvas_height_width=(512, 512), allow_warp=False, show=False, wait=False):
#     assert len(imgs) > 0, len(imgs)
#     assert canvas_height_width[0] % nrow == 0, canvas_height_width[0]
#     assert canvas_height_width[1] % ncol == 0, canvas_height_width[1]
#     height = int(canvas_height_width[0] / nrow)
#     width = int(canvas_height_width[1] / ncol)
#     rows = []
#     for i in range(nrow):
#         cols = []
#         for j in range(ncol):
#             c = i * ncol + j
#             if c >= len(imgs):
#                 img = np.zeros((height, width, 3)).astype(np.uint8)
#             else:
#                 img = imgs[c]
#                 img = misc.ensure_hwc(img)
#                 assert allow_warp or img.shape[0] / height == img.shape[1] / width
#                 img = cv2.resize(img, (width, height)).astype(np.uint8)
#             cols.append(img)
#         rows.append(np.hstack(cols))
#     res = np.vstack(rows)
#
#     if show:
#         cv2.imshow('img', res)
#     if wait:
#         cv2.waitKey(0)
#     return res

def montage(imgs, nrow, ncol, canvas_height_width=None, show=False, wait=False):
    assert isinstance(imgs, (tuple, list))
    assert len(imgs) > 0, len(imgs)
    assert len(imgs) == nrow * ncol, '{} vs. {}'.format(len(imgs), nrow*ncol)
    imgs = [misc.ensure_hwc(img0) for img0 in imgs]

    rows = []
    for i in range(nrow):
        cols = []
        for j in range(ncol):
            c = i * ncol + j
            img = imgs[c]
            img = misc.ensure_hwc(img).astype(np.uint8)
            cols.append(img)
        rows.append(np.hstack(cols))
    res = np.vstack(rows)
    if canvas_height_width is not None:
        res = cv2.resize(res, (canvas_height_width[1], canvas_height_width[0]))

    if show:
        cv2.imshow('img', res)
    if wait:
        cv2.waitKey(0)
    return res

def quick_draw(x, y, title, xlabel, ylabel, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(x, y, '-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(path)
    plt.close()
