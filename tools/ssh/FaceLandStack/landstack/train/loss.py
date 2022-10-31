from IPython import embed

def gen_l1_loss(pred_lm, label, norm=None, mask=None, name='loss', reduce=True):
    """l1 loss, set reduce=False to get loss vector"""
    pred_lm = pred_lm.reshape(pred_lm.shape[0], -1)
    label = label.reshape(label.shape[0], -1)
    loss = (pred_lm - label).abs().mean(axis=1)
    if norm is not None:
        loss /= norm.reshape(-1)
    if reduce:
        if mask is not None:
            loss = (mask*loss).sum() / mask.sum()
        else:
            loss = loss.mean()
    if hasattr(loss, 'rename'):
        loss.rename(name)
    return loss
#
def gen_l2_1d_loss(pred_lm, label, norm=None, mask=None, name='loss', reduce=True):
    """l2 loss of vector, set reduce=False to get loss vector"""
    pred_lm = pred_lm.reshape(pred_lm.shape[0], -1)
    label = label.reshape(label.shape[0], -1)
    loss = ((pred_lm - label) ** 2).mean(axis=1)
    if norm is not None:
        loss /= norm.reshape(-1)
    if reduce:
        if mask is not None:
            loss = (mask*loss).sum() / mask.sum()
        else:
            loss = loss.mean()
    if hasattr(loss, 'rename'):
        loss.rename(name)
    return loss

def gen_l2_2d_loss(pred_lm, label, norm=None, mask=None, name='loss', reduce=True):
    """l2 loss of matrix, set reduce=False to get loss vector"""
    pred_lm = pred_lm.reshape(pred_lm.shape[0], -1, 2)
    label = label.reshape(label.shape[0], -1, 2)
    loss = ((((pred_lm - label) ** 2).sum(axis=2)) ** 0.5).mean(axis=1)
    if norm is not None:
        loss /= norm.reshape(-1)
    if reduce:
        if mask is not None:
            loss = (mask*loss).sum() / mask.sum()
        else:
            loss = loss.mean()
    if hasattr(loss, 'rename'):
        loss.rename(name)
    return loss

