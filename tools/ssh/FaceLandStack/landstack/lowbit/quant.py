from megskull.opr import all as O
from neupeak.model.utils import add_name_prefix_to_subgraph
from megskull.network import RawNetworkBuilder

import numpy as np

def py_nearest_log2(x):
    return int(np.ceil(np.log2(np.abs(x).max() + 10e-8)))

def py_linear_quant(x, sf, bit):
    if bit == 1:
        delta = np.power(2.0, -sf)
        ans = np.sign(x) * delta
    else:
        delta = np.power(2.0, -sf)
        bound = np.power(2.0, bit-1)
        min_val = -bound
        max_val = bound - 1
        rounded_x = np.floor(x / delta + 0.5)
        ans = np.maximum(min_val, np.minimum(rounded_x, max_val)) * delta
    return ans

def py_int32_logical_left_shift(x, bit):
    x = np.array(x).astype(np.int32)
    sign = x & 0x80000000
    return (x << bit) & 0x7fffffff | sign

def py_int32_logical_right_shift(x, bit):
    ans = np.round(np.array(x).astype(np.float32) / 2**bit)
    ans = ans.astype(np.int32)
    return ans
    # scale = 2 ** bit
    # x / sca
    # x = np.array(x).astype(np.int32)
    # sign = x & 0x80000000
    # return (x >> bit) & 0x7fffffff | sign

def py_int32_logical_right_shift_round(x, bit):
    x += 1 << (bit-1)
    return py_int32_logical_right_shift(x, bit)

def mgb_sign_ste(x, name='sign_ste'):
    name = name or x.name
    x = O.SetGrad(x, None)
    y = 1. - 2 * O.LessEqual(x, 0)
    y.rename(name)
    x.set_grad_var(O.GradWrt(y))
    return y

def mgb_round_ste(x, name='round_ste'):
    name = name or x.name
    x = O.SetGrad(x, None)
    y = O.Floor(x+0.5)
    y.rename(name)
    x.set_grad_var(O.GradWrt(y))
    return y

def mgb_linear_quant_ste(inp, sf, bit, name=None):
    name = name or inp.name
    assert bit >= 1, bit
    inp0 = inp
    inp = inp.astype('float32')
    bit_var = O.ConstProvider(np.array(bit), name='bit', dtype=np.int32) if isinstance(bit, int) else bit.astype('int32')
    sf_var = O.ConstProvider(np.array(sf), name='sf', dtype=np.int32) if isinstance(sf, int) else sf.astype('int32')
    if bit == 1:
        delta = O.Pow(2.0, -sf_var)
        ans = mgb_sign_ste(inp) * delta
    else:
        delta = O.Pow(2.0, -sf_var)
        bound = O.Pow(2.0, bit_var - 1)
        min_val = - bound
        max_val = bound - 1
        rounded_inp = mgb_round_ste(inp / delta)
        ans = O.Max(min_val, O.Min(rounded_inp, max_val)) * delta
    add_name_prefix_to_subgraph(name, inp0, ans)
    return ans

def mgb_tanh_quant_ste(inp, sf, bit, name=None):
    name = name or inp.name
    ans = O.Tanh(inp)
    add_name_prefix_to_subgraph(name, inp, ans)
    return mgb_linear_quant_ste(ans, sf, bit, name=name)


def mgb_hardtanh_quant_ste(inp, sf, bit, name=None):
    name = name or inp.name
    ans = mgh_hardtanh(inp)
    add_name_prefix_to_subgraph(name, inp, ans)
    return mgb_linear_quant_ste(ans, sf, bit, name=name)

def mgh_hardtanh(inp, min_value=-1, max_value=1):
    ans = O.Min(O.Max(inp, min_value), max_value)
    return ans

if __name__ == '__main__':
    from IPython import embed
    x = O.ConstProvider(2.133)
    y = mgb_linear_quant_ste(x, 0, 1)
    net =  RawNetworkBuilder(inputs=[], outputs=[y])
    opr_dict = net.loss_outputs_visitor.all_oprs_dict
    for k, v in opr_dict.items():
        print(k, v)
    print(y.eval())
    embed()




