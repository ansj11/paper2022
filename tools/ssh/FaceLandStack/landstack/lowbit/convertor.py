import argparse
from neupeak.utils.cli import load_network
import megskull.opr.all as O
from IPython import embed
from collections import OrderedDict
from landstack.lowbit import lowbit_prefix, begin_suffix, stop_suffix, quant
import numpy as np
from neupeak.model.utils import  add_name_prefix_to_subgraph
from megskull.network import RawNetworkBuilder
from meghair.utils import io

from megskull.opr.contrib.lowbit import RoundShr32_8, RoundShr32_16, RoundShr16_16
from megskull.opr.contrib.lowbit import Pool2x2AvgSfa, Pool2x2Avg
from megskull.opr.contrib.lowbit import FcAffine
from megskull.opr.contrib.lowbit import ConvAffine

IMG_SFA = 7
SFA = 5
BIT_MIDOUT = 14
BIT_AFFINE_B = 32

def lowbit_pool2x2_max_converter(net, opr_dict, name, typ):
    pass

def lowbit_patch_converter(net, opr_dict, name, typ):
    inp = [v for k, v in opr_dict.items() if k.endswith('_patch')][0]
    out_opr = [v for k, v in opr_dict.items() if k.endswith('_patch:out')][0]
    if inp.dtype != np.int8:
        print("\tInput is {}, may be slow".format(inp.dtype))
    ans = O.Floor(inp + 0.5)
    ans = ans.astype(np.int8)
    net.outputs_visitor.replace_vars(
        [(out_opr.outputs[0], ans)], copy=False)
    add_name_prefix_to_subgraph(name, inp, ans)
    ans.rename(out_opr.name)

def lowbit_img_converter(net, opr_dict, name, typ):
    if 'img:sub' in opr_dict:
        inp = opr_dict['img:sub']
        print("\tFind img:sub")
        ans = inp.astype(np.int8)
    else:
        inp = opr_dict['img']
        print("\tFind img")
        ans = (inp - 128.0).astype(np.int8)

    if 'img:out' not in opr_dict:
        print("\tNo img:out found, skip it")
        return
    out_opr = opr_dict['img:out']

    if inp.dtype != np.int8:
        print("\tInput is {}, may be slow".format(inp.dtype))

    net.outputs_visitor.replace_vars(
        [(out_opr.outputs[0], ans)], copy=False)
    add_name_prefix_to_subgraph(name, inp, ans)

def lowbit_relu_converter(net, opr_dict, name, typ):
    assert len(opr_dict) == 1, len(opr_dict)
    k, v = list(opr_dict.items())[0]
    assert isinstance(v, O.ReLU), type(v)
    assert len(v.inputs) == 1, len(v.inputs)
    inp = v.inputs[0]
    if inp.dtype != np.int8:
        print("\tInput is {}, may be slow".format(inp.dtype))
        inp = inp.astype(np.int8)
    ans = O.ReLU(inp).astype(np.int8)

    net.outputs_visitor.replace_vars(
        [(v.outputs[0], ans)], copy=False)

    add_name_prefix_to_subgraph(name, inp, ans)

def lowbit_concat_converter(net, opr_dict, name, typ):
    assert len(opr_dict) == 1, len(opr_dict)
    k, v = list(opr_dict.items())[0]
    assert isinstance(v, O.Concat), type(v)
    print("\t #inputs: {}".format(len(v.inputs)))
    inps = []

    for inp in v.inputs:
        if inp.dtype != np.int8:
            print("\tInput is {}, may be slow".format(inp.dtype))
            inp = inp.astype(np.int8)
        inps.append(inp)
    ans = O.Concat(inps, axis=v._axis)

    if stop_suffix in name:
        ans = ans.astype(np.float32) * 2**-SFA
    ans.rename('out')

    net.outputs_visitor.replace_vars(
        [(v.outputs[0], ans)], copy=False)

    add_name_prefix_to_subgraph(name, inps, ans)

def lowbit_pool2x2_avg_hardtanh_converter(net, opr_dict, name, typ):
    anchor = name + ':' + lowbit_prefix + typ
    pool_opr = opr_dict[anchor + ':avg_pool']
    inp = pool_opr.inputs[0]
    out_opr = opr_dict[anchor + ':out']
    if inp.dtype != np.int8:
        print("\tInput is {}, may be slow".format(inp.dtype))
        inp = inp.astype(np.int8)
    if begin_suffix in name:
        print("\tChanging sf from {} to {}".format(IMG_SFA, SFA))
        sf_pool = int(opr_dict[anchor + ':act:sf'].eval())
        act_shr_bits = IMG_SFA - sf_pool + 2
        assert act_shr_bits >= 0, act_shr_bits
        ans = Pool2x2AvgSfa(inp, act_shr_bits, name='pool2x2_avg_round')
    else:
        print("\t sf not change")
        ans = Pool2x2Avg(inp, name='pool2x2_avg')

    if stop_suffix in name:
        ans = ans.astype(np.float32) * 2**-SFA
    ans.rename('out')

    net.outputs_visitor.replace_vars(
        [(out_opr.outputs[0], ans)], copy=False)
    add_name_prefix_to_subgraph(name, inp, ans)

def lowbit_conv_hardtanh_converter(net, opr_dict, name, typ):
    anchor = name + ':' + lowbit_prefix + typ
    conv_opr = opr_dict[anchor + ':conv']
    out_opr = opr_dict[anchor + ':out']
    conv_w = conv_opr.param_manager['W'].eval()
    conv_w_sf = int(opr_dict[anchor + ':W:sf'].eval())
    if 'b' in conv_opr.param_manager._param_map:
        conv_b = np.abs(conv_opr.param_manager['b'].eval()).sum()
    else:
        conv_b = 0.0
    assert conv_b == 0, conv_b

    # bn
    conv_bn_name = anchor + ':conv_bn'
    if conv_bn_name in opr_dict:
        mean, var, t = opr_dict[conv_bn_name].get_opr_state().val
        mean /= t
        std = np.maximum(var / t, 1e-5) ** 0.5
    else:
        print("\tNo bn in {}".format(anchor))
        mean = np.array([0])
        std = np.array([1.0])

    # affine
    conv_bn_affine_name = anchor + ':conv_bn_affine'
    if conv_bn_affine_name in opr_dict:
        k = opr_dict[conv_bn_affine_name].param_manager['k'].eval()
        b = opr_dict[conv_bn_affine_name].param_manager['b'].eval()
    else:
        print("\tNo affine in {}".format(anchor))
        k = np.array([1.0])
        b = np.array([0.0])

    affine_k = k / std
    affine_b = (conv_b - mean) * k / std + b

    inp = conv_opr.inputs[0]
    group = conv_opr._group_spec
    if group is None:
        acc_sum = np.prod((inp.partial_shape[1],) + conv_opr.kernel_shape)
    else:
        if group == 'chan':
            acc_sum = np.prod(conv_opr.kernel_shape)
        else:
            acc_sum = np.prod((inp.partial_shape[1] // group,) + conv_opr.kernel_shape)

    sf_inp = IMG_SFA if begin_suffix in name else SFA
    expected_acc = np.sqrt(acc_sum)
    expected_bit = np.ceil(np.log2(np.sqrt(acc_sum)))
    int8x8x16 = expected_bit + conv_w_sf + sf_inp + 1 <= 16 + 1

    print("\tgroup: {}, sfw: {}, sfa: {}, #sum: {}, expected: {}, bits: {}, int8x8x16: {}".format(
        group, conv_w_sf, sf_inp, acc_sum, expected_acc, expected_bit, int8x8x16))

    midout_shr_bit = 16 - BIT_MIDOUT
    assert midout_shr_bit >= 0, midout_shr_bit
    midout_sf = sf_inp + conv_w_sf - midout_shr_bit
    if inp.dtype != np.int8:
        print("\tInput is {}, may be slow".format(inp.dtype))
        assert begin_suffix not in name, name
        inp = (inp * 2**sf_inp).astype(np.int8)
    w = (conv_w * 2**conv_w_sf).astype(np.int8)
    w = O.ParamProvider('W', w, dtype=np.int8)
    if int8x8x16:
        ans = O.Conv2DVanilla(
            'conv', inp, W=w,
            stride=conv_opr.stride,
            padding=conv_opr.padding,
            data_type='INT8x8x16'
        )
        if midout_shr_bit > 0:
            ans = RoundShr16_16(ans, midout_shr_bit, name='midout')
    else:
        ans = O.Conv2DVanilla(
            'conv', inp, W=w,
            stride=conv_opr.stride,
            padding=conv_opr.padding,
            data_type='INT8x8x32'
        )
        ans = RoundShr32_16(ans, midout_shr_bit, 15-midout_shr_bit, name='midout')
    ans.rename("midout:out")


    bits_k = 32 - BIT_MIDOUT + 2
    sf_affine_k = - (quant.py_nearest_log2(affine_k) - bits_k + 1)
    sf_affine_b = - (quant.py_nearest_log2(affine_b) - BIT_AFFINE_B + 1)
    affine_k_quant = quant.py_linear_quant(affine_k, sf_affine_k, bits_k)
    affine_b_quant = quant.py_linear_quant(affine_b, sf_affine_b, BIT_AFFINE_B)
    print("\tsf_affine_k {}, bits_k {}, diff_k {}, {}".format(sf_affine_k, bits_k, np.abs(affine_k - affine_k_quant).sum(), np.abs(affine_k - affine_k_quant).mean()))
    print("\tsf_affine_b {}, bits_b {}, diff_b {}, {}".format(sf_affine_b, BIT_AFFINE_B, np.abs(affine_b - affine_b_quant).sum(), np.abs(affine_b - affine_b_quant).mean()))

    # use 16x32+32->32 affine
    k = np.array(affine_k_quant * 2**sf_affine_k).astype(np.int32)
    sf_final =  midout_sf + sf_affine_k
    b_shr_bits = - sf_final + sf_affine_b
    b0 = np.array(affine_b_quant * 2**sf_affine_b).astype(np.int32)
    print("\tmidout:{}, b_shr_bits: {}".format(midout_shr_bit, b_shr_bits))
    assert b_shr_bits >= 0, b_shr_bits
    b = quant.py_int32_logical_right_shift(b0, b_shr_bits) if b_shr_bits > 0 else b0

    k_var = O.ParamProvider('k', k, dtype=np.int32)
    b_var = O.ParamProvider('b', b, dtype=np.int32)
    ans = ConvAffine(ans, k=k_var, b=b_var, name='conv_affine_16x32x32')

    act_shr_bits = sf_final - SFA
    assert act_shr_bits > 0, act_shr_bits
    ans = RoundShr32_8(ans, act_shr_bits, SFA, name='act')

    if stop_suffix in name:
        ans = ans.astype(np.float32) * 2**-SFA
    ans.rename('out')

    # insert to graph
    net.outputs_visitor.replace_vars(
        [(out_opr.outputs[0], ans)], copy=False)
    add_name_prefix_to_subgraph(name, inp, ans)

def lowbit_fc_hardtanh_converter(net, opr_dict, name, typ):
    anchor = name + ':' + lowbit_prefix + typ
    fc_opr = opr_dict[anchor + ':fc']
    out_opr = opr_dict[anchor + ':out']
    fc_w = fc_opr.param_manager['W'].eval()
    fc_w_sf = int(opr_dict[anchor + ':W:sf'].eval())
    if 'b' in fc_opr.param_manager._param_map:
        fc_b = np.abs(fc_opr.param_manager['b'].eval()).sum()
    else:
        fc_b = 0.0
    assert fc_b == 0, fc_b

    # bn
    fc_bn_name = anchor + ':fc_bn'
    if fc_bn_name in opr_dict:
        mean, var, t = opr_dict[fc_bn_name].get_opr_state().val
        mean /= t
        std = np.maximum(var / t, 1e-5) ** 0.5
    else:
        print("\tNo bn in {}".format(anchor))
        mean = np.array([0])
        std = np.array([1.0])

    # affine
    fc_bn_affine_name = anchor + ':fc_bn_affine'
    if fc_bn_affine_name in opr_dict:
        k = opr_dict[fc_bn_affine_name].param_manager['k'].eval()
        b = opr_dict[fc_bn_affine_name].param_manager['b'].eval()
    else:
        print("\tNo affine in {}".format(anchor))
        k = np.array([1.0])
        b = np.array([0.0])

    affine_k = k / std
    affine_b = (fc_b - mean) * k / std + b

    inp = fc_opr.inputs[0]
    acc_sum = inp.partial_shape[1]
    expected_acc = np.sqrt(acc_sum)
    expected_bit = np.ceil(np.log2(np.sqrt(acc_sum)))
    int8x8x16 = False

    midout_shr_bit = 16 - BIT_MIDOUT
    assert midout_shr_bit >= 0, midout_shr_bit
    if begin_suffix in name:
        midout_sf =  IMG_SFA + fc_w_sf - midout_shr_bit
    else:
        midout_sf =  SFA + fc_w_sf - midout_shr_bit

    print("\tacc_sum: {}, expected: {}, bits: {}, int8x8x16: {}".format(acc_sum, expected_acc, expected_bit, int8x8x16))
    fc_w = (fc_w * 2**fc_w_sf).astype(np.int8)
    w = O.ParamProvider('W', fc_w, dtype=np.int8)
    if inp.dtype != np.int8:
        print("\tInput is {}, may be slow".format(inp.dtype))
        assert begin_suffix not in name, name
        inp = (inp * 2**SFA).astype(np.int8)
    if int8x8x16:
        inp = inp.reshape(inp.shape[0], -1)
        ans = O.MatMul(a=inp, b=w, data_type='INT8x8x16')
        if midout_shr_bit > 0:
            ans = RoundShr16_16(ans, midout_shr_bit, name='midout')
    else:
        inp = inp.reshape(inp.shape[0], -1)
        ans = O.MatMul(a=inp, b=w, data_type='INT8x8x32')
        ans = RoundShr32_16(ans, midout_shr_bit, 15-midout_shr_bit, name='midout')
    ans.rename("midout:out")


    bits_k = 32 - BIT_MIDOUT
    sf_affine_k = - (quant.py_nearest_log2(affine_k) - bits_k + 1)
    sf_affine_b = - (quant.py_nearest_log2(affine_b) - BIT_AFFINE_B + 1)
    affine_k_quant = quant.py_linear_quant(affine_k, sf_affine_k, bits_k)
    affine_b_quant = quant.py_linear_quant(affine_b, sf_affine_b, BIT_AFFINE_B)
    print("\tsf_affine_k {}, bits_k {}, diff_k {}, {}".format(sf_affine_k, bits_k, np.abs(affine_k - affine_k_quant).sum(), np.abs(affine_k - affine_k_quant).mean()))
    print("\tsf_affine_b {}, bits_b {}, diff_b {}, {}".format(sf_affine_b, BIT_AFFINE_B, np.abs(affine_b - affine_b_quant).sum(), np.abs(affine_b - affine_b_quant).mean()))

    # use 32x32+32->32 affine
    k = np.array(affine_k_quant * 2**sf_affine_k).astype(np.int32)
    sf_final =  midout_sf + sf_affine_k
    b_shr_bits = - sf_final + sf_affine_b
    b0 = np.array(affine_b_quant * 2**sf_affine_b).astype(np.int32)
    assert b_shr_bits >= 0, b_shr_bits
    b = quant.py_int32_logical_right_shift(b0, b_shr_bits) if b_shr_bits > 0 else b0

    k_var = O.ParamProvider('k', k, dtype=np.int32) # convert to int32
    b_var = O.ParamProvider('b', b, dtype=np.int32) # convert to int32
    ans = FcAffine(ans.astype(np.int32), k=k_var, b=b_var, name='fc_affine_32x32x32')

    # # use float for tmp
    act_shr_bits = sf_final - SFA
    assert act_shr_bits > 0, act_shr_bits
    ans = RoundShr32_8(ans, act_shr_bits, SFA, name='act')

    if stop_suffix in name:
        ans = ans.astype(np.float32) * 2**-SFA
    ans.rename('out')

    # insert to graph
    net.outputs_visitor.replace_vars(
        [(out_opr.outputs[0], ans)], copy=False)
    add_name_prefix_to_subgraph(name, inp, ans)

convert_dict = OrderedDict([
    ('pool2x2_max', lowbit_pool2x2_max_converter),
    ('pool2x2_avg_hardtanh', lowbit_pool2x2_avg_hardtanh_converter),
    ('conv_hardtanh', lowbit_conv_hardtanh_converter),
    ('fc_hardtanh', lowbit_fc_hardtanh_converter),
    ('concat', lowbit_concat_converter),
    ('img', lowbit_img_converter),
    ('patch', lowbit_patch_converter),
    ('relu', lowbit_relu_converter),
])

def group(opr_names):
    groups = OrderedDict()
    for opr in opr_names:
        if lowbit_prefix in opr:
            name, typ = opr.split(lowbit_prefix)
            name = name[:-1] # strip ':'
            typ = typ.split(':')[0] # extra opr type
        elif 'img' in opr:
            name = 'preprocess'
            typ = 'img'
        elif 'patch' in opr:
            name = [v for v in opr.split(':') if '_patch' in v]
            assert len(name) == 1, name
            name = name[0]
            typ = 'patch'
        else:
            continue
        key = (typ, name)
        groups.setdefault(key, []).append(opr)
    return groups

def run(model, output=None):
    print("sf_a: {}, bit_out: {}, bit_affine_b:{}".format(SFA, BIT_MIDOUT, BIT_AFFINE_B))
    assert SFA <= 7, SFA
    assert BIT_MIDOUT <= 16, BIT_MIDOUT
    net = model if isinstance(model, RawNetworkBuilder) else load_network(model)
    outputs = net.outputs
    net0 = RawNetworkBuilder(inputs=[], outputs=outputs)

    all_opr_dict = net0.outputs_visitor.all_oprs_dict
    opr_names = list(all_opr_dict.keys())
    groups = group(opr_names)
    for (typ, name), oprs in groups.items():
        print("Converting {} with {} converter".format(name, typ))
        opr_dict = {v:all_opr_dict[v] for v in oprs}
        convert_dict[typ](net, opr_dict, name, typ)

    if output is not None:
        print("Saving converted model to {}".format(output))
        io.dump(net, output)
    else:
        return net

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='model to convert')
    parser.add_argument('output', help='save path')
    args, unknownargs = parser.parse_known_args()
    run(args.model, args.output)

if __name__ == '__main__':
    main()
