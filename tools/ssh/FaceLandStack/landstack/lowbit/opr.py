from landstack.lowbit import lowbit_prefix
from megskull.opr.helper.elemwise_trans import Identity
from neupeak.model.utils import add_name_prefix_to_subgraph
from neupeak import model as O
from landstack.lowbit import quant, stop_suffix
import numpy as np
from IPython import embed

def lowbit_relu(name:str, inps):
    name = name + ':' + lowbit_prefix + 'relu'
    ans = O.relu(inps)
    ans.rename('out')
    add_name_prefix_to_subgraph(name, inps, ans)
    return ans

def lowbit_concat(name:str, inps, axis):
    name = name + ':' + lowbit_prefix + 'concat'
    ans = O.concat(inps, axis=axis)
    ans.rename('out')
    add_name_prefix_to_subgraph(name, inps, ans)
    return ans

def lowbit_pool2x2_max(name:str, inp):
    name = name + ':' + lowbit_prefix + 'pool2x2_max'
    ans = O.pool2d('max_pool', inp, window=2, stride=2, mode='max')
    ans.rename('out')
    add_name_prefix_to_subgraph(name, inp, ans)
    return ans

def lowbit_pool2x2_avg_hardtanh(name:str, inp, sfa=7, bita=8):
    name = name + ':' + lowbit_prefix + 'pool2x2_avg_hardtanh'
    ans = O.pool2d('avg_pool', inp, window=2, stride=2, mode='average')
    ans = O.min(O.max(ans, -1), 1)
    ans = quant.mgb_linear_quant_ste(ans, sf=sfa, bit=bita, name='act')

    ans.rename('out')
    add_name_prefix_to_subgraph(name, inp, ans)
    return ans

def lowbit_conv_hardtanh(name:str, inp, sfw=5, sfa=5, bn=True, bit_midout=16, sfa_in=5, **kwargs):
    assert bit_midout <= 16, bit_midout
    name = name + ':' + lowbit_prefix + 'conv_hardtanh'
    input_nr_channel = inp.partial_shape[1]
    kernel_shape = kwargs['kernel_shape']
    output_nr_channel = kwargs['output_nr_channel']
    if 'W' not in kwargs:
        group = kwargs.get('group', None)
        kernel_shape0 = (kernel_shape, kernel_shape) if isinstance(kernel_shape, int) else kernel_shape
        if group is not None:
            if group == 'chan':
                group = input_nr_channel
            else:
                group = int(group)
            assert output_nr_channel % group == 0, output_nr_channel
            assert input_nr_channel % group == 0, input_nr_channel
            kernel_shape0 = (group, output_nr_channel//group, input_nr_channel//group) + kernel_shape0
        else:
            kernel_shape0 = (output_nr_channel, input_nr_channel) + kernel_shape0
        W = O.param('W', np.random.normal(size=kernel_shape0)*0.01)
        kwargs.pop('kernel_shape')
        kwargs.pop('output_nr_channel')
        kwargs.pop('group', None)
    else:
        W = kwargs['W']
    kwargs['W'] = quant.mgb_tanh_quant_ste(W, sf=sfw, bit=sfw+1)


    if 'nonlinearity' in kwargs:
        nonlinearity_name = getattr(kwargs['nonlinearity'], 'name')
        print("WARN: Ignore {} in lowbit_conv_hardtanh".format(nonlinearity_name))
        kwargs.pop('nonlinearity', None)

    ans = O.conv2d_vanilla('conv', inp, **kwargs)

    shr_bit = 16 - bit_midout
    assert shr_bit >= 0, shr_bit
    ans = quant.mgb_linear_quant_ste(ans, sf=sfa_in+sfw-shr_bit, bit=bit_midout, name='midout')
    ans.rename('midout:out')

    if bn:
        ans = O.batch_normalization_channel_affine('conv_bn', ans, affine=True)

    assert sfa + 1 <= 8, sfa + 1
    ans = quant.mgb_hardtanh_quant_ste(ans, sf=sfa, bit=sfa+1, name='act')
    ans.rename('out')

    add_name_prefix_to_subgraph(name, inp, ans)
    return ans

def lowbit_fc_hardtanh(name:str, inp, sfw=5, sfa=5, bn=True, bit_midout=16, sfa_in=5, **kwargs):
    name = name + ':' + lowbit_prefix + 'fc_hardtanh'
    if 'W' not in kwargs:
        filter_shape = (np.prod(inp.partial_shape[1:]), kwargs['output_dim'])
        W = O.param('W', np.random.normal(size=filter_shape) * 0.01)
    else:
        W = kwargs['W']
    kwargs['W'] = quant.mgb_tanh_quant_ste(W, sf=sfw, bit=sfw+1)

    if 'nonlinearity' in kwargs:
        nonlinearity_name = getattr(kwargs['nonlinearity'], 'name')
        print("WARN: Ignore {} in lowbit_fc_hardtanh".format(nonlinearity_name))
        kwargs.pop('nonlinearity', None)

    ans = O.fully_connected('fc', inp, nonlinearity=Identity(), **kwargs)

    ans.param_manager['b'].owner_opr.set_freezed()

    shr_bit = 16 - bit_midout
    assert shr_bit >=0, shr_bit
    ans = quant.mgb_linear_quant_ste(ans, sf=sfa_in+sfw-shr_bit, bit=bit_midout, name='midout')
    ans.rename('midout:out')

    if bn:
        ans = O.batch_normalization_channel_affine('fc_bn', ans, affine=True)

    assert sfa + 1 <= 8, sfa + 1
    ans = quant.mgb_hardtanh_quant_ste(ans, sf=sfa, bit=sfa+1, name='act')
    ans.rename('out')

    add_name_prefix_to_subgraph(name, inp, ans)
    return ans


if __name__ == '__main__':
    from megskull.network import NetworkVisitor
    x = O.constant(np.random.rand(64, 3, 112, 112))
    x = lowbit_conv_hardtanh('conv0', x, bn=True, bit_midout=16, kernel_shape=3, output_nr_channel=8)
    x = lowbit_pool2x2_avg_hardtanh('pool0', x)
    x = lowbit_fc_hardtanh('fc0', x, bn=True, bit_midout=16, output_dim=10)
    visitor = NetworkVisitor(x)
    for k, v in visitor.all_oprs_dict.items():
        print(k)

    embed()
