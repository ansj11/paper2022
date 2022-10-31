

import time
import numpy as np

from megskull.graph import FpropEnv, iter_dep_opr
from megskull.network import NetworkVisitor
from megskull.network import RawNetworkBuilder

import collections
from benchmark.tester import TestUnitBase
from landstack.utils import misc
from neupeak.utils.cli import load_network
from IPython import embed




def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "{:3.3f} {}{}".format(num, unit, suffix)
        num /= 1024.0
    sign_str = '-' if num  < 0 else ''
    return "{}{:.1f} {}{}".format(sign_str, num, 'Yi', suffix)


def network_visitor(network):
    targets = ([network.loss_var] if network.loss_var is not None else []) + list(
        network.outputs)
    return NetworkVisitor(targets)


def network_initialized(network):
    for param in network_visitor(network).all_params:
        if param._shared_nd is None:
            return False
    return True

def initialize_network(network):
    env = FpropEnv(comp_graph_opt={'log_static_mem_alloc':False}, verbose_fprop=False)
    output_vars = filter(lambda x: x is not None,
                         [network.loss_var] + list(network.outputs))
    for var in output_vars:
        env.get_mgbvar(var, allow_fprop=True)


def get_visitor(network):
    try:
        return NetworkVisitor(network.outputs)
    except Exception as e:
        import traceback
        traceback.print_exc()

def get_num_oprs(var):
    return len(NetworkVisitor(var).all_oprs)

def get_num_params(var):
    return len(NetworkVisitor(var).all_params)

class TestUnit(TestUnitBase):
    def __init__(self, caches):
        name = "Model Info Tester ({}-{})".format(time.strftime("%x"), time.strftime("%X"))
        super(TestUnit, self).__init__(name, caches)
        self._info = collections.OrderedDict([
            ('#oprs', 0), ('#params', 0), ('tot_param_dim', 0), ('param_size', 0),
            ('total_flops', 0), ('flops/param_size', 0)
        ])

    def compute_metric(self, network, out_name=None):
        if out_name is not None:
            outputs = network.extra.get('outputs', [])
            if isinstance(outputs, dict): # in case of dict
                outputs = list(outputs.values())
            outputs += list(network.outputs)
            out = misc.find_opr_by_name(outputs, out_name)
            if len(out) == 0:
                print("Could not find {}".format(out_name))
                return self._info
        else:
            out = network.outputs

        network = RawNetworkBuilder(inputs=[], outputs=out)
        self._info['#oprs'] = get_num_oprs(out)
        self._info['#params'] = get_num_params(out)

        self.param_stats(network)
        self.flops_stats(network)
        return self._info


    def param_stats(self, network):
        if not network_initialized(network):
            initialize_network(network)

        bitwidth = network.extra.get("meta", {}).get("bitwidth", {})

        tot_param_dim, param_size_bit = 0, 0
        for param in network_visitor(network).all_params:
            param_dim = np.prod(param.imm_shape)

            nbits = bitwidth.get(param.name, 32)
            tot_param_dim += int(param_dim)
            param_size_bit += param_dim * nbits

        param_size = sizeof_fmt(param_size_bit / 8)
        self._info['tot_param_dim'] = tot_param_dim
        self._info['param_size'] = param_size
        self._param_size = param_size_bit / 8


    def flops_stats(self, network):
        from neupeak.model.model_manip.stats import get_flops_single_opr

        total_flops = 0
        batch_size = (
            get_visitor(network)
            .all_data_providers[0].partial_shape[0]
        )
        for opr in iter_dep_opr(*network.outputs):
            flops = get_flops_single_opr(opr) // batch_size
            if flops == 0:
                continue
            total_flops += flops

        total_flops_str = sizeof_fmt(total_flops, suffix='OPs')
        self._info['total_flops'] = total_flops_str
        self._info['flops/param_size'] = '{:.3g}'.format(
            total_flops / self._param_size)

def main(model, devices, args_input, caches):
    import argparse
    import collections

    parser = argparse.ArgumentParser()
    parser.add_argument('--lm_name', default='s-pred')
    args, unknownargs = parser.parse_known_args(args_input)

    # load model
    net = load_network(model)
    # net = net['network'] if not isinstance(net, RawNetworkBuilder) else net

    # init TestUnit
    t = TestUnit(caches)
    res = t.compute_metric(
        network = net,
        out_name = args.lm_name,
    )
    res_wrapper = collections.OrderedDict([('model_info', res)])
    return res_wrapper




