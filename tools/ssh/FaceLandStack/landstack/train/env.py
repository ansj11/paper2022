import numpy as np
from neupeak.utils.inference import get_fprop_env, FunctionMaker
from IPython import embed
import tqdm
import collections

def make_func(train_state=False, fast_run=True, devices='gpu0', **kwargs):
    func_maker = FunctionMaker.get_instance(devices=devices)
    if train_state:
        assert 'loss_var' in kwargs
        train_func = func_maker.make_func(
            optimizable=True,
            env=get_fprop_env(fast_run=fast_run, train_state=True),
            **kwargs,
        )
        return train_func

    val_func = func_maker.make_func(env=get_fprop_env(fast_run=fast_run))
    val_func._env.comp_graph.set_option('log_static_mem_alloc', False)
    val_func._env.flags.verbose_fprop = False
    val_func._env.flags.enforce_var_shape = False
    return val_func

def compile_inf_func(outputs, fast_run=False, devices='gpu0'):
    if isinstance(outputs, (list, tuple)):
        outputs = collections.OrderedDict([(v.name, v) for v in outputs])
    inf_func = make_func(False, fast_run, devices)
    inf_func.compile(outputs)
    return inf_func


def forward(func, data_dict, batch_size=np.inf, verbose=False):
    from megskull.graph.fprop import  Function
    from megskull.graph import iter_dep_opr
    # check and infer n_samples
    n_samples = [len(v) for v in list(data_dict.values())]
    assert all(x == n_samples[0] for x in n_samples), 'not equal samples'
    n_samples = n_samples[0]
    assert isinstance(func, Function)

    # convert type
    oprs_dict = collections.OrderedDict([(v.name, v) for v in list(iter_dep_opr(list(func.output_vars.values())))])
    for k in data_dict.keys():
        if k in oprs_dict:
            dtype = oprs_dict[k].dtype
            data_dict[k] = data_dict[k].astype(dtype)

    # forward
    batch_size = min(batch_size, n_samples)
    n_batch = int(np.ceil(n_samples / batch_size))
    res_dict = collections.OrderedDict()
    bs_iter = tqdm.tqdm(range(n_batch)) if verbose else range(n_batch)
    for i in bs_iter:
        data_dict_batch = {k:v[i*batch_size:(i+1)*batch_size] for k, v in data_dict.items()}
        r = func(**data_dict_batch)
        for k, v in r.items():
            res_dict.setdefault(k, []).append(v)
    outputs = func.output_names
    res_dict = collections.OrderedDict([(k, np.concatenate(res_dict[k])) for k in outputs]) # ensure order
    return res_dict

