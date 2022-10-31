from setproctitle import setproctitle

import os

import megbrain as mgb
from megskull.utils import logconf
from meghair.utils import io
from meghair.utils.misc import ensure_dir

from neupeak.train.utils import TrainClock
from neupeak.train.logger.tensorboard_logger import TensorBoardLogger

from neupeak.utils.fs import make_symlink_if_not_exists
from neupeak.dataset.server import create_remote_combiner_dataset_auto_desc

from landstack.train.env import make_func
import numpy as np
import collections


logger = logconf.get_logger(__name__)
class Session:
    def __init__(self, config, devices, net=None, train_func=None):
        setproctitle(config.exp_name)

        # create log folders and symbolic link
        ensure_dir(config.log_dir)
        ensure_dir(config.log_model_dir)
        make_symlink_if_not_exists(config.log_dir, config.symlink_log_dir)
        make_symlink_if_not_exists(config.dp_dir, config.symlink_dp_dir, overwrite=True)
        self.config = config

        logconf.set_output_file(os.path.join(config.log_dir, 'log.txt'))

        self.net = net
        self.extra_info = collections.OrderedDict() # {key:[current_value, best_value, best_epoch, higher_is_better]}
        self.train_func = train_func
        self.clock = TrainClock()
        self.tb_loggers = []

        self.devices = devices

    def make_func(self, train_state=False, fast_run=True, **kwargs):
        func = make_func(train_state=train_state, fast_run=fast_run, devices=self.devices, **kwargs)
        if train_state:
            self.train_func = func
        return func

    def tensorboards(self, *names):
        self.tb_loggers = [
            TensorBoardLogger(os.path.join(self.config.log_dir, d)) for d in names
            ]
        return self.tb_loggers

    def start(self):
        self.save_checkpoint('start')
        for b in self.tb_loggers:
            b.put_start(self.clock.step)


    def get_train_ds(self, name='train'):
        train_ds = create_remote_combiner_dataset_auto_desc(
            self.config.dpflow[name],
            self.config.num_mini_batch_per_epoch_train
        )
        return train_ds

    def monitor_param_histogram(self, histogram_logger, rms_logger, interval=200):
        # Watch var and grad rms
        def get_var_watcher(key, interval, clock):
            def cb(gpu_tensor):
                nonlocal interval, clock, histogram_logger, rms_logger
                if clock.step % interval == 0:
                    var_value = gpu_tensor.get_value()
                    histogram_logger.put_tensor_as_histogram(key, var_value, clock.step)
                if clock.minibatch == 0:
                    var_value = gpu_tensor.get_value()
                    rms_logger.put_tensor_rms(key, var_value, clock.step)

            return mgb.callback_lazycopy(cb)

        loss_mgbvar = self.train_func.loss_mgbvar
        for param in self.net.loss_visitor.all_params:
            if param.freezed:
                continue
            name = param.name.replace(':', '/')  # tensorflow name convention
            param_var = self.train_func.get_mgbvar(param)
            grad_var = mgb.grad(loss_mgbvar, param_var)
            self.train_func.add_extra_outspec(
                (param_var, get_var_watcher(name, interval, self.clock))
            )
            self.train_func.add_extra_outspec(
                (grad_var, get_var_watcher(name+'/grad', interval, self.clock))
            )

    def save_checkpoint(self, name, old=None):
        ckp_path = os.path.join(self.config.log_model_dir, name)
        tmp = {
            'network': self.net,
            'opt_state': self.train_func.optimizer_state.make_checkpoint(),
            'clock': self.clock.make_checkpoint(),
            'extra_info': self.extra_info,
        }
        io.dump(tmp, ckp_path)

        if old is not None:
            old_ckp_path = os.path.join(self.config.log_model_dir, old)
            if os.path.exists(old_ckp_path):
                os.remove(old_ckp_path)

    def load_checkpoint(self, ckp_path):
        checkpoint = io.load(ckp_path, dict)
        self.net.loss_visitor.set_all_stateful_opr_from(checkpoint['network'].loss_visitor)
        try:
            self.train_func.optimizer_state.restore_checkpoint(checkpoint['opt_state'])
        except:
            pass
        self.clock.restore_checkpoint(checkpoint['clock'])
        self.extra_info = checkpoint['extra_info']

    # for logging
    def log_best_value(self, var_name, higher_is_better=True):
        if higher_is_better:
            self.extra_info.setdefault(var_name, [-np.inf, -np.inf, 0, True])
        else:
            self.extra_info.setdefault(var_name, [np.inf, np.inf, 0, False])

    def update_best_state(self, kv_dict, current_epoch):
        assert isinstance(kv_dict, dict)
        for k, v in kv_dict.items():
            if isinstance(v, (tuple, list, np.ndarray)):
                v = v[0] if len(v) == 1 else v
            if k in self.extra_info:
                _, old_best_value, old_best_epoch, flag = self.extra_info[k]
                old_ckpt_name = 'best-{}-{}'.format(old_best_epoch, k)
                new_ckpt_name = 'best-{}-{}'.format(current_epoch, k)
                if (flag and v > old_best_value) or (not flag and v < old_best_value):
                    self.extra_info[k] = [v, v, current_epoch, flag]
                    self.save_checkpoint(new_ckpt_name, old_ckpt_name)
                self.extra_info[k][0] = v


    def put_scalar_best(self, tb, current_step):
        for k, v in self.extra_info.items():
            tb.put_scalar('best_{}'.format(k), v[1], current_step)

    @staticmethod
    def put_scalar(tb, kv_dict, current_step):
        assert isinstance(kv_dict, dict)
        for k, v in kv_dict.items():
            if isinstance(v, (tuple, list, np.ndarray)):
                v = v[0]
            tb.put_scalar(k, v, current_step)








