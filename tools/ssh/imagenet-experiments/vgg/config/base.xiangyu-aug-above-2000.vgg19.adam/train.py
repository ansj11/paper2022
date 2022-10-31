#!/usr/bin/env mdl
import os
import time
import socket
from setproctitle import setproctitle

import megskull
from megskull.utils import logconf
from meghair.utils.misc import ensure_dir
from meghair.train.env import Action

from neupeak import optimizer
from neupeak.dataset.server import (
    create_remote_dataset, create_remote_combiner_dataset,
    create_remote_combiner_dataset_auto_desc
)
from neupeak.train.env import NeupeakTrainingEnv, log_rate_limited
from neupeak.utils.fs import change_dir, make_symlink_if_not_exists
from neupeak.train.utils import vars_to_combiner_descriptor
from neupeak.utils.misc import get_input_vars_of

from common import config
import model
import dataset as dataset_desc


def main():
    setproctitle(config.exp_name)

    ensure_dir(config.log_model_dir)
    # make symlink of training log
    with change_dir(config.base_dir):
        make_symlink_if_not_exists(
            os.path.relpath(config.log_dir, config.base_dir),
            os.path.join(config.base_dir, 'train_log'),
            overwrite=True)

    with NeupeakTrainingEnv(config.exp_name, config.log_dir) as env:
        logger = logconf.get_logger(__name__)

        dataset_names = ['train', 'validation']
        datasets = {
            name: dataset_desc.get(name)
            for name in dataset_names
        }

        # check servable names to prevent from using the same dataset
        # for training and validation
        servable_names = [getattr(ds, 'servable_name') for name, ds in datasets.items()]
        assert len(set(servable_names)) == len(servable_names), (
            'duplicated servable names: {}'.format(servable_names))

        net = model.get()


        if env.args.local:
            logger.info('use local dataset')
        else:
            # we use remote dataset by default
            logger.info('use remote dataset')
            datasets = {
                name: create_remote_combiner_dataset_auto_desc(
                    ds.servable_name, ds.nr_minibatch_in_epoch)
                for name, ds in datasets.items()
            }

        train_func = env.make_func_from_loss_var(net.loss_var, 'train', train_state=True)
        val_func = env.make_func_from_loss_var(net.loss_var, 'val', train_state=False, auto_parallel=False)

        # a large learning rate, not used actually
        opt = megskull.optimizer.AdamV8(learning_rate=10) # , momentum=0.9)

        # opt = optimizer.utils.clip_gradient_inplace(opt, 0.0001)
        opt(train_func)

        # you still need share working space manually
        train_func.comp_graph.share_device_memory_with(val_func.comp_graph)

        """
        watch all params RMS & gradRMS only on training function
        We use `interval=50` because I don't need paramRMS every minibatch...
        You can ignore interval if you like.
        """
        env.monitor.watch_paramRMS(net).on(train_func, interval=200)
        env.monitor.watch_gradRMS(net).on(train_func, interval=200)
#         env.monitor.watch(net.loss_var, mode='numeric') \
#             .on(train_func, askey='train_loss')

        monitor_vars = list(net.extra \
            .get("extra_config", {}) \
            .get('monitor_vars', []))
        outspec = {'loss': net.loss_var}
        outspec.update(net.extra.get("extra_outputs", {}))

        if 'loss' not in monitor_vars:
            monitor_vars.append('loss')

        for mv in monitor_vars:
            env.monitor.watch(outspec[mv], mode='numeric') \
                .on(train_func, askey='train_{}'.format(mv))

        def is_epoch_start():
            return env.clock.now[0] > 0 and env.clock.now[1] == 0

        log_epoch = env.worklog.get_logger(flush_on='epoch.done')

        env.worklog.watch_paramRMS(net).on(
            train_func, logger=log_epoch, should_record=is_epoch_start)
        env.worklog.watch_gradRMS(net).on(
            train_func, logger=log_epoch, should_record=is_epoch_start)

        # after done all decorations, compile the function
        train_func.compile(outspec)
        val_func.compile(outspec)

        env.register_checkpoint_component("network", net)
        env.register_checkpoint_component("opt_state", train_func.optimizer_state)

        def get_learning_rate(epoch):
            if epoch <= 30: return 1e-3
            if epoch <= 60: return 1e-4
            return 1e-5

        # save at beginning
        start_model = os.path.join(config.log_model_dir, "start")
        env.save_checkpoint(start_model)

        def save_training_id():
            # save training id
            training_id_path = os.path.join(config.log_dir, 'training_id.txt')
            with open(training_id_path, 'w') as f:
                f.write(str(env.monitor.training_id) + '\n')

            # save movnitor url
            monitor_url = 'http://monitor-sm.brain.megvii-inc.com/?trainings={}'.format(
                        env.monitor.training_id)
            monitor_url_path = os.path.join(config.log_dir, 'monitor_url.txt')

            with open(monitor_url_path, 'w') as f:
                f.write(monitor_url + '\n')

            # save hostname
            hostname_path = os.path.join(config.log_dir, 'hostname.txt')
            hostname = socket.gethostname()
            with open(hostname_path, 'w') as f:
                f.write(hostname)

            # save all-in-one info
            info_path = os.path.join(config.log_dir, 'training_info.txt')
            with open(info_path, 'w') as f:
                f.write('training_id\t{}\n'.format(env.monitor.training_id))
                f.write('monitor_url\t{}\n'.format(monitor_url))
                f.write('hostname\t{}\n'.format(hostname))
                f.write('base_dir\t{}\n'.format(
                    os.path.dirname(os.path.realpath(__file__))))

        def save_model():
            epoch = env.clock.now[0]

            latest_model = os.path.join(config.log_model_dir, "latest")
            epoch_model = os.path.join(
                config.log_model_dir, "epoch-{}".format(epoch))

            env.save_checkpoint(latest_model)
            if epoch % 5 == 0:
                env.save_checkpoint(epoch_model)

            # TODO: full-validation at the end of epoch for model selection

        # infinitely lone validation stream
        def gen_inf_iter_from_dataset(ds):
            def get_inf_iter_ds():
                while True:
                    yield from ds.get_epoch_minibatch_iter()
            return iter(get_inf_iter_ds())

        val_ds_names = ['validation']
        val_datasets = [datasets[name] for name in val_ds_names]

        val_ds_inf_iters = dict(zip(
            val_ds_names,
            list(map(gen_inf_iter_from_dataset, val_datasets))))

        def do_work():
            save_training_id()

            epoch = env.clock.now[0]
            minibatch = env.clock.now[1]

            if epoch > 10000:
                return Action.EXIT

            # adjust learning rate
            lr = get_learning_rate(epoch)
            opt.learning_rate = lr
            env.monitor.put('learning_rate', lr, env.monitor.NUMERIC)

            log_output = log_rate_limited(min_interval=1)(env.worklog.log)

            train_ds = datasets['train']

            time_epoch_start = tstart = time.time()
            for idx, minibatch in enumerate(train_ds.get_epoch_minibatch_iter()):
                tdata = time.time() - tstart
                out = train_func(**minibatch.get_kvmap())

                cur_time = time.time()
                time_passed = cur_time - time_epoch_start
                ttrain = cur_time - tstart

                time_expected = time_passed / (idx + 1) * train_ds.nr_minibatch_in_epoch
                eta = time_expected - time_passed

                outputs = [
                    "b:{}/{}".format(idx, train_ds.nr_minibatch_in_epoch),
                    "{:.2g} mb/s".format(1./ttrain),
                ] + [
                    'passed:{:.2f}'.format(time_passed),
                    'eta:{:.2f}'.format(eta),
                ] + [
                    "{}:{:.2g}".format(k, float(v)) for k, v in out.items()
                ]
                if tdata/ttrain > .05:
                    outputs += ["dp/tot: {:.2g}".format(tdata/ttrain)]


                post_proc_start_time = time.time()
                env.monitor.put("train-perpare-time", tdata, env.monitor.NUMERIC)
                env.monitor.put("train-total-time", ttrain, env.monitor.NUMERIC)

                # Validate on the go
                if idx % 100 == 0:
                    for dsname, inf_iter in val_ds_inf_iters.items():
                        try:
                            vb = next(inf_iter)
                        except StopIteration:
                            pass
                        else:
                            val_out = val_func(**vb.get_kvmap())
                            for mv in monitor_vars:
                                name = '{}:{}'.format(dsname, mv)
                                fval = float(val_out[mv])

                                env.monitor.put(name, fval, env.monitor.NUMERIC)
                                env.worklog.log('{}={}'.format(name, fval))

                env.event.trigger('minibatch.done')
                env.clock.tick(idx=-1)
                post_proc_duration = time.time() - post_proc_start_time

                outputs.append('post_proc:{:.2f}'.format(post_proc_duration))
                log_output(" ".join(outputs))

                tstart = time.time()


            save_model()

            env.event.trigger('epoch.done')
            env.clock.tick(idx=0)  # tick to next epoch

        env.invoke_forever(do_work)

if __name__ == '__main__':
    main()
