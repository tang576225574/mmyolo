# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import torch
import inspect
from collections import OrderedDict
from typing import Dict, Optional, Sequence, Union
from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.model import is_model_wrapper
from mmyolo.registry import HOOKS
from line_profiler import LineProfiler
from mmengine.dist import get_dist_info
DATA_BATCH = Optional[Union[dict, tuple, list]]
SUFFIX_TYPE = Union[Sequence[str], str]


def _sync_cuda_func(func):
    def _func(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return func(*args, **kwargs)
    return _func


@HOOKS.register_module()
class TimeMonitorHook(Hook):
    def __init__(self,
                 monitor_funcs=[],
                 monitor_interval='epochs',
                 sync_cuda=False,
                 **kwargs):
        super().__init__(**kwargs)
        assert monitor_interval in ['iters', 'epochs', 'total']
        self.monitor_funcs = monitor_funcs
        self.monitor_interval = monitor_interval
        self.sync_cuda = sync_cuda
        self.lp = None
        self.finished = False
        _rank, _world_size = get_dist_info()
        if _rank != 0:
            self.monitor_interval = None

    def before_train(self, runner) -> None:
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        dataset = runner.train_dataloader.dataset
        optimizer = runner.optim_wrapper
        for monitor_idx, monitor_str in \
            enumerate(self.monitor_funcs):
            assert (monitor_str.startswith(('model.', 'dataset.', 'optimizer.'))), \
                'monitor_funcs only supports the types of model, dataset or optimizer'
            try:
                monitor_func = eval(monitor_str)
                self.monitor_funcs[monitor_idx] = monitor_func
                if self.sync_cuda:
                    if monitor_str == 'dataset.__getitem__':
                        print_log('the param sync_cuda for dataset.__getitem__ is not supported')
                        continue
                    if isinstance(monitor_func, torch.nn.Module):
                        exec(f'{monitor_str}.forward=_sync_cuda_func(self.monitor_funcs[monitor_idx].forward)')
                    else:
                        exec(f'{monitor_str}=_sync_cuda_func(self.monitor_funcs[monitor_idx])')
            except Exception as e:
                print_log(f'can not find the monitor monitor_func: {monitor_str}')
                raise e
        if self.monitor_interval == 'total':
            self.start(runner)

    def after_train(self, runner) -> None:
        if (self.monitor_interval == 'total') and \
            (self.lp is not None):
            self.stop(runner)

    def before_train_epoch(self, runner) -> None:
        if self.monitor_interval == 'epochs':
            self.start(runner)

    def after_train_epoch(self, runner) -> None:
        if (self.monitor_interval == 'epochs') and \
            (self.lp is not None):
            self.stop(runner)

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        pass

    def start(self, runner) -> None:
        if self.finished:
            return
        self.lp = LineProfiler()
        for monitor_func in self.monitor_funcs:
            self.lp.add_function(monitor_func)
        self.lp.enable()

    def stop(self, runner) -> None:
        if self.finished:
            return
        lp_dump_path = osp.join(runner.work_dir, 'line_profiler.txt')
        self.lp.disable()
        self.lp.print_stats()
        self.lp.dump_stats(lp_dump_path)
        self.finished = True
