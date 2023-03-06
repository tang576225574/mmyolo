# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import torch
import logging
from typing import Dict, Optional, Sequence, Union
from mmengine.hooks import LoggerHook
from mmengine.logging import print_log
from mmengine.model import is_model_wrapper
from mmyolo.registry import HOOKS
from line_profiler import LineProfiler
from mmengine.dist import get_dist_info
DATA_BATCH = Optional[Union[dict, tuple, list]]
SUFFIX_TYPE = Union[Sequence[str], str]


# def _sync_before_func(func):
#     def _func(*args, **kwargs):
#         if torch.cuda.is_available():
#             torch.cuda.synchronize()
#         return func(*args, **kwargs)
#     return _func

# def _sync_after_func(func):
#     def _func(*args, **kwargs):
#         _res = func(*args, **kwargs)
#         if torch.cuda.is_available():
#             torch.cuda.synchronize()
#         return _res
#     return _func


def _sync_cuda_func(func):
    def _func(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _res = func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return _res
    return _func

@HOOKS.register_module()
class TimeMonitorHook(LoggerHook):
    def __init__(self,
                 monitor_funcs=[],
                 monitor_type='epochs',
                 sync_cuda=False,
                 **kwargs):
        kwargs['out_suffix'] = ('.log', '.lprof')
        super().__init__(**kwargs)
        assert monitor_type in ['iters', 'epochs', 'total']
        self.monitor_funcs = monitor_funcs
        self.monitor_type = monitor_type
        self.sync_cuda = sync_cuda
        self.lp = None
        self.finished = False
        _rank, _world_size = get_dist_info()
        if _rank != 0:
            self.monitor_type = None

    def before_run(self, runner) -> None:
        if self.out_dir is not None:
            basename = osp.basename(runner.work_dir.rstrip(osp.sep))
            self.out_dir = self.file_backend.join_path(self.out_dir, basename)
            runner.logger.info(
                f'time monitor logs will be saved to {self.out_dir} after the '
                'training process.')
        self.lprof_log_path = f'{runner.timestamp}_line_profiler.lprof'
        self.text_log_path = f'{runner.timestamp}_line_profiler.log'

    def before_train(self, runner) -> None:
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        dataloader = runner.train_dataloader
        dataset = dataloader.dataset
        optimizer = runner.optim_wrapper
        for monitor_idx, monitor_str in \
            enumerate(self.monitor_funcs):
            assert (monitor_str.startswith(('model.', 'dataset.', 'dataloader.', 'optimizer.'))), \
                'monitor_funcs only supports the types of model, dataloader or optimizer'
            try:
                monitor_func = eval(monitor_str)
                if isinstance(monitor_func, torch.nn.Module):
                    monitor_func = monitor_func.forward
                    monitor_str = f'{monitor_str}.forward'
                self.monitor_funcs[monitor_idx] = monitor_func
                if self.sync_cuda:
                    if (monitor_str.startswith(('dataset.', 'dataloader.'))):
                        print_log(f'the param sync_cuda for {monitor_str} is not supported',
                                  level=logging.WARNING)
                        continue
                    # exec_obj = 'self.monitor_funcs[monitor_idx]'
                    # exec(f'{exec_obj} = _sync_after_func({exec_obj})')
                    # exec(f'{monitor_str}=_sync_before_func({exec_obj})')
                    exec(f'{monitor_str}=_sync_cuda_func(self.monitor_funcs[monitor_idx])')
            except Exception as e:
                print_log(f'can not find the monitor monitor_func: {monitor_str}')
                raise e
        if self.monitor_type == 'total':
            self.start(runner)

    def after_train(self, runner) -> None:
        if (self.monitor_type == 'total') and \
            (self.lp is not None):
            self.stop(runner)

    def before_train_epoch(self, runner) -> None:
        if self.monitor_type == 'epochs':
            self.start(runner)

    def after_train_epoch(self, runner) -> None:
        if (self.monitor_type == 'epochs') and \
            (self.lp is not None):
            self.stop(runner)

    def start(self, runner) -> None:
        if self.finished:
            return
        self.lp = LineProfiler()
        for monitor_func in self.monitor_funcs:
            if hasattr(monitor_func, '__wrapped__'):
                monitor_func = monitor_func.__wrapped__
            self.lp.add_function(monitor_func)
        self.lp.enable()

    def stop(self, runner) -> None:
        if self.finished:
            return
        self.lp.disable()
        with open(osp.join(runner.log_dir, self.text_log_path),
                  'w', encoding='utf-8') as fw:
            self.lp.print_stats(stream=fw)
        self.lp.dump_stats(osp.join(runner.log_dir,
                                    self.lprof_log_path))
        self.finished = True

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        pass

    def after_val_iter(self,
                       runner,
                       batch_idx: int,
                       data_batch: DATA_BATCH = None,
                       outputs: Optional[Sequence] = None) -> None:
        pass

    def after_test_iter(self,
                        runner,
                        batch_idx: int,
                        data_batch: DATA_BATCH = None,
                        outputs: Optional[Sequence] = None) -> None:
        pass

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        pass

    def after_test_epoch(self,
                         runner,
                         metrics: Optional[Dict[str, float]] = None) -> None:
        pass

    def after_run(self, runner) -> None:
        pass
