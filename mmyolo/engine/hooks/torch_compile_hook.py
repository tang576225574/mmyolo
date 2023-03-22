# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmyolo.registry import HOOKS
from mmengine.hooks import Hook


@HOOKS.register_module()
class TorchCompileHook(Hook):
    def __init__(self,
                 mode='default',
                 **kwargs):
        super().__init__(**kwargs)
        allowed_mode = ['default', 'reduce-overhead', 'max-autotune']
        self.mode = mode
        assert mode in allowed_mode, \
            f'{mode} is not supported, allowed_mode: {allowed_mode}'

    def before_train(self, runner) -> None:
        model = runner.model
        compiled_model = torch.compile(model)
        runner.model = compiled_model
