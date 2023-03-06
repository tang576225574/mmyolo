_base_ = '../yolov7/yolov7_l_syncbn_fast_8x16b-300e_coco.py'
# _base_ = '../yolov7/yolov7_tiny_syncbn_fast_8x16b-300e_coco.py'


custom_hooks = _base_.custom_hooks + [
    dict(
        type='TimeMonitorHook',
        monitor_funcs=[
            'model.loss',               # 总前向时间
            'model.backbone',           # backbone时间
            'model.neck',               # neck时间
            'model.bbox_head.loss_by_feat',     # 总的head推理时间
            'model.bbox_head.assigner',      # label assignment时间
            'model.bbox_head._calc_loss',    # 计算Loss的时间
            'dataset.__getitem__',          # 数据读取时间
            'optimizer.update_params'       # 更新参数的总时间
        ],
        sync_cuda=True,
        monitor_interval='total',
        priority=45
    )
]
