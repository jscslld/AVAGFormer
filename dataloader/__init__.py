from .avaffordance import AVAffordanceDataset
from mmengine import Config


def build_dataset(type, split, **kwargs):
    if type == 'AVAffordanceDataset':
        return AVAffordanceDataset(split=split, cfg=Config(kwargs))
    else:
        raise ValueError


__all__ = ['build_dataset']
