from .AVAGFormer import AVAGFormer


def build_model(type, **kwargs):
    if type == 'AVAGFormer':
        return AVAGFormer(**kwargs)
    else:
        raise ValueError
