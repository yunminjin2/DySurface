import torch
systems = {}


def register(name):
    def decorator(cls):
        systems[name] = cls
        return cls
    return decorator


def make(name, config, load_from_checkpoint=None):
    
    if load_from_checkpoint is None:
        system = systems[name](config)
    else:
        system = systems[name].load_from_checkpoint(load_from_checkpoint, strict=False, config=config)
        model = torch.load(load_from_checkpoint)
        system.custom_global_step = model['global_step']
    return system


from . import sparse_neus_gags  # noqa: F401 - registers the system
