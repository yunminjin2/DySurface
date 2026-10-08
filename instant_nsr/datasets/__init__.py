datasets = {}


def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def make(name, config):
    return datasets[name](config)


from . import blender_dynamic  # noqa: F401 - registers the D-NeRF dataset
