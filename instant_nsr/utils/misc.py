import os
from omegaconf import OmegaConf
from packaging import version


# ============ Register OmegaConf Recolvers ============= #
OmegaConf.register_new_resolver('calc_exp_lr_decay_rate', lambda factor, n: factor**(1./n))
OmegaConf.register_new_resolver('add', lambda a, b: a + b)
OmegaConf.register_new_resolver('sub', lambda a, b: a - b)
OmegaConf.register_new_resolver('mul', lambda a, b: a * b)
OmegaConf.register_new_resolver('div', lambda a, b: a / b)
OmegaConf.register_new_resolver('idiv', lambda a, b: a // b)
OmegaConf.register_new_resolver('basename', lambda p: os.path.basename(p))
# ======================================================= #


def prompt(question):
    inp = input(f"{question} (y/n)").lower().strip()
    if inp and inp == 'y':
        return True
    if inp and inp == 'n':
        return False
    return prompt(question)


def _load_config_with_base(yaml_file, stack=None):
    yaml_file = os.path.abspath(os.fspath(yaml_file))
    stack = [] if stack is None else stack
    if yaml_file in stack:
        cycle = " -> ".join(stack + [yaml_file])
        raise ValueError(f"Cyclic _base_ config reference: {cycle}")

    conf = OmegaConf.load(yaml_file)
    base_value = conf.pop("_base_", None)
    if base_value is None:
        return conf

    base_files = [base_value] if isinstance(base_value, str) else list(base_value)
    parent_confs = []
    for base_file in base_files:
        base_file = os.fspath(base_file)
        if not os.path.isabs(base_file):
            base_file = os.path.join(os.path.dirname(yaml_file), base_file)
        parent_confs.append(_load_config_with_base(base_file, stack + [yaml_file]))
    return OmegaConf.merge(*parent_confs, conf)


def load_config(*yaml_files, cli_args=[]):
    yaml_confs = [_load_config_with_base(f) for f in yaml_files]
    cli_conf = OmegaConf.from_cli(cli_args)
    conf = OmegaConf.merge(*yaml_confs, cli_conf)
    OmegaConf.resolve(conf)
    return conf


def config_to_primitive(config, resolve=True):
    return OmegaConf.to_container(config, resolve=resolve)


def dump_config(path, config):
    with open(path, 'w') as fp:
        OmegaConf.save(config=config, f=fp)

def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def parse_version(ver):
    return version.parse(ver)
