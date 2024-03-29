import os, yaml
from easydict import EasyDict
from configparser import ConfigParser


def get_configs(config_path: str, filename: str):
    with open(os.path.join(config_path, filename)) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        args = EasyDict(config)

        with open(
            os.path.join(config_path, f"model_configs/{args.model_name}.yaml")
        ) as file:
            model_config = yaml.load(file, Loader=yaml.FullLoader)
            args.update(model_config)

    return args
