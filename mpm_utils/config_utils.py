from pyhocon import ConfigFactory


def load_config(config_path: str):
    with open(config_path) as f:
        return dict(ConfigFactory.parse_string(f.read()))
