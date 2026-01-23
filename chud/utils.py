from pathlib import Path
import yaml

def deep_merge(base, override):
    results = base.copy()
    for key, value in override.items():
        if key in results and isinstance(results[key],dict) and isinstance(value, dict):
            results[key] = deep_merge(results[key], value)
        else:
            results[key] = value
    return results

def load_yaml(filepath):
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileExistsError(f'No config file found.')
    
    with open(filepath) as f:
        return yaml.safe_load(f) or {}
    
def save_yaml(data, filepath, **kwargs):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    defaults = {
        "default_flow_style": False,
        "sort_keys": False,
        "allow_unicode": True,
    }
    defaults.update(kwargs)
    
    with open(filepath, 'w') as f:
        yaml.dump(data, f, **defaults)

def load_config(config_path, defaults = None, verbose = True):
    config = defaults.copy() if defaults else {}

    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            user_config = load_yaml(config_path)
            config = deep_merge(config, user_config)
            if verbose:
                print(f"Loaded config from {config_path}")
        else:
            if verbose:
                print(f"Config file not found: {config_path}")
                print("Using default configuration")
    else:
        if verbose:
            print("Using default configuration")
    
    return config

def apply_overrides(config, overrides, key_map = None):
    key_map = key_map or {}

    for key, value in overrides.items():
        if value is None:
            continue

        path = key_map.get(key,key)

        if '.' in path:
            parts = path.split('.')
            target = config
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value
        else:
            config[path] = value

    return config

class Config:
    def __init__(self, data = None):
        self._data = data or {}
        for key, value in self._data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __getattr__(self, name):
        if name.startswith("_"):
            return super().__getattribute__(name)
        return None
    
    def __getitem__(self, key):
        return self._data.get(key)
    
    def get(self, key, default = None):
        return self._data.get(key, default)
    
    def to_dict(self):
        return self._data
    
    def __repr__(self):
        return f"Config({self._data})"

"""
Project-specific functions.
"""

def project_config(config_path = None, task = 'train', **overrides):
    config = load_config(config_path)

    if task == 'scrape':
        pass
    elif task == 'train':
        override_map = {
            "model": "model.name",
            "mode": "data.mode",
            "input": "data.input_file",
            "output": "training.output_dir",
            "epochs": "training.epochs",
            "batch_size": "training.batch_size",
            "learning_rate": "training.learning_rate"
        }
        config = apply_overrides(config, overrides, override_map)

    return config