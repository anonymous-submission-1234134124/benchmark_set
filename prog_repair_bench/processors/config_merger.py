from omegaconf import OmegaConf, DictConfig
from pathlib import Path

def paths_to_abs_str(cfg: DictConfig) -> DictConfig:
    # Replace every Path object with its string representation
    def convert(v):
        if isinstance(v, Path):
            return str(v.resolve())
        return v

    def recurse(obj):
        if isinstance(obj, DictConfig):
            return OmegaConf.create({k: recurse(obj[k]) for k in obj.keys()})
        elif isinstance(obj, list):
            return [recurse(i) for i in obj]
        else:
            return convert(obj)

    return recurse(cfg)

def load_config_with_default(user_cfg_path: str | Path) -> DictConfig:

    default_cfg_path = Path(__file__).parent.parent / "resources" / "config_default.yaml"
    default_cfg = OmegaConf.load(default_cfg_path)
    user_cfg = OmegaConf.load(user_cfg_path)

    return OmegaConf.merge(default_cfg, user_cfg)