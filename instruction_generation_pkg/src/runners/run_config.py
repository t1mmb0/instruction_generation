from dataclasses import dataclass

@dataclass
class RunConfig:
    seeds: list[int]
    max_epochs: int
    compute_feature_importance: bool
    feature_names: list[str]
    fi_loader: str = "val"
