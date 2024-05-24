from dataclasses import dataclass, field
from typing import Optional

from simple_parsing.helpers import Serializable


@dataclass
class LoraArgs(Serializable):
    enable: bool = True
    rank: int = 16
    dropout: float = 0.0
    scaling: float = 2.0

    def __post_init__(self):
        if self.enable:
            assert self.rank > 0
            assert self.scaling > 0.0


@dataclass
class MoeArgs(Serializable):
    num_experts: int = 8
    num_experts_per_tok: int = 2


@dataclass
class ModelArgs(Serializable):
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int
    rope_theta: float = 10000.0

    lora: LoraArgs = field(default_factory=LoraArgs)
    moe: Optional[MoeArgs] = None
