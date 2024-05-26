import dataclasses
from typing import Any, Iterator, List, Optional

import numpy as np
from mistral_common.tokens.tokenizers.sentencepiece import InstructTokenizerBase

from .args import DataArgs
from .dataset import build_dataset


@dataclasses.dataclass
class Batch:
    x: np.ndarray
    y: np.ndarray
    sizes: List[int]
    y_mask: Optional[np.ndarray] = None
    is_pad_only: bool = False

    def __post_init__(self):
        assert self.x.ndim == 1
        assert self.x.shape == self.y.shape
        assert self.x.dtype == np.int64
        assert self.y.dtype == np.int64
        assert isinstance(self.sizes, list)
        assert sum(self.sizes) == self.x.size == self.y.size

        if self.y_mask is not None:
            assert self.y_mask.size == self.y.size, (self.y_mask.shape, self.y.shape)
            assert self.y_mask.dtype == bool
            assert sum(self.sizes) == self.y_mask.size
            assert not self.y_mask.all()
            assert self.y_mask.any()

        if self.is_pad_only:
            assert np.sum(np.abs(self.y)) == 0
            assert np.sum(np.abs(self.x)) == 0
            assert self.y_mask is None
            # create all 0's mask for pad samples
            self.y_mask = np.zeros_like(self.x)




@dataclasses.dataclass
class BatchList:
    x: List[List[int]] = dataclasses.field(default_factory=list)
    y: List[List[int]] = dataclasses.field(default_factory=list)
    sizes: List[List[int]] = dataclasses.field(default_factory=list)
    y_mask: List[List[bool]] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        assert self.x == [], "`BatchList` has to be empty at init."
        assert self.y == [], "`BatchList` has to be empty at init."
        assert self.sizes == [], "`BatchList` has to be empty at init."
        assert self.y_mask == [], "`BatchList` has to be empty at init."

    def __len__(self) -> int:
        return len(self.x)

    def add(self, x: List[int], y: List[int], sizes: List[int], y_mask: List[bool]):
        self.x.append(x)
        self.y.append(y)
        self.sizes.append(sizes)
        self.y_mask.append(y_mask)

    def empty(self):
        self.x = []
        self.y = []
        self.sizes = []
        self.y_mask = []

    @staticmethod
    def flatten_to_numpy(list_of_lists: List[List[Any]], dtype: type) -> np.ndarray:
        return np.array([el for sublist in list_of_lists for el in sublist], dtype=dtype)

    def create_batch(self) -> Batch:
        x_np: np.ndarray = self.flatten_to_numpy(self.x, dtype=np.int64)
        y_np: np.ndarray = self.flatten_to_numpy(self.y, dtype=np.int64)
        sizes = sum(self.sizes, [])  # noqa

        y_mask_flatten = self.flatten_to_numpy(self.y_mask, dtype=bool)
        y_mask_np: Optional[np.ndarray] = None if y_mask_flatten.all() else y_mask_flatten

        return Batch(x_np, y_np, sizes, y_mask_np)




def build_data_loader(
    instruct_tokenizer: InstructTokenizerBase,
    args: DataArgs,
    batch_size: int,
    seq_len: int,
    seed: Optional[int],
    rank: int,
    world_size: int,
    is_eval: bool,
) -> Iterator[Batch]:
    pretrain_data = args.data if not is_eval else ""
    instruct_data = args.instruct_data if not is_eval else args.eval_instruct_data

    dataset = build_dataset(
        pretrain_data=pretrain_data,
        instruct_data=instruct_data,
        instruct_args=args.instruct,
        instruct_tokenizer=instruct_tokenizer,
        seq_len=seq_len,
        seed=seed,
        rank=rank,
        world_size=world_size,
        is_eval=is_eval,
        shuffle_pretrain=args.shuffle,
    )

    batch_list = BatchList()
    for sample in dataset:
        assert all(s >= 0 for s in sample.sizes)

        batch_list.add(sample.x, sample.y, sample.sizes, sample.mask)

        if len(batch_list) == batch_size:
            batch: Batch = batch_list.create_batch()
            yield batch

            batch_list.empty()

