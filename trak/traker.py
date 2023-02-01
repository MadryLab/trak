from typing import Iterable, Optional
import torch as ch
from torch.nn.parameter import Parameter
from torch import Tensor

class TRAKer():
    def __init__(self, save_dir: str, load_from_existing: bool = False):
        self.save_dir = save_dir
        self.last_ind = 0
        # self.mmap = open_mmap
    
    def featurize(self, val: Tensor, 
                        params: Iterable[Parameter], 
                        inds: Optional[list[int]] = None,
                        functional: bool = False):
        # Make sure param grads are zero
        if not functional:
            val.backward()
            g = None
        else:
            # from functorch import vmap, grad
            # functorch stuff
            pass

    def finalize(self, out_dir: Optional[str] = None, 
                       cleanup: bool = False, 
                       agg: bool = False):
        pass

    def score(self, val: Tensor, params: Iterable[Parameter]) -> Tensor:
        return ch.zeros(1, 1)