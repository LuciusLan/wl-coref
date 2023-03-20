""" Contains functions not directly linked to coreference resolution """

from typing import List, Set

import torch
from sklearn.metrics import f1_score

from coref.const import EPSILON


class GraphNode:
    def __init__(self, node_id: int):
        self.id = node_id
        self.links: Set[GraphNode] = set()
        self.visited = False

    def link(self, another: "GraphNode"):
        self.links.add(another)
        another.links.add(self)

    def __repr__(self) -> str:
        return str(self.id)


def add_dummy(tensor: torch.Tensor, eps: bool = False):
    """ Prepends zeros (or a very small value if eps is True)
    to the first (not zeroth) dimension of tensor.
    """
    kwargs = dict(device=tensor.device, dtype=tensor.dtype)
    shape: List[int] = list(tensor.shape)
    shape[1] = 1
    if not eps:
        dummy = torch.zeros(shape, **kwargs)          # type: ignore
    else:
        dummy = torch.full(shape, EPSILON, **kwargs)  # type: ignore
    return torch.cat((dummy, tensor), dim=1)

def non_max_sup(candidates, thres=0.95):
    """
    Using Non-Maximum Suppression (NMS) to filter out highly similar spans
    """
    pick = []
    filtered = {}
    for i, row in enumerate(candidates):
        filtered[i]= row
    while len(filtered) > 0:
        last = list(filtered.keys())[-1]
        pick.append(filtered[last])
        for key in list(filtered.keys()):
            if key == last:
                continue
            f = f1_score(filtered[last], filtered[key], zero_division=1, average='macro')
            if f < thres:
                pick.append(filtered[key])
            filtered.pop(key)
        filtered.pop(last)
    return pick