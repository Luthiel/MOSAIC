from collections import deque

import torch
from typing import List

class TreeNode(object):
    def __init__(self, idx=None):
        self.idx = idx
        self.children = []
        self.parent = None
        self.timestamp = None
        self.depth = None
        self.x = None
        self.degree = 0

def construct_tree(x: torch.tensor, edge_index: torch.tensor, timestamps: List[int], depths: List[int], event_id) -> List[TreeNode]:
    root = TreeNode(0)
    root.timestamp = timestamps[0]
    root.depth = depths[0]
    root.x = x[0]
    
    nodes = [root]
    for source, target in zip(edge_index[0], edge_index[1]):
        cur = TreeNode(target)
        try:
            parent = nodes[source]
        except:
            print(event_id)
        cur.parent = parent
        cur.timestamp = timestamps[target]
        cur.depth = depths[target]
        cur.x = x[target]
        nodes.append(cur)
        parent.children.append(cur)
        parent.degree += 1
    return nodes    