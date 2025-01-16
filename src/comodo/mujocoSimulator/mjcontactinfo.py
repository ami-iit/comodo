from dataclasses import dataclass
from typing import Sequence, List

class MjContactInfo:
    """Wrapper class for mjContact struct of MuJoCo.
    Accepts the struct instance individually or by unpacking the struct.    
    """
    def __init__(self, t: float, iter: int, *args):
        self.t = t
        self.iter = iter
        if len(args) == 1:
            self.mj_struct = args[0]
            self.dist: float = self.mj_struct.dist[0] if self.mj_struct.dist.size > 0 else None
            self.pos: List[float] = self.mj_struct.pos.flatten().tolist() if self.mj_struct.pos.size > 0 else None
            self.frame: List[float] = self.mj_struct.frame.flatten().tolist() if self.mj_struct.frame.size > 0 else None
            self.dim: int = self.mj_struct.dim[0] if self.mj_struct.dim.size > 0 else None
            self.geom: List[int] = self.mj_struct.geom.flatten().tolist() if self.mj_struct.geom.size > 0 else None
            self.flex: List[int] = self.mj_struct.flex.flatten().tolist() if self.mj_struct.flex.size > 0 else None
            self.elem: List[int] = self.mj_struct.elem.flatten().tolist() if self.mj_struct.elem.size > 0 else None
            self.vert: List[int] = self.mj_struct.vert.flatten().tolist() if self.mj_struct.vert.size > 0 else None
            self.mu: float = float(self.mj_struct.mu[0]) if self.mj_struct.mu.size > 0 else None
            self.H: List[float] = self.mj_struct.H.flatten().tolist() if self.mj_struct.H.size > 0 else None
            
            self._is_none = self.pos is None
        else:
            raise NotImplementedError("Unpacking the struct is not implemented yet.")

    def __str__(self):
        return f"ContactInfo(t={self.t}, iter={self.iter}, dist={self.dist}, pos={self.pos}, frame={self.frame}, dim={self.dim}, geom={self.geom}, flex={self.flex}, elem={self.elem}, vert={self.vert}, mu={self.mu}, H={self.H})"

    def __repr__(self):
        return f"ContactInfo(t={self.t}, iter={self.iter}, dist={self.dist}, pos={self.pos}, frame={self.frame}, dim={self.dim}, geom={self.geom}, flex={self.flex}, elem={self.elem}, vert={self.vert}, mu={self.mu}, H={self.H})"

    def is_none(self) -> bool:
        return self._is_none

    def get_time(self) -> float:
        return self.t
    