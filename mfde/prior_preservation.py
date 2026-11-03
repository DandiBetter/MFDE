import copy
import torch
from torch import nn

class PriorPreservationTeacher(nn.Module):

    def __init__(self, student_model: nn.Module, decay: float=0.999, include_keys=None, exclude_keys=None):
        super().__init__()
        self.decay = decay
        self.include_keys = include_keys
        self.exclude_keys = exclude_keys or []
        self.teacher = copy.deepcopy(student_model)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            return self.teacher(*args, **kwargs)

    def _should_update(self, name: str) -> bool:
        if self.include_keys is not None:
            if not any((name.startswith(k) for k in self.include_keys)):
                return False
        if any((name.startswith(k) for k in self.exclude_keys)):
            return False
        return True

    @torch.no_grad()
    def update(self, student_model: nn.Module):
        student_state = student_model.state_dict()
        teacher_state = self.teacher.state_dict()
        for name, t_val in teacher_state.items():
            if not self._should_update(name):
                t_val.copy_(student_state[name])
                continue
            s_val = student_state[name]
            if torch.is_floating_point(t_val):
                t_val.mul_(self.decay).add_(s_val.detach(), alpha=1.0 - self.decay)
            else:
                t_val.copy_(s_val)

    @torch.no_grad()
    def to(self, *args, **kwargs):
        self.teacher = self.teacher.to(*args, **kwargs)
        return self
