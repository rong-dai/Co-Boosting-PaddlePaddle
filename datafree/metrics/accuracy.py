import numpy as np
import paddle
from .stream_metrics import Metric
from typing import Callable

__all__ = ['Accuracy', 'TopkAccuracy']

class Accuracy(Metric):
    def __init__(self):
        self.reset()

    @paddle.no_grad()
    def update(self, outputs, targets):
        outputs = paddle.argmax(outputs, axis=1)
        self._correct += paddle.sum(outputs.flatten() == targets.flatten()).item()
        self._cnt += targets.numel()

    def get_results(self):
        return (self._correct / self._cnt * 100.)

    def reset(self):
        self._correct = 0.0
        self._cnt = 0.0


class TopkAccuracy(Metric):
    def __init__(self, topk=(1, 5)):
        self._topk = topk
        self.reset()

    @paddle.no_grad()
    def update(self, outputs, targets):
        for k in self._topk:
            _, topk_outputs = paddle.topk(outputs, k=k, axis=1, largest=True, sorted=True)
            correct = topk_outputs.equal(targets.reshape([-1, 1]).expand_as(topk_outputs))
            self._correct[k] += paddle.sum(correct[:, :k].flatten().astype('float32')).item()
        self._cnt += len(targets)

    def get_results(self):
        return tuple(self._correct[k] / self._cnt * 100. for k in self._topk)

    def reset(self):
        self._correct = {k: 0 for k in self._topk}
        self._cnt = 0.0
