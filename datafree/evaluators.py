from . import metrics
import paddle
from tqdm import tqdm

class Evaluator(object):
    def __init__(self, metric, dataloader):
        self.dataloader = dataloader
        self.metric = metric

    def eval(self, model, progress=False):
        self.metric.reset()
        model.eval()
        with paddle.no_grad():
            try:
                for i, (inputs, targets, _) in enumerate(tqdm(self.dataloader, disable=not progress)):
                    inputs, targets = inputs.cuda(), targets.cuda()
                    outputs = model(inputs)
                    self.metric.update(outputs, targets)
            except:
                for i, (inputs, targets) in enumerate(tqdm(self.dataloader, disable=not progress)):
                    inputs, targets = inputs.cuda(), targets.cuda()
                    outputs = model(inputs)
                    self.metric.update(outputs, targets)
        return self.metric.get_results()
    
    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)

def classification_evaluator(dataloader):
    metric = metrics.MetricCompose({
        'Acc': metrics.TopkAccuracy(),
        'Loss': metrics.RunningLoss(paddle.nn.CrossEntropyLoss(reduction='sum'))
    })
    return Evaluator(metric, dataloader=dataloader)