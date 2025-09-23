import torchmetrics

from .configurationmanager import Configuration, Hyperparameters

class MetricsFactory:

    METRIC_MODULES = [
        torchmetrics,
        torchmetrics.classification,
        torchmetrics.regression,
        torchmetrics.text,
        torchmetrics.image,
        torchmetrics.audio,
    ]

            

    @classmethod
    def create_metric(self, metric_name, **kwargs):
        metric_class = None
        for module in self.METRIC_MODULES:
            try:
                self.get_metric(module, metric_name)
                break
            except AttributeError:
                continue
        return metric_class(**kwargs)

    @staticmethod
    def get_metric(module, metric_name):
        metric_cls = getattr(module, metric_name)
        return metric_cls
    

    """
    The configuration must come with a metrics parameter like:
    configuration.metrics = {
            'train': 
                {
                 'MulticlassAccuracy':
                    {'average':'macro'},
                 'MulticlassAccuracyWithMicro':
                    {'average':'micro'}
                },
            'eval':{},
            'val':{}
            }
    The structure is {pipeline step: {metric: {parameter: value}}}
    """
    def create_metrics_collection(self,configuration: Configuration, hyperparameters: Hyperparameters):
        
        metrics_collection = {}

        for i,v in configuration.metrics.items():
            parameters = v.values()
            if hyperparameters.num_classes is not None:
                parameters['num_classes'] = hyperparameters.num_classes
            metric = self.create_metric(v.keys(),parameters)
            metrics_collection[i] = metric

        return metrics_collection