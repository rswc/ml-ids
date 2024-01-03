from river.metrics.base import MultiClassMetric, BinaryMetric, ClassificationMetric
from river.utils import Rolling
from typing import Optional

class MetricWrapper(ClassificationMetric):
    """Wrapper class to easily add sliding window or multiclass collapse to any `ClassificationMetric`

    Parameters
    ----------
    metric
        A `river.metrics.base.ClassificationMetric` object to use inside wrapper.
    name
        (optional) A custom name for this wrapper. When not defined, wrapper name is created based on given parameters.
    window_size
        (optional) A window_size parameter for `river.utils.Rolling`. When not defined wrapper uses plain `metric` without `Rolling`  .
    collapse_label
        (optional) The target label for given `collapse_classes` multiclass collapse to binary. `True` for positive, `False` for negative.
    collapse_classes
        (optional) A list of current multiclass problem class labels which will belong to `collapse_label` binary class after collapse. 
    
    """

    def __init__(self, metric: ClassificationMetric, name: str = None, window_size: int = None, collapse_label: bool = None, collapse_classes=[]):
        self.metric = metric
        self.window_size = window_size
        self.collapse_label = collapse_label
        self.collapse_classes = sorted(collapse_classes)

        self.is_rolling = False
        self.is_collapsed = False

        #HACK: If metric is already wrapped in Rolling, unwrap so it can be
        #correctly wrapped again. This can happen when cloning the MetricWrapper.
        if isinstance(metric, Rolling):
            metric = metric.obj

        self.is_multiclass = isinstance(metric, MultiClassMetric)
        self.is_binary = isinstance(metric, BinaryMetric)
        
        # Boolean NOT(XOR)
        if not (self.is_multiclass or self.is_binary) or (self.is_multiclass and self.is_binary):
            raise ValueError(f"Given metric {self.metric} should inherit from only one of river's: [`MultiClassMetric`, `BinaryMetric`].")

        if self.window_size is not None:
            if not isinstance(self.window_size, int) or self.window_size <= 0:
                raise ValueError(f"Invalid value {window_size = }, only positive integers are supported")

            self.is_rolling = True
            self.metric = Rolling(metric, window_size=self.window_size)

        if self.collapse_label is not None:
            if self.is_multiclass:
                raise ValueError(f"Cannot collapse labels to binary with MultiClass metric {self.metric}.")
            elif not isinstance(self.collapse_label, bool):
                raise ValueError(f"Specified {collapse_label = } is not of type `bool`")
            elif len(collapse_classes) == 0:
                raise ValueError("Not specified `collapse_classes` but given `collapse_label`. Both variables should be defined to correctly collapse multiclass dataset.")

            self.is_collapsed = True
        elif len(collapse_classes) > 0:
            raise ValueError("Not specified `collapse_label` but given `collapse_classes`. Both variables should be defined to correctly collapse multiclass dataset.")

        # Check for `:` in name
        # due to current parsing of River __repr__ for Metric
        if name is not None and ':' in name:
            raise ValueError(f"Invalid name '{name}' - Character ':' is forbidden to use inside metric name.")
                
        self.name = name or self.__generate_name_str()
    
    @property
    def works_with_multiclass(self) -> bool:
        return self.is_multiclass or (self.is_binary and self.is_collapsed)
    
    def get(self) -> float:
        return self.metric.get()
        
    def update(self, y_true, y_pred, sample_weight=1.0):
        if self.is_collapsed:
            y_true = self.__collapse_lookup(y_true)
            y_pred = self.__collapse_lookup(y_pred)
        
        self.metric.update(y_true, y_pred, sample_weight=sample_weight)

    def revert(self, y_true, y_pred, sample_weight=1.0):
        if self.is_collapsed:
            y_true = self.__collapse_lookup(y_true)
            y_pred = self.__collapse_lookup(y_pred)

        self.metric.revert(y_true, y_pred, sample_weight=sample_weight)

    def __repr__(self):
        return f"{self.name}: {self.get():{self._fmt}}".rstrip("0")
    
    def __str__(self):
        return repr(self)

    def __collapse_lookup(self, label: str) -> Optional[bool]:
        assert self.is_collapsed
        return self.collapse_label if label in self.collapse_classes else (not self.collapse_label)
    
    def __generate_name_str(self) -> str:
        params = []
        rolling = ''
        if self.is_rolling:
            params.append(f"ws={self.window_size}")
            rolling = 'Rolling'
        if self.is_collapsed:
            params.append(f"{['negative','positive'][self.collapse_label]}[{','.join(str(c) for c in self.collapse_classes)}]")
        
        m_name = str(self.metric).split(':')[0]
        param_str = f"({';'.join(params)})" if len(params) > 0 else ''
        return f"{rolling}{m_name}{param_str}"
    
