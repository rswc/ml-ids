from river.base.drift_detector import DriftDetector
from .base import ModelAdapterBase, DriftModelAdapterBase
from .mixin import PerClassMetricsMixin
from cbce import CBCE
from river.forest import ARFClassifier
from rbc import ResamplingBaggingClassifier
from collections import defaultdict


def extract_mean(prop, models):
    return sum(map(lambda m: getattr(m, prop, 0.0), models)) / len(models)


class CBCEAdapter(PerClassMetricsMixin, DriftModelAdapterBase[CBCE]):

    @classmethod
    def get_target_class(self):
        return CBCE

    def _warning_detectors_separate(self) -> bool:
        return False
    
    def _get_drift_prototype(self, model: CBCE):
        return model.drift_detector
    
    def _get_drift_detectors(self) -> dict[str, DriftDetector]:
        return self._model.drift_detectors

    def _get_inner_classifiers(self) -> dict:
        return self._model.classifiers
    
    def get_loggable_state(self) -> dict:
        state = {
            "class_priors": self._model._class_priors,
        }

        return self.add_drift_state(self.add_per_class_state(state), active_classes=self._model.classifiers.keys())

class ARFAdapter(PerClassMetricsMixin, DriftModelAdapterBase[ARFClassifier]):
    
    # Params specified in river.tree.HoeffdingTree 
    BASE_ATTRIBUTES_ALL = [
        "n_nodes", "n_branches", "n_leaves", 
        "n_active_leaves", "n_inactive_leaves", "height", 
        "_train_weight_seen_by_model"
    ]

    def __init__(self, *args, base_tree_attributes: list = BASE_ATTRIBUTES_ALL, track_background: bool = True, track_weighted_vote: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.base_tree_attributes = base_tree_attributes
        self.track_background = track_background
        self.track_wv = track_weighted_vote
    
    @classmethod
    def get_target_class(self) -> type[ARFClassifier]:
        return ARFClassifier

    # Required by PerClassMetricsMixin
    def _get_inner_classifiers(self) -> dict:
        return self._model.models

    def __calc_mean_attributes(self, for_background: bool = False) -> dict:
        models = self._model._background if for_background else self._model.models

        # One pass over models
        means = defaultdict(float) 
        cnt_models = 0
        for tree in models:
            # Background models can be None
            if tree is None:
                continue 

            for attr in self.base_tree_attributes:
                # Potentially None after drift warning
                attr_val = getattr(tree, attr)
                if attr_val is not None:
                    means[attr] += attr_val 
            cnt_models += 1
            
        if cnt_models > 0:
            means = dict([(f"{'bckg.' if for_background else ''}ht.attr.{attr}.mean", sum_val / cnt_models) for attr, sum_val in means.items() ])
        return means
    
    def __calc_wv_metric_stats(self) -> dict:
        """Get weighted vote metric values for active trees"""
        wv_metric_values = [ m.get() for m in self._model._metrics ]
        metric_name = self._model.metric.__class__.__name__

        return {
            f"wv.{metric_name}.mean": sum(wv_metric_values) / len(wv_metric_values),
            f"wv.{metric_name}.max": max(wv_metric_values),
            f"wv.{metric_name}.min": min(wv_metric_values)
        }
        
    def _warning_detectors_separate(self) -> bool:
        return True
        
    def _get_drift_prototype(self, model: ARFClassifier) -> DriftDetector:
        return model.drift_detector

    def _get_drift_warning_prototype(self, model: ARFClassifier) -> DriftDetector:
        return model.warning_detector

    # Required by DriftModelAdapterBase 
    def _get_drift_detectors(self) -> dict[str, DriftDetector]:
        return dict(enumerate(self._model._drift_detectors))

    # Required by DriftModelAdapterBase 
    def _get_drift_warning_detectors(self) -> dict[str, DriftDetector]:
        return dict(enumerate(self._model._warning_detectors))

    def get_loggable_state(self) -> dict:

        # Current state is means of all base trees 
        models_attribs = self.__calc_mean_attributes()
        background_attribs = {}
        if self.track_background:
            background_attribs = self.__calc_mean_attributes(for_background=True)
            background_attribs['bckg.n_models'] = sum(1 if tree is not None else 0 for tree in self._model._background)
            
        wv_stats = {}
        if self.track_wv:
            wv_stats = self.__calc_wv_metric_stats()
            
        state = { **models_attribs, **wv_stats, **background_attribs }
        
        return self.add_drift_state(self.add_per_class_state(state))


class RBCAdapter(PerClassMetricsMixin, ModelAdapterBase[ResamplingBaggingClassifier]):

    @classmethod
    def get_target_class(self) -> type[ResamplingBaggingClassifier]:
        return ResamplingBaggingClassifier

    def _get_inner_classifiers(self) -> dict:
        return self._model.models

    def get_loggable_state(self) -> dict:
        state = {
            "class_priors": self._model._class_priors,
        }

        return self.add_per_class_state(state)
