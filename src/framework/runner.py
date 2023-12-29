import os
from datetime import datetime
import csv
import json
from river.base import Classifier
from river.datasets.base import Dataset, MULTI_CLF
from river.metrics.base import Metrics, BinaryMetric
from metrics import MetricWrapper
import wandb
from framework.adapters.base import ModelAdapterBase
from framework.util import *

class ExperimentRunner:
    """Helper class for running experiments in a standardized way.

    Parameters
    ----------
    model
        The river-compatible classifier to be tested.
    dataset
        The river-compatible dataset class for this experiment.
    metrics
        A `river.metrics.base.Metrics` object containing a group of metrics to be collected.
    out_dir
        The directory to which `.csv` logs will be saved.
    name
        (optional) A custom name for this experiment.
    model_adapter
        (optional) An adapter class for collecting aditional data from the model
    enable_tracker
        (default: `True`) Whether or not to use the tracker (currently, wandb) to log data
        from this experiment online.
    project
        (wandb only, optional) The name of the project under which this exepriment should be categorized.
    entity
        (wandb only, optional) The entity (user or team) which owns this experiment.
    
    """

    def __init__(
            self,
            model: Classifier,
            dataset: Dataset,
            metrics: Metrics,
            out_dir: str,
            name: str = None,
            model_adapter: ModelAdapterBase = None,
            enable_tracker: bool = True,
            project: str = None,
            entity: str = None,
        ) -> None:
        self.model = model
        self.dataset = dataset
        self.metrics = metrics
        self.out_dir = out_dir
        self.entity = entity
        self.project = project

        self.model_adapter = model_adapter
        if model_adapter:
            self.model_adapter.model = model

        self._enable_tracker = enable_tracker

        assert metrics.works_with(model), "Invalid metrics for model"
        assert self._metrics_match_dataset(), "Invalid use of binary metric for multiclass dataset"
        assert os.path.isdir(out_dir), f"{out_dir} is not a directory"

        time = datetime.now().strftime("%y-%m-%d_%H%M%S")
        dataset_name = dataset.__class__.__name__
        self._id = f"{time}_{str(model)}_{dataset_name}" + (f"_{name}" if name else "")
    
    @property
    def _parameters(self):
        """Parameters of the experiment."""
        return {
            "id": self._id,
            "model": self.model.__class__.__name__,
            "hyperparameters": self.model._get_params(),
            "dataset": self.dataset._repr_content,
            "metrics": self._metrics_names,
            "adapter": self.model_adapter.get_parameters() if self.model_adapter else None
        }
    
    def run(self):
        wandb.init(
            entity=self.entity,
            project=self.project,
            config=self._parameters,
            mode=["disabled", "online"][self._enable_tracker]
        )

        print("Starting experiment:", self._id)

        with open(self._meta_path, "x") as file_meta:
            json.dump(self._parameters, file_meta, default=lambda o: repr(o), indent=4)

        print("Metadata available at:", os.path.abspath(self._meta_path))
        print("Metrics log available at:", os.path.abspath(self._metrics_path))

        with open(self._metrics_path, "x", newline="") as file_metrics:
            writer_metrics = csv.writer(file_metrics)

            writer_metrics.writerow(self._metrics_names)

            # Training loop
            for x, y in self.dataset:
                y_pred = self.model.predict_one(x)
                self.model.learn_one(x, y)

                # Evaluation
                if y_pred is not None:
                    self.metrics.update(y, y_pred)
                    writer_metrics.writerow(self.metrics.get())

                    if self.model_adapter:
                        self.model_adapter.update(y, y_pred)
                        wandb.log({f"Model.{self.model.__class__.__name__}": self.model_adapter.get_loggable_state()}, commit=False)
                    
                    wandb.log(self._metrics_dict)

        print("Experiment DONE")
        wandb.finish()


    def _metrics_match_dataset(self) -> bool:
        """Check if there exist binary-only metric for multiclass dataset"""
        if self.dataset.task != MULTI_CLF:
            return True
        
        for m in self.metrics:
            if isinstance(m, BinaryMetric): 
                return False
            elif isinstance(m, MetricWrapper):
                if not m.works_with_multiclass:
                    return False
        return True
    
    @property
    def _metrics_names(self):
        return [extract_metric_name(metric) for metric in self.metrics]
    
    @property
    def _metrics_path(self):
        return os.path.join(self.out_dir, self._id + "_METRICS.csv")
    
    @property
    def _meta_path(self):
        return os.path.join(self.out_dir, self._id + "_META.json")
    
    @property
    def _metrics_dict(self):
        """Current values of the metrics, as a Python dict"""
        return get_metrics_dict(self.metrics)
