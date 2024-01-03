import base64
from copy import deepcopy
from pathos.multiprocessing import Pool
import os
from datetime import datetime
import csv
import json
from typing import Iterable
from river.base import Classifier
from river.datasets.base import Dataset, MULTI_CLF
from river.metrics.base import Metrics, BinaryMetric
from metrics import MetricWrapper
import wandb
from framework.adapters.base import ModelAdapterBase
from framework.util import *

class BaseRunner:

    def __init__(
            self,
            model: Classifier,
            dataset: Dataset,
            metrics: Metrics,
            out_dir: str,
            model_adapter: ModelAdapterBase = None,
            enable_tracker: bool = True,
            project: str = None,
            entity: str = None,
            notes: str = None,
            tags: list[str] = None
        ) -> None:
        self.model = model
        self.dataset = dataset
        self.metrics = metrics
        self.out_dir = out_dir
        self.entity = entity
        self.project = project
        self.notes = notes
        self.tags = tags

        self.model_adapter = model_adapter

        self._enable_tracker = enable_tracker

        assert metrics.works_with(model), "Invalid metrics for model"
        assert self._metrics_match_dataset(), "Invalid use of binary metric for multiclass dataset"
        assert os.path.isdir(out_dir), f"{out_dir} is not a directory"

    def _metrics_match_dataset(self) -> bool:
        """Check if binary metrics were requested with multiclass dataset"""
        if self.dataset.task != MULTI_CLF:
            return True
        
        for m in self.metrics:
            if isinstance(m, BinaryMetric): 
                return False
            elif isinstance(m, MetricWrapper):
                if not m.works_with_multiclass:
                    return False
        return True

class ExperimentRunner(BaseRunner):
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
    notes
        (wandb only, optional) Freeform notes about this experiment.
    tags
        (wandb only, optional) Tags which will be shown on this experiment.
    
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
            notes: str = None,
            tags: list[str] = None
        ) -> None:
        super().__init__(
            model=model,
            dataset=dataset,
            metrics=metrics,
            out_dir=out_dir,
            model_adapter=model_adapter,
            enable_tracker=enable_tracker,
            project=project,
            entity=entity,
            notes=notes,
            tags=tags
        )

        if model_adapter:
            self.model_adapter.model = model

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
            mode=["disabled", "online"][self._enable_tracker],
            notes=self.notes,
            tags=[*self._autotags, *self.tags]
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
    
    @property
    def _autotags(self):
        """Tags which will automatically be added to this run, for easier filtering."""
        
        #TODO: Get more detailed tags from adapter?
        
        return [
            f"model:{self.model.__class__.__name__}",
            f"data:{self.dataset.__class__.__name__}"
        ]

class HyperparameterScanRunner(BaseRunner):
    """Helper class for generating & running multiple `ExperimentRunner` instances, to test
    influence of given hyperparameters on the given model's behavior.

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
    notes
        (wandb only, optional) Freeform notes about this experiment.
    tags
        (wandb only, optional) Tags which will be shown on this experiment.

    """
    
    @property
    def __dataset(self):
        return deepcopy(self.dataset)
    
    def __flatten_paramsets(paramsets: dict):
        #TODO: different flattening strategies, e.g. cartesian product?

        return [{hyperparam: val} for hyperparam, vals in paramsets.items() for val in vals]

    def _run_instance(self, paramset: dict):
        model = self.model.clone(new_params=paramset)

        param_tags = [f"param:{p}" for p in paramset.keys()]
        user_tags = self.tags or []

        user_notes = f"\n{self.notes}" if self.notes is not None else ""

        name = base64.urlsafe_b64encode(str(paramset).encode())

        runner = ExperimentRunner(
            model=model,
            dataset=self.__dataset,
            metrics=self.metrics.clone(),
            out_dir=self.out_dir,
            name=name,
            model_adapter=self.model_adapter,
            enable_tracker=self._enable_tracker,
            project=self.project,
            notes=f"Generated via HyperparameterScanRunner with {paramset}.{user_notes}",
            tags=["hparam-scan", *param_tags, *user_tags]
        )
        runner.run()

    def run(self, hyperparameters: dict[str, list] | Iterable[dict[str, list]], parallel_workers: int = 1) -> None:
        """Start the hyperparameter scan.
        
        Parameters
        ----------
        hyperparameters
            If dict mapping hyperparameter names to lists of values: will generate separate series
            of test runs for each hyperparameter, with each instance having that parameter set
            to one of the listed values, leaving all others as defined on the provided `model`.
            If list of dicts, will generate a test run for each dict, overriding
            the provided `model`'s parameters with those specified by each of the dicts.
        parallel_workers
            (default: 1) If equal to 2 or greater, experiment instances will be run using
            a `multiprocessing` `Pool` of this many processes.

        """

        if isinstance(hyperparameters, dict):
            hyperparameters = HyperparameterScanRunner.__flatten_paramsets(hyperparameters)
        
        if parallel_workers >= 2:
            with Pool(parallel_workers) as p:
                p.map(self._run_instance, hyperparameters)

        else:
            for paramset in hyperparameters:
                self._run_instance(paramset)
