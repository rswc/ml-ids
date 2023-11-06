import os
from datetime import datetime
import csv
import json
from river.base import Classifier
from river.datasets.base import Dataset
from river.metrics.base import Metrics

class ExperimentRunner:

    def __init__(self, model: Classifier, dataset: Dataset, metrics: Metrics, out_dir: str, name: str = None) -> None:
        self.model = model
        self.dataset = dataset
        self.metrics = metrics
        self.out_dir = out_dir

        assert metrics.works_with(model), "Invalid metrics for model"
        assert os.path.isdir(out_dir), f"{out_dir} is not a directory"

        time = datetime.now().strftime("%y-%m-%d_%H%M%S")
        dataset_name = dataset.__class__.__name__
        self._id = f"{time}_{str(model)}_{dataset_name}" + (f"_{name}" if name else "")
    
    @property
    def _parameters(self):
        return {
            "id": self._id,
            "model": self.model._get_params(),
            "dataset": self.dataset._repr_content,
            "metrics": self._metrics_names
        }
    
    def run(self):
        print("Starting experiment:", self._id)

        with open(self._meta_path, "x") as file_meta:
            json.dump(self._parameters, file_meta, default=lambda o: repr(o))

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
        
        print("Experiment DONE")
    
    @property
    def _metrics_names(self):
        return [metric.__class__.__name__ for metric in self.metrics]
    
    @property
    def _metrics_path(self):
        return os.path.join(self.out_dir, self._id + "_METRICS.csv")
    
    @property
    def _meta_path(self):
        return os.path.join(self.out_dir, self._id + "_META.json")