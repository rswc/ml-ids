from river.datasets.base import Dataset
from synthstream import SyntheticStream
from datetime import datetime
import wandb
import json
import os
from collections import Counter
from collections import deque

class DatasetAnalyzer:
    """Helper class for analyzing datasets.

    Parameters
    ----------
    dataset
        The river-compatible dataset class for this experiment.
    window-size
        Specifies range of class count calculations. Passing `None` implies `window_size = n_samples` 
    out_dir
        The directory to which logs will be saved.
    name
        (optional) A custom name for this experiment.
    enable_tracker
        (default: `True`) Whether or not to use the tracker (currently, wandb) to log data
        from this experiment online.
    project
        (wandb only, optional) The name of the project under which this exepriment should be categorized.
    entity
        (wandb only, optional) The entity (user or team) which owns this experiment.
    
    """

    def __init__(self, dataset: Dataset, window_size: int, out_dir: str, name: str = None, enable_tracker: bool = True, project: str = None, entity: str = None):
        self.dataset = dataset
        self.out_dir = out_dir
        self.entity = entity
        self.project = project

        # Set cumulative mode if window_size is None
        self.window_size = self.dataset.n_samples if window_size is None else window_size

        self._enable_tracker = enable_tracker

        time = datetime.now().strftime("%y-%m-%d_%H%M%S")
        dataset_name = dataset.__class__.__name__
        self._id = f"{time}_{dataset_name}_ws={window_size}" + (f"_{name}" if name else "")
        
    @property
    def _parameters(self):
        """Parameters of the analysis"""
        return {
            "id": self._id,
            "dataset": self.dataset._repr_content,
            "window_size": self.window_size, 
        }
        
    def analyze(self):
        wandb.init(
            entity=self.entity,
            project=self.project,
            config=self._parameters,
            mode=["disabled", "online"][self._enable_tracker]
        )
        
        print("Starting Dataset Analysis:", self._id)
        with open(self._meta_path, "x") as file_meta:
            json.dump(self._parameters, file_meta, default=lambda o: repr(o), indent=4)

        print("Metadata available at:", os.path.abspath(self._meta_path))

        window = deque()
        counts = Counter()
        first_occr = dict()
        last_occr = dict()

        for t, (x, y) in enumerate(self.dataset):

            counts[y] += 1
            window.append(y)
            
            if y not in first_occr.keys():
                first_occr[y] = t
                print(f"First occurence of class {y} = {t}")
            last_occr[y] = t

            if len(window) > self.window_size:
                label = window.popleft()
                counts[label] -= 1
            
            synth_stats = {}
            if isinstance(self.dataset, SyntheticStream):
                for c, w in self.dataset.class_weights.items():
                    synth_stats[f"SynthClassWeight({c})"] = w
                    
                for c, p in self.dataset.class_probabilities.items():
                    synth_stats[f"SynthClassProbability({c})"] = p
                    
            class_counts = {f"ClassCount({c})[{self.window_size}]": count for c, count in counts.items() }
            wandb.log({**class_counts, **synth_stats})
            
        for c, t in last_occr.items():
            print(f"Last occurence of class {c} = {t}")

        print("Analysis DONE")
        wandb.finish()
    
    @property
    def _meta_path(self):
        return os.path.join(self.out_dir, self._id + "_ANALYSIS_META.json")