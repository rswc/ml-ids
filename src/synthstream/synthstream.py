from random import Random
from .csampler import ClassSampler
from river.datasets.base import MULTI_CLF

class NoActiveSamplersError(Exception):
    pass

class ActiveLabelDuplicateError(Exception):
    pass

class SyntheticStream:
    def __init__(self, max_samples: int = None, seed: int = None, init_csamplers: list[ClassSampler] = []):
        self._random = Random(seed)
        self.max_samples = max_samples
        self.active_samplers = dict()
        self.future_samplers: list(csampler) = []
        self.t = 0 
        self.all_labels_set: set(str) = set()

        self.last_class_probabilities = None
        self.last_class_weights = None
        self.last_sample = None
        self.last_label = None
        
        for csampler in init_csamplers: 
            self.add_sampler(csampler)
            
    @property
    def task(self):
        return MULTI_CLF

    @property
    def _repr_content(self):
        """The items that are displayed in the __repr__ method.

        This property can be overridden in order to modify the output of the __repr__ method.

        """

        content = {}
        content["Name"] = self.__class__.__name__
        content["Task"] = self.task
        content["Samples"] = "∞"
        return content


    def __iter__(self):
        return self
    
    def __next__(self) -> tuple:
        self._advance_state()
        return self.output

    @property
    def output(self) -> tuple:
        return (self.last_sample, self.last_label)
    
    @property
    def class_probabilities(self) -> dict:
        return self.last_class_probabilities
    
    @property
    def class_weights(self) -> dict:
        return self.last_class_weights
        
    def add_sampler(self, csampler: ClassSampler):
        if csampler.stream_t_start < self.t:
            raise ValueError("Trying to add class sampler in the past")
        elif csampler.stream_t_start == self.t:
            self._activate_sampler(csampler)
        elif csampler.stream_t_start > self.t:
            self._add_future_sampler(csampler)
            
        self.all_labels_set.add(csampler.label)
        
    def _activate_sampler(self, csampler: ClassSampler):
        """Add `csampler` to dictionary of active samplers - currently used in a stream"""
        if csampler.label in self.active_samplers.keys():
            raise ActiveLabelDuplicateError(f"An active sampler of class {csampler.label} is already present")
        
        print(f"[.]: Activating class sampler {csampler.label} at time {self.t}.")
        self.active_samplers[csampler.label] = csampler

    def _remove_active_sampler(self, csampler: ClassSampler):
        assert csampler.label in self.active_samplers.keys()
        del self.active_samplers[csampler.label]
        print(f"[.]: Removing active class sampler {csampler.label} at time {self.t}.")
        
    def _add_future_sampler(self, csampler: ClassSampler):
        """Add `csampler` with `stream_t_start` > current stream `t`, to use-in-the-future samplers list """

        # Warn the user about consequences: 
        # if there exists duplicate of a class in "future" there is a possibility of collision (exception)
        # rule violation: there can be only one example of active class
        if csampler.label in self.active_samplers.keys():
            active_cs = self.active_samplers[csampler.label]
            samples_left = active_cs.samples_left or 'inf' 
            print(
                f"[!]: WARNING:A sampler of class {csampler.label} is currently active "
                f"(since t={active_cs.stream_t_start}), with {samples_left} samples left. "
                "Make sure there is no chance of collision between two active samplers of the same class. "
                "In such cases `SyntheticStream` will raise `ActiveClassDuplicateError`"
            )
            
        assert csampler.stream_t_start > self.t
        self.future_samplers.append(csampler)
    
    def _advance_state(self):
        # Raise StopIteration to correctly handle `for x in stream` syntax
        if self.t == self.max_samples:
            raise StopIteration
        
        # Raise Exception if there is no samplers to sample from
        if len(self.active_samplers.items()) == 0:
            raise NoActiveSamplersError(f"[!]: No more samplers in the stream at time {self.t}")
        
        weights = [ csampler.weight(self.t) for csampler in self.active_samplers.values() ]
        active_csamplers = list(self.active_samplers.values())
        
        weight_sum = sum(weights)
        
        if weight_sum < 1e-9:
            raise ValueError("[!]: Sum of weights in active ClassSamplers is equal to 0.0")

        # TODO: there is probably better way to allow both class_weights and class_probabilities 
        # without code repetition

        # Create dictionary with all classes, set probability to 0.0
        self.last_class_probabilities = dict([(label, 0.0) for label in self.all_labels_set])
        self.last_class_weights = dict([(label, 0.0) for label in self.all_labels_set])

        # Store normalized results for every label
        for w, csampler in zip(weights, active_csamplers):
            self.last_class_weights[csampler.label] = w
            self.last_class_probabilities[csampler.label] = w / weight_sum
            
        selected_csampler = self._random.choices(active_csamplers, weights, k=1)[0]

        # This may raise `EndOfClassError` and it is OK. 
        # It means it is an Error that user was prepared for while declaring ClassSampler strategy
        self.last_sample = next(selected_csampler)
        self.last_label = selected_csampler.label

        # This means we specified `n_samples` param inside ClassSampler
        # deactivate because we already sampled `n_samples` from it.
        # Calling next() on csampler would raise StopIteration
        if selected_csampler.end_of_iteration:
            self._remove_active_sampler(selected_csampler)
            
        # If future sampler is meant to start next round, activate it
        for csampler in self.future_samplers:
            if csampler.stream_t_start == self.t + 1:
                self._activate_sampler(csampler)

        self.t += 1
        
