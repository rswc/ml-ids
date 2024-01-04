from random import Random
from .csampler import ClassSampler
from river.datasets.base import MULTI_CLF, Dataset
import re

class NoActiveSamplersError(Exception):
    pass

class ActiveLabelDuplicateError(Exception):
    pass

class SyntheticStream(Dataset):
    """SyntheticStream
    
    This dataset simulates synthetic online stream with weighted sampling based on `ClassSamplers` mechanism.

    Parameters
    ----------
    n_features
        Number of features in original dataset 
    max_samples
        Number of examples after which datastream will stop sampling
    seed
        Seed to initialize the random module for selecting samplers based on their weight
    init_csamplers
        List of `ClassSampler` instances that make this stream

    """

    def __init__(self, n_features: int, max_samples: int = None, seed: int = None, init_csamplers: list[ClassSampler] = []):
        super().__init__(task=MULTI_CLF, n_features=n_features, n_samples=max_samples)
        self._random = Random(seed)
        self.active_samplers = dict()
        self.future_samplers: list(ClassSampler) = []
        self.t = 0 
        self.all_labels_set: set(str) = set()

        self.last_class_probabilities = None
        self.last_class_weights = None
        self.last_sample = None
        self.last_label = None
        
        for csampler in init_csamplers: 
            self._add_sampler(csampler)
            
        # Used by river.datasets.base.Dataset
        # But can be calculated after all csamplers have been added
        self.n_classes = len(self.all_labels_set)
        
    def __csampler_to_dict(self, csampler: ClassSampler):
        cs_dict = {}
        cs_dict["Label"] = csampler.label
        cs_dict["NumSamples"] = len(csampler.sample_list)
        cs_dict["StreamStartIdx"] = csampler.stream_t_start
        cs_dict["MaxSamples"] = csampler.max_samples
        cs_dict["EOCStrategy"] = csampler.eoc_strategy
        return cs_dict
        
    @property
    def _repr_content(self) -> dict:
        """The items that are displayed in the __repr__ method.

        This property can be overridden in order to modify the output of the __repr__ method.

        """

        content = {}
        content["Name"] = self.__class__.__name__
        content["Task"] = self.task
        if self.n_samples is None:
            content["Samples"] = f"inf"
        else:
            content["Samples"] = f"{self.n_samples:,}"
        content["Classes"] = f"{self.n_classes:,}"
        content["ActiveClassSamplers"] = [ 
            self.__csampler_to_dict(csampler) 
            for csampler in self.active_samplers.values() 
        ] 
        content["FutureClassSamplers"] = [
            self.__csampler_to_dict(csampler) 
            for csampler in self.future_samplers
        ]
        
        return content

    def __repr__(self):
        # Derived from River.datasets.base.Dataset, but modified to parse the nested csampler structure
        mod_repr_content = self._repr_content
        mod_repr_content["ActiveClassSamplers"] = f'{len(mod_repr_content["ActiveClassSamplers"]):,}'
        mod_repr_content["FutureClassSamplers"] = f'{len(mod_repr_content["FutureClassSamplers"]):,}'
        l_len = max(map(len, mod_repr_content.keys()))
        r_len = max(map(len, mod_repr_content.values()))

        out = f"{self.desc}\n\n" + "\n".join(
            k.rjust(l_len) + "  " + v.ljust(r_len) for k, v in mod_repr_content.items()
        )

        if "Parameters\n    ----------" in self.__doc__:
            params = re.split(
                r"\w+\n\s{4}\-{3,}",
                re.split("Parameters\n    ----------", self.__doc__)[1],
            )[0].rstrip()
            out += f"\n\nParameters\n----------{params}"

        return out

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
        
    def _add_sampler(self, csampler: ClassSampler):
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
        if self.t == self.n_samples:
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
        
