from random import Random
from .csampler import ClassSampler, EndOfClassSamples

class NoActiveSamplersError(Exception):
    pass

class ActiveLabelDuplicateErorr(Exception):
    pass

class SyntheticStream:
    def __init__(self, max_samples: int = None, seed: int = None, init_csamplers: list[ClassSampler] = []):
        self._random = Random(seed)
        self.max_samples = max_samples
        self.active_samplers = dict()
        self.future_samplers: list(csampler) = []
        self.next_t = 0 
        self.all_labels_set: set(str) = set()

        self.last_class_probabilities = None
        self.last_sample = None
        self.last_label = None
        
        for csampler in init_csamplers: 
            self.add_csampler(csampler)

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
        
    def add_csampler(self, csampler: ClassSampler):
        if csampler.stream_t_start < self.next_t:
            raise ValueError("Trying to add class sampler in the past")
        elif csampler.stream_t_start == self.next_t:
            self._activate_csampler(csampler)
        elif csampler.stream_t_start > self.next_t:
            self._add_future_csampler(csampler)
            
        self.all_labels_set.add(csampler.label)
        
    """
    def _calc_curr_class_probs(self) -> dict:
        label_weight_pairs = []
        weight_sum = 0.0
        for label in self.all_labels_set:
            if label in self.active_samplers.keys():
                csampler = self.active_samplers[label]
                weight_sum += csampler.curr_weight 
                label_weight_pairs.append((label, csampler.curr_weight))
        normalized_pairs = [ (label, w/weight_sum) for label, w in label_weight_pairs ]
        return dict(normalized_pairs)
    """

    def _activate_csampler(self, csampler):
        """Add `csampler` to dictionary of active samplers - currently used in a stream"""
        if csampler.label in self.active_samplers.keys():
            raise ActiveLabelDuplicateErorr(f"Class {csampler.label} is already present as active")
        
        print(f"[.]: Activating class sampler {csampler.label} at time {self.next_t}.")
        csampler.next_state()
        self.active_samplers[csampler.label] = csampler

    def _remove_active_csampler(self, csampler):
        assert csampler.label in self.active_samplers.keys()
        del self.active_samplers[csampler.label]
        print(f"[.]: Removing active class sampler {csampler.label} at time {self.next_t}.")
        
    def _add_future_csampler(self, csampler):
        """Add `csampler` with `stream_t_start` > current stream `t`, to use-in-the-future samplers list """

        # Warn the user about consequences: 
        # if there exists duplicate of a class in "future" there is a possibility of collision (exception)
        # rule violation: there can be only one example of active class
        if csampler.label in self.active_samplers.keys():
            active_cs = self.active_samplers[csampler.label]
            samples_left = active_cs.n_samples or 'inf' # if None
            print(
                f"[!]: WARNNING: Currently there exists active sampler "
                f"since {active_cs.stream_t_start} timestamp, with "
                f"{samples_left} samples left in local context.\n"
                "Make sure there is no chance of collision of two active samplers "
                "from one class, in such case `SyntheticStream` will raise `ActiveClassDuplicateError.`"
            )
            
        assert csampler.stream_t_start > self.next_t
        self.future_samplers.append(csampler)
    
    def _advance_state(self):
        # raise StopIteration to correctly handle `for x in stream` syntax
        if self.next_t == self.max_samples:
            raise StopIteration
        
        # raise Exception if there is no samplers to sample from
        if len(self.active_samplers.items()) == 0:
            raise NoActiveSamplersError(f"[!]: No more samplers in the stream at time {self.next_t}")
        
        weights = [ csampler.get_state()[0] for csampler in self.active_samplers.values() ]
        active_csamplers = [ csampler for csampler in  self.active_samplers.values() ]
        
        weight_sum = sum(weights)

        # Store results as probabilities inside dictionary: label -> prob
        self.last_class_probabilities = dict(
            [(csampler.label, w / weight_sum) for w, csampler in zip(weights, active_csamplers)]
        )

        # fill not used classes with probability 0.0
        for label in self.all_labels_set:
            if label not in self.last_class_probabilities:
                self.last_class_probabilities[label] = 0.0

        selected_csampler = self._random.choices(active_csamplers, weights, k=1)[0]

        self.last_sample = selected_csampler.curr_sample
        self.last_label = selected_csampler.label

        # This may raise `EndOfClassError` and it is OK. 
        # It means it is an Error that user was prepared for while declaring ClassSampler strategy
        try:
            selected_csampler.next_state()
        except EndOfClassSamples as e:
            # This means we specified `n_samples` param inside ClassSampler
            # deactivate because we already sampled `n_samples` from it.
            self._remove_active_csampler(selected_csampler)
            
        # If future sampler is meant to start next round, activate it
        for csampler in self.future_samplers:
            if csampler.stream_t_start == self.next_t + 1:
                self._activate_csampler(csampler)

        self.next_t += 1
        
