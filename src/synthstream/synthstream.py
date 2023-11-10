from random import Random
from .csampler import ClassSampler, EndOfClassSamples

class NoActiveSamplersError(Exception):
    pass

class ActiveLabelDuplicateErorr(Exception):
    pass

# TODO:
# 1. Add random source for stream with possible Seed (do not affect global random state) 
# 2. Use ClassSamplers as init list [cs1, cs2, cs3, ...], or via add_class_sampler()
# 3. Track intervals - [start, end) - or simply "active" / "non-active" labels
#      a) do not allow for class labels that are copy of one/another (?)
#      b) allow (?) - how can we allow for "death" of class and "reapearance" (?)
#          probably model weights? 0 if in intervals, 1 otherwise, etc?
# 4. add some sort of "global stream state" -> fetch EGR values for every current class in stream
#   will be useful for plotting values

class SyntheticStream:

    def __init__(self, size: int = None, seed: int = None, init_csamplers: list[ClassSampler] = []):
        self.active_samplers = dict()
        self.future_samplers = list()
        self.t = 0 # next_t ?
        self.n = size
        self._random = Random(seed)
        self.curr_x = None
        self.curr_y = None
        
        for csampler in init_csamplers: 
            self.add_csampler(csampler)
            
    # how to deal with labels
    def _activate_csampler(self, csampler):
        if csampler.label in self.active_samplers.keys():
            raise ActiveLabelDuplicateErorr
        
        # arm for the first use
        print(f"[.]: Activating class sampler {csampler.label} at time {self.t}.")
        csampler.next_state()
        self.active_samplers[csampler.label] = csampler

    def _remove_active_csampler(self, csampler):
        assert csampler.label in self.active_samplers.keys()
        del self.active_samplers[csampler.label]
        print(f"[.]: Removing active class sampler {csampler.label} at time {self.t}.")
        
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
            
        assert csampler.stream_t_start > self.t
        self.future_samplers.append(csampler)
     
    def add_csampler(self, csampler: ClassSampler):
        # maybe merge <= ? what if we try to add to the past?
        if csampler.stream_t_start < self.t:
            raise ValueError("Trying to add class sampler in the past")
        elif csampler.stream_t_start == self.t:
            self._activate_csampler(csampler)
        elif csampler.stream_t_start > self.t:
            self._add_future_csampler(csampler)
    
    def next_state(self):
        if len(self.active_samplers.items()) == 0:
            raise NoActiveSamplersError
        
        weights = []
        data = []

        for csampler in self.active_samplers.values():
            w, x = csampler.get_state()
            weights.append(w)
            data.append((x, csampler))
            
        win_x, win_csampler = self._random.choices(data, weights, k=1)[0]

        # TODO: maybe think of better way to store the results?
        self.curr_x = win_x
        self.curr_y = win_csampler.label

        # arm for the next state, or remove from current one if ended
        # this may raise `EndOfClassError` and it is OK. 
        # It means it is an Error that user was prepared for while declaring ClassSampler strategy
        try:
            win_csampler.next_state()
        except EndOfClassSamples as e:
            # delete from the stream
            self._remove_active_csampler(win_csampler)
            
        # state transition
        self.t += 1

        # activate future classes 
        for csampler in self.future_samplers:
            if csampler.stream_t_start == self.t:
                self._activate_csampler(csampler)
                
    def get_state(self):
        return self.curr_x, self.curr_y
