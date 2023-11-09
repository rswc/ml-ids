class EndOfClassSamples(Exception):
    pass

class EndOfClassError(Exception):
    pass

class ClassSampler:
    """
    ClassSampler allows for easy in-stream class-examples weight-based generation as a wrapper around simple examples list.
    
    Args:
        label (str): Class label 
        samples (list): Class examples which create a baseline for generation
        weight_func (callable): Function W(t), returns class "weight" (importance) in the stream at local (class) time t 
        stream_t_start (int): index in stream when class examples emerge
        n_samples (int | None): optional number of samples after which sampler raises EndOfClassSamples
        eoc_strategy (str): one of EOC_STRATEGIES, specify behaviour after iteration over "samples_list" ends
            'raise': raise EndOfClassError
            'loop': reset iterator, start from beginning
            'none': start returning None as class example
    """
    
    EOC_STRATEGIES = ['raise', 'loop', 'none']
    
    def __init__(self, label: str, samples: list, weight_func: callable, stream_t_start: int, n_samples: int | None = None, eoc_strategy: str = 'raise'):

        if eoc_strategy not in ClassSampler.EOC_STRATEGIES:
            raise ValueError(f"Invalid eoc_strategy: {eoc_strategy} - does not match {ClassSampler.EOC_STRATEGIES}")
        
        if eoc_strategy == 'loop' and len(samples) == 0:
            raise ValueError(f"Invalid samples: {samples} - list cannot be empty when eoc_strategy is set to 'loop'")
        
        if n_samples is not None and n_samples < 0:
            raise ValueError("Number of samples cannot be less than 0")

        if stream_t_start < 0:
            raise ValueError("Stream start index cannot be less than 0")

        self.label = label
        self.sample_list = samples
        self.weight_func = weight_func
        self.stream_t_start = stream_t_start
        self.n_samples = n_samples
        self.eoc_strategy = eoc_strategy
        self.sample_iter = iter(self.sample_list)
        self.next_t = 0
        self.curr_sample = None
        self.curr_weight = None
        
        
    def next_state(self):
        if self.n_samples is not None and self.next_t >= self.n_samples:
            raise EndOfClassSamples(f"ClassSampler({self.label}) already produced {self.n_samples} samples") 
        
        try:
            self.curr_sample = next(self.sample_iter)
        except StopIteration:
            if self.eoc_strategy == 'none':
                # Probably should mark some boolean flag to speed up creating 'none'-streams 
                # and do not throw exceptions every time we change state
                self.curr_sample = None
            elif self.eoc_strategy == 'loop':
                self.sample_iter = iter(self.sample_list)
                # possible of raising another exception when list is empty and 'loop' strategy applied
                # currently checked in __init__, otherwise undefined behaviour 
                self.curr_sample = next(self.sample_iter)
            elif self.eoc_strategy == 'raise':
                raise EndOfClassError(f"ClassSampler({self.label}) end of stream after advancing {self.next_t} samples")

        self.curr_weight = self.weight_func(self.next_t)
        # larger than 0 for some epsilon error, but maybe 0.0 is enough? 
        if self.curr_weight < -1e-9:
            raise ValueError("Weight function returned value below zero")

        self.next_t += 1
        
    def get_state(self):
        return self.curr_weight, self.curr_sample
    
    def next_and_get_state(self):
        self.next_state()
        return self.get_state()