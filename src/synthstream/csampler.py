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
        max_samples (int | None): optional number of samples after which sampler raises EndOfClassSamples
        eoc_strategy (str): one of EOC_STRATEGIES, specify behaviour after iteration over "samples_list" ends
            'raise': raise EndOfClassError
            'loop': reset iterator, start from beginning
            'none': start returning None as class example
    """
    
    EOC_STRATEGIES = ['raise', 'loop', 'none']
    
    def __init__(self, label: str, samples: list, weight_func: callable, stream_t_start: int = 0, max_samples: int | None = None, eoc_strategy: str = 'raise'):

        if eoc_strategy not in ClassSampler.EOC_STRATEGIES:
            raise ValueError(f"Invalid eoc_strategy: {eoc_strategy} - does not match {ClassSampler.EOC_STRATEGIES}")
        
        if len(samples) == 0:
            raise ValueError(f"Invalid samples: {samples} - list cannot be empty")
        
        if max_samples is not None and max_samples < 0:
            raise ValueError("Number of samples cannot be less than 0")

        if stream_t_start < 0:
            raise ValueError("Stream start index cannot be less than 0")

        self.label = label
        self.sample_list = samples
        self.weight_func = weight_func
        self.stream_t_start = stream_t_start
        self.max_samples = max_samples
        self.eoc_strategy = eoc_strategy
        self.sample_iter = iter(self.sample_list)

        self.index = 0
        self.last_output = None
        
    def __iter__(self):
        return self
    
    def __next__(self):
        self._advance_state()
        return self.last_output
    
    @property
    def samples_left(self) -> int | None:
        return self.max_samples - self.index if self.max_samples is not None else None
    
    @property
    def end_of_iteration(self) -> bool:
        return (self.index == self.max_samples)
        
    def weight(self, t: int):
        w = self.weight_func(t)
        if w < 0:
            raise ValueError("Weight function returned value below zero")
        return w

    def _advance_state(self):
        if self.end_of_iteration: 
            raise StopIteration(f"ClassSampler({self.label}) already produced {self.max_samples} samples") 
        
        try:
            self.last_output = next(self.sample_iter)
        except StopIteration:
            if self.eoc_strategy == 'none':
                # Probably should mark some boolean flag to speed up creating 'none'-streams 
                # and do not throw exceptions every time we change state
                self.last_output = None
            elif self.eoc_strategy == 'loop':
                self.sample_iter = iter(self.sample_list)
                # possible of raising another exception when list is empty and 'loop' strategy applied
                # currently checked in __init__, otherwise undefined behaviour 
                self.last_output = next(self.sample_iter)
            elif self.eoc_strategy == 'raise':
                raise EndOfClassError(f"ClassSampler({self.label}) end of stream after advancing {self.index} samples")

        self.index += 1
