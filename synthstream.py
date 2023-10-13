import random 

class StreamNotInitialized(Exception):
    pass

class EmptyClassStream(Exception):
    pass

class SyntheticStream:
    def __init__(self, size: int = None):
        self.prob_samplers: dict[str, callable] = {}
        self.x_generators: dict[str, iter] = {}
        self.active_labels = []
        self.t = 0
        self.n = size
     
    def add_class(self, label: str, prob_sampler: callable, x_generator: iter):
        self.active_labels.append(label)
        self.prob_samplers[label] = prob_sampler
        self.x_generators[label] = x_generator
    
    def __sample_class(self) -> tuple[dict, str]:
        weights = [ self.prob_samplers[label](self.t) for label in self.active_labels]

        # TODO: Create specification for prob_sampler output domain
        assert all(0.0 <= p <= 1.0 for p in weights)

        label = random.choices(self.active_labels, weights, k=1)[0]
        try:
            x = next(self.x_generators[label])
        except StopIteration:
            raise EmptyClassStream(f"[ERROR]: No more examples of class {label}")
        return (x, label)

    def __iter__(self):
        if len(self.active_labels) == 0:
            raise StreamNotInitialized
        while (self.n is None or self.t < self.n):
            yield self.__sample_class()
            self.t += 1