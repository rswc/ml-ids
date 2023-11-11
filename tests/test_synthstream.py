from synthstream.synthstream import SyntheticStream, NoActiveSamplersError
from synthstream.csampler import ClassSampler
import pytest

class TestSyntheticStream:
    
    def test_loop_sampler_with_late_start(self): 
        ex1 = [ (1,1), (1,2) ]
        ex2 = [ (2,1), (2,2) ]
        
        cs1 = ClassSampler(
            label='A', samples=ex1, weight_func=lambda t: 1, 
            stream_t_start=0, n_samples=2, eoc_strategy='raise'
        )
        
        cs2 = ClassSampler(
            label='B', samples=ex2, weight_func=lambda t: 2, 
            stream_t_start=2, n_samples=500, eoc_strategy='loop'
        )

        ss = SyntheticStream(max_samples=None, seed=42, init_csamplers=[cs1, cs2])

        assert next(ss) == (ex1[0], 'A') 
        assert next(ss) == (ex1[1], 'A')
        assert next(ss) == (ex2[0], 'B')
        assert next(ss) == (ex2[1], 'B')
        assert next(ss) == (ex2[0], 'B')

    def test_stream_end_of_iteration(self): 
        none_sampler = ClassSampler(
            label='None', samples=[], weight_func=lambda t: 5, 
            stream_t_start=0, n_samples=10, eoc_strategy='none'
        )
        
        ss = SyntheticStream(max_samples=2, seed=42, init_csamplers=[none_sampler])

        assert next(ss)[0] is None
        assert next(ss)[0] is None
        
        with pytest.raises(StopIteration) as e:
            next(ss)
            
    def test_iterator_interface_correctness(self):
        none_sampler = ClassSampler(
            label='None', samples=[], weight_func=lambda t: 1, 
            n_samples=20, eoc_strategy='none'
        )
        
        ss = SyntheticStream(max_samples=10, init_csamplers=[none_sampler])
        
        assert next(ss) == (None, 'None')
        # 9 samples left in the stream
        assert sum([1 for _ in ss]) == 9

        # trying to read from ended stream
        with pytest.raises(StopIteration):
            next(ss)
            
    def test_synthstream_end_of_classsamplers(self):
        none_sampler = ClassSampler(
            label='None', samples=[], weight_func=lambda t: 1,
            n_samples=10, eoc_strategy='none'
        )
        
        ss = SyntheticStream(max_samples=20)
        ss.add_csampler(none_sampler)
        
        for i in range(10):
            next(ss) == (None, 'None')
            
        with pytest.raises(NoActiveSamplersError) as e:
            next(ss)
        
    def test_class_sampler_with_raise_eoc_raises_exception(self):
        raise_sampler = ClassSampler(
            label = 'TEST', samples=[1] * 20, weight_func=lambda t: 1,
        )
    
    def test_valid_class_probabilities_in_stream(self):
        FLOAT_EPS = 1e-9
        sampler_a = ClassSampler(
            label = 'A', samples=[1] * 20, weight_func=lambda t: 10, n_samples=2
        )
        sampler_b = ClassSampler(
            label= 'B', samples=[2] * 20, weight_func=lambda t: 20,
            stream_t_start=1, n_samples=1
        )
        
        ss = SyntheticStream(seed=42, init_csamplers=[sampler_a, sampler_b])
        assert ss.class_probabilities is None
        assert next(ss) == (1, 'A')
        assert abs(ss.class_probabilities['A'] - 1.0) < FLOAT_EPS 
        assert ss.class_probabilities['B'] < FLOAT_EPS

        
