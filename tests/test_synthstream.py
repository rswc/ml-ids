from synthstream.synthstream import SyntheticStream
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

        ss = SyntheticStream(size=None, seed=42, init_csamplers=[cs1, cs2])
        
        ss.next_state()
        assert ss.get_state() == (ex1[0], 'A' )
        ss.next_state()
        assert ss.get_state() == (ex1[1], 'A' )
        ss.next_state()
        assert ss.get_state() == (ex2[0], 'B' )
        ss.next_state()
        assert ss.get_state() == (ex2[1], 'B' )
        ss.next_state()
        assert ss.get_state() == (ex2[0], 'B' )
                

