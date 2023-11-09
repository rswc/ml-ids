from synthstream import ClassSampler, EndOfClassSamples, EndOfClassError
import pytest

class TestClassSampler:
    
    def test_invalid_init_params_throws_exception(self):
        with pytest.raises(ValueError) as e:
            csampler = ClassSampler('test', [], lambda t: 1, 0, 500, 'loop')

        with pytest.raises(ValueError) as e:
            csampler = ClassSampler('test', [(1, 1)], lambda t: 1, 0, -1)
    
        with pytest.raises(ValueError) as e:
            csampler = ClassSampler('test', [(1, 1)], lambda t: 1, -1)

        with pytest.raises(ValueError) as e:
            csampler = ClassSampler('test', [(1, 1)], lambda t: 1, 0, eoc_strategy='!@#$')
            
    def test_negative_weight_raises_exception(self):
        csampler = ClassSampler(
            label='None', samples=[], weight_func=lambda t: -1,
            stream_t_start=0, n_samples=None, eoc_strategy='none'
        )

        with pytest.raises(ValueError) as e:
            _, x = csampler.next_and_get_state()

        csampler = ClassSampler(
            label='None', samples=[], weight_func=lambda t: 0,
            stream_t_start=0, n_samples=None, eoc_strategy='none'
        )
        
        assert csampler.next_and_get_state() == (0, None)
        
            
    def test_none_stream(self):
        csampler = ClassSampler(
            label='None', samples=[], weight_func=lambda _: 1,
            stream_t_start=0, n_samples=None, eoc_strategy='none'
        )
        
        for _ in range(1000):
            assert csampler.next_and_get_state()[1] is None

    def test_multiple_end_of_stream_samples_exception(self):
        example = (1, 1, 1)
        n_iters = 0
        csampler = ClassSampler(
            label='class', samples=[example, example], weight_func=lambda t: 1, 
            stream_t_start=0, n_samples=n_iters, eoc_strategy='loop'
        )

        with pytest.raises(EndOfClassSamples) as e:
            csampler.next_state()

        with pytest.raises(EndOfClassSamples) as e:
            csampler.next_state()

        

    def test_loop_eoc_strategy(self):
        example = (1, 1, 1)
        n_iters = 5
        csampler = ClassSampler(
            label='class', samples=[example], weight_func=lambda t: 1, 
            stream_t_start=0, n_samples=n_iters, eoc_strategy='loop'
        )
        
        for _ in range(n_iters):
            csampler.next_state()
            _, x = csampler.get_state()
            assert x == example
        
        with pytest.raises(EndOfClassSamples) as e:
            csampler.next_state()
        
    def test_none_eoc_strategy(self):
        example = (1, 1, 1)
        n_iters = 3
        csampler = ClassSampler(
            label='class', samples=[example], weight_func=lambda t: 1, 
            stream_t_start=0, n_samples=n_iters, eoc_strategy='none'
        )

        _, x = csampler.next_and_get_state()
        assert x == example
        
        _, x = csampler.next_and_get_state()
        assert x is None
            
        _, x = csampler.next_and_get_state()
        assert x is None

        with pytest.raises(EndOfClassSamples) as e:
            csampler.next_state()

    def test_raise_eoc_strategy(self):
        example = (1, 1, 1)
        n_iters = 3
        csampler = ClassSampler(
            label='class', samples=[example], weight_func=lambda t: 1, 
            stream_t_start=0, n_samples=n_iters, eoc_strategy='raise'
        )

        _, x = csampler.next_and_get_state()
        assert x == example
        
        with pytest.raises(EndOfClassError) as e:
            _, x = csampler.next_state()

        with pytest.raises(EndOfClassError) as e:
            _, x = csampler.next_state()


        



