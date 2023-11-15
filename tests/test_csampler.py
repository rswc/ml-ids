from synthstream import ClassSampler, EndOfClassError
import pytest

class TestClassSampler:
    
    def test_invalid_init_params_throws_exception(self):
        with pytest.raises(ValueError) as e:
            csampler = ClassSampler('test', [], lambda t: 1, 0, 500, 'loop')

        with pytest.raises(ValueError) as e:
            csampler = ClassSampler('test', [], lambda t: 1, 0, 500, 'none')

        with pytest.raises(ValueError) as e:
            csampler = ClassSampler('test', [(1, 1)], lambda t: 1, 0, -1)
    
        with pytest.raises(ValueError) as e:
            csampler = ClassSampler('test', [(1, 1)], lambda t: 1, -1)

        with pytest.raises(ValueError) as e:
            csampler = ClassSampler('test', [(1, 1)], lambda t: 1, 0, eoc_strategy='!@#$')
            
    def test_negative_weight_raises_exception(self):
        csampler = ClassSampler(
            label='None', samples=[None], weight_func=lambda t: -1,
            stream_t_start=0, max_samples=None, eoc_strategy='loop'
        )

        with pytest.raises(ValueError) as e:
            csampler.weight(0)
            
    def test_none_strategy_start_to_produce_none(self):
        csampler = ClassSampler(
            label='Test', samples=[1, 2, 3], weight_func=lambda _: 1,
            stream_t_start=0, max_samples=None, eoc_strategy='none'
        )
        
        assert next(csampler) == 1
        assert next(csampler) == 2
        assert next(csampler) == 3
        
        for _ in range(1000):
            assert next(csampler) is None

    def test_multiple_end_of_stream_samples_exception(self):
        example = (1, 1, 1)
        n_iters = 0
        csampler = ClassSampler(
            label='class', samples=[example, example], weight_func=lambda t: 1, 
            stream_t_start=0, max_samples=n_iters, eoc_strategy='loop'
        )

        with pytest.raises(StopIteration) as e:
            next(csampler)

        with pytest.raises(StopIteration) as e:
            next(csampler)

    def test_loop_eoc_strategy_ends_after_max_samples(self):
        example = (1, 1, 1)
        n_iters = 5
        csampler = ClassSampler(
            label='class', samples=[example], weight_func=lambda t: 1, 
            stream_t_start=0, max_samples=n_iters, eoc_strategy='loop'
        )
        
        for _ in range(n_iters):
            assert next(csampler) == example
        
        with pytest.raises(StopIteration) as e:
            next(csampler)
        
    def test_none_eoc_strategy_correctly_appends_none(self):
        example = (1, 1, 1)
        n_iters = 3
        csampler = ClassSampler(
            label='class', samples=[example], weight_func=lambda t: 1, 
            stream_t_start=0, max_samples=n_iters, eoc_strategy='none'
        )

        assert next(csampler) == example
        assert next(csampler) is None
        assert next(csampler) is None

        with pytest.raises(StopIteration) as e:
            next(csampler)

    def test_raise_eoc_strategy_after_class_ended(self):
        example = (1, 1, 1)
        n_iters = 3
        csampler = ClassSampler(
            label='class', samples=[example], weight_func=lambda t: 1, 
            stream_t_start=0, max_samples=n_iters, eoc_strategy='raise'
        )

        assert next(csampler) == example

        with pytest.raises(EndOfClassError) as e:
            next(csampler)

        with pytest.raises(EndOfClassError) as e:
            next(csampler)

        with pytest.raises(EndOfClassError) as e:
            next(csampler)

        with pytest.raises(EndOfClassError) as e:
            next(csampler)


    def test_iter_and_next_interface(self):
        csampler = ClassSampler(
            label='test', samples=[1, 2, 3], weight_func=lambda t: 1, 
            max_samples=3, eoc_strategy='raise'
        )
        
        assert next(csampler) == 1
        assert sum([x for x in csampler]) == 2 + 3
        
        with pytest.raises(StopIteration):
            next(csampler)


