import pytest
from metrics import MetricWrapper
from river.metrics import F1, MacroF1
from river.utils import Rolling

class TestMetricWrapper:
    
    def test_incorrect_input_raises_value_error(self):
        with pytest.raises(ValueError):
            MetricWrapper(metric=7, collapse_label=True, collapse_classes=['A', 'B']) 

        with pytest.raises(ValueError):
            MetricWrapper(F1(), name="::::")

        with pytest.raises(ValueError):
            MetricWrapper(F1(), window_size='hello') 

        with pytest.raises(ValueError):
            MetricWrapper(F1(), collapse_label='there', collapse_classes=['A']) 

        with pytest.raises(ValueError):
            MetricWrapper(F1(), collapse_classes=['A', 'B']) 

        with pytest.raises(ValueError):
            MetricWrapper(MacroF1(), collapse_label=True, collapse_classes=['A', 'B']) 
        
    def test_same_results_with_plain_metric(self):
        
        y_true = [0, 1, 2, 2, 2]
        y_pred = [0, 0, 2, 2, 1]

        mw = MetricWrapper(MacroF1())
        m = MacroF1()

        for yt, yp in zip(y_true, y_pred):
            m.update(yt, yp)
            mw.update(yt, yp)
            
            assert m.get() == mw.get()
            
    def test_same_results_with_rolling_window(self):
        y_true = [0, 1, 2, 2, 2] * 10
        y_pred = [0, 0, 2, 2, 1] * 10

        mw = MetricWrapper(MacroF1(), window_size=3)
        m = Rolling(MacroF1(), window_size=3)

        for yt, yp in zip(y_true, y_pred):
            m.update(yt, yp)
            mw.update(yt, yp)
            
            assert m.get() == mw.get()

    def test_same_results_with_collapse(self):
        y_true = [0, 1, 2, 2, 2, 0, 1, 2, 0, 1, 1, 2]
        y_pred = [0, 0, 2, 2, 1, 0, 2, 0, 0, 1, 1, 1]
        # Map 0->True, 1,2 -> False
        cy_true = [True, False, False, False, False, True, False, False, True, False, False, False]
        cy_pred = [True, True, False, False, False, True, False, True, True, False, False, False]

        mw = MetricWrapper(F1(), collapse_label=True, collapse_classes=[0])
        m = F1()

        for (yt, yp), (cyt, cyp) in zip(zip(y_true, y_pred), zip(cy_true, cy_pred)):
            mw.update(yt, yp)
            m.update(cyt, cyp)
            
            assert m.get() == mw.get()