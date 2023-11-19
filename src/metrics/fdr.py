from river.metrics.base import BinaryMetric

class FalseDiscoveryRate(BinaryMetric):
    """The proportion of the alerts that are irrelevant: FP/(FP + TP).

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion matrix between multiple metrics. Sharing
        a confusion matrix reduces the amount of storage and computation time.
    pos_val
        Value to treat as "positive".
    """
    
    def get(self):
        fp = self.cm.false_positives(self.pos_val)
        tp = self.cm.true_positives(self.pos_val)
        try:
            return fp / (fp + tp)
        except ZeroDivisionError:
            return 0.0