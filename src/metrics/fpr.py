from river.metrics.base import BinaryMetric

class FalsePositiveRate(BinaryMetric):
    """The proportion of bening instances that have triggered a false alarm: FP/(FP + TN)."

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
        tn = self.cm.true_negatives(self.pos_val)
        try:
            return fp / (fp + tn)
        except ZeroDivisionError:
            return 0.0