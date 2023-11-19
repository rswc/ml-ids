from river.metrics.base import BinaryMetric

class AttackDetectionRate(BinaryMetric):
    """Attack Detection Rate: TP/(TP + FN).

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion matrix between multiple metrics. Sharing
        a confusion matrix reduces the amount of storage and computation time.
    pos_val
        Value to treat as "positive".
    """
    
    def get(self):
        tp = self.cm.true_positives(self.pos_val)
        fn = self.cm.false_negatives(self.pos_val)
        try:
            return tp / (tp + fn)
        except ZeroDivisionError:
            return 0.0