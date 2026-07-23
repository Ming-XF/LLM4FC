"""Early stopping utility — usable by any Trainer subclass.

Usage::

    from utils.early_stopping import EarlyStopping

    early_stopper = EarlyStopping(patience=20, min_delta=0.001, mode='max')
    for epoch in range(num_epochs):
        ...
        val_result = evaluate(dataloader_key='val')
        score = val_result['Accuracy']
        if early_stopper.step(score):
            save_model()                          # is_best  == True
        if early_stopper.early_stop:
            break
"""


class EarlyStopping:
    """Monitor a metric and signal when training should stop.

    Args:
        patience:  number of epochs without improvement before stopping.
        min_delta: minimum absolute change to count as an improvement.
        mode:      ``'max'`` (larger is better, e.g. Accuracy / AUC) or
                   ``'min'`` (smaller is better, e.g. Loss).
    """

    def __init__(self, patience=20, min_delta=0.001, mode='max'):
        if patience <= 0:
            patience = float('inf')               # effectively disabled
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.is_best = False

    def step(self, score: float) -> bool:
        """Update internal state with a new epoch's metric.

        Returns:
            True if this epoch produced a new **best** score (caller should
            checkpoint the model).  Callers should read ``self.early_stop``
            after this method returns to decide whether to break the loop.
        """
        self.is_best = False

        if self.best_score is None:
            self.best_score = score
            self.is_best = True
            return True

        improved = ((self.mode == 'max' and score > self.best_score + self.min_delta) or
                    (self.mode == 'min' and score < self.best_score - self.min_delta))

        if improved:
            self.best_score = score
            self.counter = 0
            self.is_best = True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.is_best

    def state_dict(self) -> dict:
        return {'best_score': self.best_score,
                'counter': self.counter,
                'early_stop': self.early_stop}

    def load_state_dict(self, d: dict) -> None:
        self.best_score = d['best_score']
        self.counter = d['counter']
        self.early_stop = d['early_stop']
