class LinearModel:
    """A simple linear model."""

    class History:
        def __init__(self, history):
            self.history = history

    def __init__(self):
        self._state = 0

        self.metrics_names = ['acc', 'loss', 'val_acc', 'val_loss']

    def __copy__(self):
        x = LinearModel()
        x._state = self._state

        return x

    def fit(self, x=None, y=None, epochs=None, initial_epoch=None,
            validation_data=None, batch_size=None):
        history = {k: [] for k in self.metrics_names}
        history['epochs'] = []
        
        for e in range(initial_epoch, epochs):
            self._state += 1

            history['epochs'].append(e)
            history['acc'].append(self.acc_fn(self._state))
            history['loss'].append(self.loss_fn(self._state))
            history['val_acc'].append(self.val_acc_fn(self._state))
            history['val_loss'].append(self.val_loss_fn(self._state))

        return self.History(history)

    def evaluate(self, x=None, y=None):
        return self.loss_fn(self._state), self.acc_fn(self._state)

    def acc_fn(self, x):
        return x

    def loss_fn(self, x):
        return -x

    def val_acc_fn(self, x):
        return self.acc_fn(x) + 0.5

    def val_loss_fn(self, x):
        return self.loss_fn(x) - 0.5

