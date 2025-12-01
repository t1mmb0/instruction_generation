class Regularizer():

    def __init__(self, patience,):

        self.best_loss = float("inf")
        self.patience = patience
        self.counter = 0


    def check_early_stopping(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False