from src.utils.general_utils import set_seed



class ExperimentRunner:
    def __init__(self, trainer, seeds: list[int], train_loader, val_loader, max_epochs, **trainer_kwargs):
        self.trainer = trainer
        self.seeds = seeds
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.trainer_kwargs = trainer_kwargs
        self.history = {}

    def run(self):
        for seed in self.seeds:
            set_seed(seed)
            trainer = self.trainer(**self.trainer_kwargs)
            trainer.fit(self.train_loader, self.val_loader, self.max_epochs)
            history, roc_auc, ave_prec = trainer._return_history()
            self.history[seed] = {"history": history, "metrics": {
                    "roc_auc": roc_auc,
                    "average_precision": ave_prec,
                },
            }
        return history