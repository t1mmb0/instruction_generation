from src.runners.run_config import RunConfig
import pandas as pd

class SingleRunExecutor:

    def __init__(self,trainer_cls, factory, device, max_epochs, criterion, train_loader, val_loader, test_loader):
        self.trainer_cls = trainer_cls
        self.factory = factory
        self.device = device
        self.max_epochs = max_epochs
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def run(self, seed:int, config: RunConfig):
        model, optimizer, regularizer, lr_scheduler_builder_instance = self.factory.build()
        trainer = self.trainer_cls(model = model, 
                                   optimizer = optimizer,
                                   Regularizer = regularizer,
                                   LR_Scheduler_Builder = lr_scheduler_builder_instance,
                                   device = self.device,
                                   criterion = self.criterion)
        trainer.fit(self.train_loader, self.val_loader, self.max_epochs)
        trainer.evaluate_model(self.test_loader)
        history, roc_auc, ave_prec = trainer._return_history()
         # 1) run-level summary (serialisierbar)

        summary = {
            "seed": seed,
            "best_epoch": trainer.best_epoch,
            "best_val_loss": float(trainer.best_val_loss),
            "roc_auc": None if roc_auc is None else float(roc_auc),
            "average_precision": None if ave_prec is None else float(ave_prec),
            "n_epochs": len(history.get("train", [])),
            "final_train_loss": float(history["train"][-1]) if history.get("train") else None,
            "final_val_loss": float(history["val"][-1]) if history.get("val") else None,
            "model_class": type(model).__name__,
            "optimizer_class": type(optimizer).__name__,
        }


        return summary, history
        