from src.runners.run_config import RunConfig


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
        return {
            "seed": seed,
            "trainer": trainer,
            "history": history,
            "metrics": {
                "roc_auc": roc_auc,
                "average_precision": ave_prec,
            },
        }