from src.utils.general_utils import set_seed



class ExperimentRunner:
    def __init__(self, trainer, seeds: list[int], train_loader, val_loader, test_loader, max_epochs, model_builder, optimizer_builder, regularizer_builder, lrScheduler_builder, **trainer_kwargs):
        self.trainer = trainer
        self.seeds = seeds
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.max_epochs = max_epochs
        self.model_builder = model_builder
        self.optimizer_builder = optimizer_builder
        self.trainer_kwargs = trainer_kwargs
        self.regularizer_builder = regularizer_builder
        self.lrScheduler_builder = lrScheduler_builder
        self.history = {}

    def run(self):
        results = {}
        print("Train Run is starting...")
        for run_idx, seed in enumerate(self.seeds):
            total_runs = len(self.seeds)
            print("\n" + "=" * 60)
            print(f" RUN {run_idx+1} / {total_runs}   (seed = {seed})")
            print("=" * 60)
            model, optimizer, regularizer, lr_scheduler_builder_instance = self._build_components()
            trainer = self.trainer(model = model, 
                                   optimizer = optimizer,
                                   Regularizer = regularizer,
                                   LR_Scheduler_Builder = lr_scheduler_builder_instance,
                                   **self.trainer_kwargs)
            
            trainer.fit(self.train_loader, self.val_loader, self.max_epochs)
            trainer.evaluate_model(self.test_loader)
            history, roc_auc, ave_prec = trainer._return_history()
            results[seed] = {
                "trainer": trainer,
                "history": history,
                "metrics": {
                    "roc_auc": roc_auc,
                    "average_precision": ave_prec,
                },
            }
            print("\n[Evaluation – Test]")
            print(f"  • ROC-AUC        : {roc_auc:.4f}")
            print(f"  • Avg Precision  : {ave_prec:.4f}")
            print("-" * 60)



        return results
    
    def _build_components(self):
        model = self.model_builder()
        optimizer = self.optimizer_builder(model)
        regularizer = self.regularizer_builder()
        lr_scheduler_builder_instance = self.lrScheduler_builder()
        return model, optimizer, regularizer, lr_scheduler_builder_instance