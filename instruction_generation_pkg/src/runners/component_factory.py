class ComponentFactory:
    def __init__(self, model_builder, optimizer_builder,
                 regularizer_builder, lr_scheduler_builder):
        self.model_builder = model_builder
        self.optimizer_builder = optimizer_builder
        self.regularizer_builder = regularizer_builder
        self.lr_scheduler_builder = lr_scheduler_builder

    def build(self):
        model = self.model_builder()
        optimizer = self.optimizer_builder(model)
        regularizer = self.regularizer_builder()
        lr_sched_builder = self.lr_scheduler_builder()
        return model, optimizer, regularizer, lr_sched_builder
