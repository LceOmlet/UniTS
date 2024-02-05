class Registry:
    def __init__(self):
        self._registry = {}

    def register(self, name):
        def decorator(func_or_class):
            self._registry[name] = func_or_class
            return func_or_class
        return decorator

    def get(self, name):
        return self._registry.get(name)
    
EVALUATORS = Registry()
TRAINERS = Registry()
DATALOADERS = Registry()
MODELS = Registry()
PRETRAINING_TRAIN_LOADER = Registry()
COLLATE_FN = Registry()
TRAIN_STEP = Registry()
TRAIN_AGG = Registry()

PRETRAIN_STEP = Registry()
LOSSES = Registry()
PRETRAIN_LOSSES = Registry()
EVALUATE_STEP = Registry()
PRETRAIN_EVALUATE_INFER = Registry()
EVALUATE_AGG = Registry()
TRAIN_LOOP_INIT = Registry()
EVAL_LOOP_INIT = Registry()

TRAINER_INIT = Registry()
TEST_MODULE = Registry()