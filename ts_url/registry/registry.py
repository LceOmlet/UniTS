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




# EVALUATE: specific interpretations of evaluation process 
# which is called in every evaluation stage of each epoch.
# Members: "default"

# TRAIN_FN: specific interpretations of training process
# which is called in evary training stage of each epoch.
# Members: "default"
EVALUATE = Registry()
TRAIN_FN = Registry()



# DATALOADERS: functions for creating Dataloaders of different
# predifined tasks, which uses specific Dataset class and collate_fn
# functions to create Dataloaders.
# Member: "imputation", "pretraining", "clustering", "classfication"
# "regression", "anomaly_detection"

# PRETRAIN_LOADERS: functions for createing Dataloaders of different
# predified URL pretrain methods for time series.
# Member: "csl", "ts2vec", "t_loss", "mvts_transformer", "ts_tcc"

# COLLATE_FN: collate_fn for different training methods.
# Member: "csl", "ts2vec", "ts_tcc", "unsupervise", "t_loss"
# "mvts_transformer"
DATALOADERS = Registry()
PRETRAIN_LOADERS = Registry()
COLLATE_FN = Registry()


# MODELS: model classes with specific archietectures.
# Member: "ts_tcc", "t_loss", "ts2vec", "mvts_transformer", "csl"
MODELS = Registry()


# Members: "defalt"
EVALUATOR = Registry()
# Members: 'kmeans', 'svm', 'logistic_regression', 'ridge'
TEST_METHODS = Registry()

# Members: 'classification', 'anomaly_detection', 'regression', 
# 'imputation', 'pretraining'
TRAIN_STEP = Registry()
# Members: 'mvts_transformer', 'ts2vec', 'csl', 'ts_tcc', 't_loss'
PRETRAIN_STEP = Registry()
# Members: 'classification', 'anomaly_detection', 'regression', 
# 'imputation', 'pretraining'
EVALUATE_STEP = Registry()
# Members: 'mvts_transformer', 'csl', 't_loss', 'ts_tcc', 'ts2vec'
PRETRAIN_EVALUATE_STEP = Registry()

# Member: 'imputation', 'anomaly_detection', 'regression', 
# 'classification', 'clustering', 'pretraining'
LOSSES = Registry()
# Members: 'mvts_transformer', 'csl', 't_loss', 'ts_tcc', 'ts2vec'
PRETRAIN_LOSSES = Registry()
# Member: 'imputation', 'anomaly_detection', 'regression', 
# 'classification', 'clustering', 'pretraining'
MAKE_EVAL_REPORT = Registry()

# Member: 'pretraining', 't_loss', 'ts_tcc'
TRAINER_INIT = Registry()
# Member: 'csl'
TRAIN_LOOP_INIT = Registry()
# Member: 'pretraining'
EVAL_LOOP_INIT = Registry()