from ..registry.registry import DATALOADERS, PRETRAINING_TRAIN_LOADER, COLLATE_FN
from ..process_data import *
from torch.utils.data import DataLoader

@DATALOADERS.register("imputation")
def get_imputation_loaders(dls, fine_tune_config, optim_config, logger, **kwargs):
    optim_config["masking_ratio"] = 1.0 - fine_tune_config["i_ratio"]
    optim_config["mask_mode"] = "concurrent"
    train_ds = ImputationDataset(dls.train_ds, mean_mask_length=optim_config['mean_mask_length'],
                masking_ratio=optim_config['masking_ratio'], mode=optim_config['mask_mode'],
                distribution=optim_config['mask_distribution'], exclude_feats=optim_config['exclude_feats'], mask_row=False)
    valid_ds = ImputationDataset(dls.valid_ds, mean_mask_length=optim_config['mean_mask_length'],
                masking_ratio=optim_config['masking_ratio'], mode=optim_config['mask_mode'],
                distribution=optim_config['mask_distribution'], exclude_feats=optim_config['exclude_feats'])
    valid_dataloader = DataLoader(valid_ds, batch_size=1, collate_fn=collate_unsuperv)
    dataloader = DataLoader(train_ds, batch_size=optim_config["batch_size"], collate_fn=collate_unsuperv)
    logger.info("train_ds length: " + str(len(train_ds)) + ", valid_ds length: " + str(len(valid_ds)))
    return dataloader, valid_dataloader

@DATALOADERS.register("pretraining")
def get_pretrain_loaders(dls, optim_config, model_name, logger, **kwargs):
    train_ds = ImputationDataset(dls.train_ds, mean_mask_length=optim_config['mean_mask_length'],
                masking_ratio=optim_config['masking_ratio'], mode=optim_config['mask_mode'],
                distribution=optim_config['mask_distribution'], exclude_feats=optim_config['exclude_feats'], mask_row=False)
    valid_ds = ImputationDataset(dls.valid_ds, mean_mask_length=optim_config['mean_mask_length'],
                masking_ratio=optim_config['masking_ratio'], mode=optim_config['mask_mode'],
                distribution=optim_config['mask_distribution'], exclude_feats=optim_config['exclude_feats'])
    valid_dataloader = DataLoader(valid_ds, batch_size=1, collate_fn=collate_unsuperv)

    loader_config = {
        "train_ds": train_ds,
        "optim_config": optim_config,
        "collate_fn": COLLATE_FN.get(model_name)
    }
    dataloader = PRETRAINING_TRAIN_LOADER.get(model_name)(**loader_config)
    logger.info("train_ds length: " + str(len(train_ds)) + ", valid_ds length: " + str(len(valid_ds)))
    return  dataloader, valid_dataloader

@PRETRAINING_TRAIN_LOADER.register("mvts_transformer")
@PRETRAINING_TRAIN_LOADER.register("t_loss")
@PRETRAINING_TRAIN_LOADER.register("ts2vec")
def get_mvts_t_loss_loader(train_ds, optim_config, collate_fn, **kwargs):
    batch_size = optim_config["batch_size"]
    dataloader = DataLoader(train_ds, batch_size=batch_size, collate_fn=collate_fn)
    return dataloader

@PRETRAINING_TRAIN_LOADER.register("ts_tcc")
def get_ts_tcc(train_ds, optim_config, collate_fn, **kwargs):
    dataloader = DataLoader(train_ds, batch_size=optim_config["batch_size"], 
        collate_fn=lambda batch: collate_fn(batch, optim_config["jitter_scale_ratio"], 
        optim_config["jitter_ratio"], optim_config["max_seg"]))
    return dataloader

@DATALOADERS.register("classification")
@DATALOADERS.register("clustering")
def get_classification_loaders(dls, fine_tune_config, optim_config, logger, **kwargs):
    data = [dls.train_ds[i][0] for i in range(len(dls.train_ds))]
    label = [dls.train_ds[i][1] for i in range(len(dls.train_ds))]
    train_ds = ClassiregressionDataset(data, labels=label)
    data = [dls.valid_ds[i][0] for i in range(len(dls.valid_ds))]
    label = [dls.valid_ds[i][1] for i in range(len(dls.valid_ds))]
    valid_ds = ClassiregressionDataset(data, labels=label)
    dataloader = DataLoader(train_ds, batch_size=optim_config["batch_size"], collate_fn=collate_superv)
    valid_dataloader = DataLoader(valid_ds, batch_size=1, collate_fn=collate_superv)
    logger.info("train_ds length: " + str(len(train_ds)) + ", valid_ds length: " + str(len(valid_ds)))
    return dataloader, valid_dataloader

@DATALOADERS.register("regressPRETRAINING_LOADERion")
def get_regression_loader(dls, fine_tune_config, optim_config, logger, **kwargs):
    pred_len = fine_tune_config["pred_len"]
    pred_len = pred_len
    data = [dls.train_ds[i][0] for i in range(len(dls.train_ds))]
    train_ds = RegressionDataset(data)
    data = [dls.valid_ds[i][0] for i in range(len(dls.valid_ds))]
    valid_ds = RegressionDataset(data)
    dataloader = DataLoader(train_ds, batch_size=optim_config["batch_size"], collate_fn=
                                    lambda x: collate_superv_regression(x, pred_len=pred_len))
    valid_dataloader = DataLoader(valid_ds, batch_size=1, collate_fn=
                                        lambda x: collate_superv_regression(x, pred_len=pred_len))
    logger.info("train_ds length: " + str(len(train_ds)) + ", valid_ds length: " + str(len(valid_ds)))
    return dataloader, valid_dataloader

@DATALOADERS.register("anomaly_detection")
def get_anomaly_detection_loaders(dls, optim_config, logger, **kwargs):
    data = [dls.train_ds[i][0] for i in range(len(dls.train_ds))]
    label = [dls.train_ds[i][1] for i in range(len(dls.train_ds))]
    train_ds = ClassiregressionDataset(data, labels=label)
    data = [dls.valid_ds[i][0] for i in range(len(dls.valid_ds))]
    label = [dls.valid_ds[i][1] for i in range(len(dls.valid_ds))]
    valid_ds = ClassiregressionDataset(data, labels=label)
    dataloader = DataLoader(train_ds, batch_size=optim_config["batch_size"], collate_fn=collate_superv)
    valid_dataloader = DataLoader(valid_ds, batch_size=1, collate_fn=collate_superv)
    logger.info("train_ds length: " + str(len(train_ds)) + ", valid_ds length: " + str(len(valid_ds)))
    return dataloader, valid_dataloader