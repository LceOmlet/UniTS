from ..registry.registry import DATALOADERS, PRETRAIN_LOADERS, COLLATE_FN, DATASET
from ..process_data import *
from torch.utils.data import DataLoader

@DATALOADERS.register("imputation")
def get_imputation_loaders(dls, fine_tune_config, optim_config, logger, **kwargs):
    optim_config["masking_ratio"] = 1.0 - fine_tune_config["i_ratio"]
    optim_config["mask_mode"] = "concurrent"
    train_ds = [(dls.train_ds[i][0],) for i in range(len(dls.train_ds))]
    valid_ds = [(dls.valid_ds[i][0],) for i in range(len(dls.valid_ds))]
    train_ds = ImputationDataset(train_ds, mask_row=False, shuffle=False, **optim_config)
    valid_ds = ImputationDataset(valid_ds, **optim_config)
    valid_dataloader = DataLoader(valid_ds, batch_size=64, collate_fn=COLLATE_FN.get("unsupervise"))
    dataloader = DataLoader(train_ds, batch_size=optim_config["batch_size"], collate_fn=COLLATE_FN.get("unsupervise"), drop_last=True)
    logger.info("train_ds length: " + str(len(train_ds)) + ", valid_ds length: " + str(len(valid_ds)))
    return dataloader, valid_dataloader

@DATALOADERS.register("pretraining")
def get_PRETRAINLOADERSs(dls, optim_config, model_name, logger, data_configs, device, **kwargs):
    # print(data_configs)
    d_name = "_".join([d['dsid'] for d in data_configs])
    data = [dls.train_ds[i][0] for i in range(len(dls.train_ds))]
    label = [dls.train_ds[i][1] for i in range(len(dls.train_ds))]
    dataset_kwargs = dict(data=data, 
        label=label, 
        d_name=d_name, 
        mask_row=False, 
        optim_config=optim_config, 
        device=device
    )
    DataSet = DATASET.get(model_name)
    train_ds = DataSet(**dataset_kwargs)
    dataset_kwargs.update(optim_config)
    data = [dls.valid_ds[i][0] for i in range(len(dls.valid_ds))]
    label = [dls.valid_ds[i][1] for i in range(len(dls.valid_ds))]
    # print(f"valid anom num: {sum(label)}")
    # raise RuntimeError()
    valid_ds = ImputationDataset(data, label=label, shuffle=False, **optim_config)
    valid_dataloader = DataLoader(valid_ds, batch_size=optim_config["batch_size"], collate_fn=COLLATE_FN.get("unsupervise"), shuffle=False)
    loader_config = dict(train_ds=train_ds, 
        optim_config=optim_config, 
        collate_fn=COLLATE_FN.get(model_name)
    )

    dataloader = PRETRAIN_LOADERS.get(model_name)(**loader_config, drop_last=True)
    # print(len(dataloader.dataset), len(valid_dataloader.dataset))
    logger.info("train_ds length: " + str(len(train_ds)) + ", valid_ds length: " + str(len(valid_ds)))
    return  dataloader, valid_dataloader

@PRETRAIN_LOADERS.register("default")
def get_mvts_t_loss_loader(train_ds, optim_config, collate_fn, **kwargs):
    batch_size = optim_config["batch_size"]
    dataloader = DataLoader(train_ds, batch_size=batch_size, collate_fn=collate_fn, drop_last=True)
    return dataloader

@PRETRAIN_LOADERS.register("ts_tcc")
def get_ts_tcc(train_ds, optim_config, collate_fn, **kwargs):
    dataloader = DataLoader(train_ds, batch_size=optim_config["batch_size"], 
        collate_fn=lambda batch: collate_fn(batch, optim_config["jitter_scale_ratio"], 
        optim_config["jitter_ratio"], optim_config["max_seg"]),  drop_last=True)
    return dataloader

@DATALOADERS.register("classification")
@DATALOADERS.register("clustering")
def get_classification_loaders(dls, fine_tune_config, optim_config, logger, **kwargs):
    data = [dls.train_ds[i][0] for i in range(len(dls.train_ds))]
    label = [dls.train_ds[i][1] for i in range(len(dls.train_ds))]
    data = [dls.valid_ds[i][0] for i in range(len(dls.valid_ds))]
    label = [dls.valid_ds[i][1] for i in range(len(dls.valid_ds))]
    train_ds = ClassiregressionDataset(data, labels=label)
    valid_ds = ClassiregressionDataset(data, labels=label)
    dataloader = DataLoader(train_ds, batch_size=optim_config["batch_size"], collate_fn=collate_superv)
    valid_dataloader = DataLoader(valid_ds, batch_size=1, collate_fn=collate_superv)
    logger.info("train_ds length: " + str(len(train_ds)) + ", valid_ds length: " + str(len(valid_ds)))
    return dataloader, valid_dataloader

@DATALOADERS.register("regression")
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