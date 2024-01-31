from .training_methods import Trainer
import json
import threading
import myeel as eel
from .models.default_configs.configues import optim_configures, task_configures, model_configures
from tkinter.filedialog import (askdirectory, askopenfile, askopenfilename)
import tkinter as tk
import torch
import numpy as np
import random
import time
import os
import logging
import traceback
from .visualization import visualize_sample_, plot_loss


def setup_logger(name, log_file, level=logging.INFO):
	"""To setup as many loggers as you want"""
	handler = logging.FileHandler(log_file)   
	formatter = logging.Formatter('%(asctime)s | %(levelname)s : %(message)s')	 
	handler.setFormatter(formatter)
	logger = logging.getLogger(name)
	logger.setLevel(level)
	logger.addHandler(handler)
	return logger

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

lock = threading.Lock()
training_states = []
classification_states = []
forecasting_states = []
clustering_states = []
anomaly_detection_states = []
imputation_states = []
fusion_methods = ["concat", "projection"]

per_batch_load_states = {
	"pretraining": dict(),
	"classification": dict(),
	"anomaly_detection": dict(),
	"imputation": dict(),
	"regression": dict(),
	"clustering": dict()
}
pbls_lock = threading.Lock()


@eel.expose
def get_task_samples_num(task_name, task_id):
	task_id = int(task_id)
	if task_id in per_batch_load_states[task_name]:
		per_batch_path = per_batch_load_states[task_name][task_id]
		if os.path.exists(per_batch_path["per_batch_path"] + ".json"):
			with per_batch_path["file_lock"]:
				with open(per_batch_path["per_batch_path"] + ".json", mode="r") as f:
					per_batch_json = json.load(f)
			return per_batch_json["X"]["length"]
	return 0
# parsed_expamle:
{
	"task_id": 1,
	"task_name": "pretraining",
	"sample_index": 10
}
@eel.expose
def visualize_sample(task_name, task_id, sample_index):
	print(task_name, task_id, sample_index)
	if sample_index is not None and sample_index != "null":
		sample_index = int(sample_index)
	else:
		sample_index = None
	task_id = int(task_id)
	if task_id in per_batch_load_states[task_name]:
		per_batch_path = per_batch_load_states[task_name][task_id]
		if os.path.exists(per_batch_path["per_batch_path"]):
			with per_batch_path["file_lock"]:
				per_batch = np.load(per_batch_path["per_batch_path"], allow_pickle=True)
				per_batch = dict(per_batch) 
			visualize_sample_(per_batch, task_name, sample_index)
			return True
	return False

@eel.expose
def visualize_loss(task_name, task_id):
	print("loss called.")
	task_id = int(task_id)
	if task_id in per_batch_load_states[task_name]:
		per_batch_path = per_batch_load_states[task_name][task_id]
		if os.path.exists(per_batch_path["epoch_losses_path"]):
			with per_batch_path["file_lock"]:
				with open(per_batch_path["epoch_losses_path"], mode="r") as f:
					epoch_losses = json.load(f)
			plot_loss(epoch_losses["train_loss"], epoch_losses["valid_loss"])
			return True
	return False


@eel.expose
def send_imputation_states_from_js(imputation_states_):
	global imputation_states
	lock.acquire()
	imputation_states = imputation_states_
	lock.release()

@eel.expose
def send_imputation_states_from_python():
	global imputation_states
	return imputation_states

@eel.expose
def send_anomaly_detection_states_from_python():
	global anomaly_detection_states
	return anomaly_detection_states

@eel.expose
def send_anomaly_detection_states_from_js(anomaly_detection_states_):
	global anomaly_detection_states
	lock.acquire()
	anomaly_detection_states = anomaly_detection_states_
	lock.release()


@eel.expose
def get_fusion_methods():
	global fusion_methods
	return fusion_methods

@eel.expose
def send_classification_states_from_python():
	global classification_states
	return classification_states

@eel.expose
def send_classification_states_from_js(classification_states_):
	global classification_states
	lock.acquire()
	classification_states = classification_states_
	lock.release()

@eel.expose
def send_forecasting_states_from_python():
	global forecasting_states
	return forecasting_states

@eel.expose
def send_forecasting_states_from_js(forecasting_states_):
	global forecasting_states
	lock.acquire()
	forecasting_states = forecasting_states_
	lock.release()

@eel.expose
def send_clustering_states_from_python():
	global clustering_states
	return clustering_states

@eel.expose
def send_clustering_states_from_js(clustering_states_):
	global clustering_states
	lock.acquire()
	clustering_states = clustering_states_
	lock.release()

@eel.expose
def send_training_states_from_python():
	global training_states
	return training_states

@eel.expose
def send_training_states_from_js(training_states_):
	global training_states
	lock.acquire()
	training_states = training_states_
	lock.release()
	print("set neew item stats: " + str(training_states))

@eel.expose
def get_model_names():
	print(str(list(optim_configures.keys())))
	return list(optim_configures.keys())

@eel.expose
def get_ckpt_dir():
	# root.lift()
	dir_name = askdirectory(initialdir='./ckpts')
	return dir_name

@eel.expose
def select_file():
	# root.lift()
	filepath = askdirectory(initialdir="./datasets")
	return filepath

@eel.expose
def select_p_file():
	# root.lift()
	filepath = askopenfilename(initialdir="./ts_url/models/default_configs")
	return filepath

def process_training_tasks(task):
	print(task)
	file_lock = threading.Lock()
	task_name = task["task"]
	task_id = int(task['task_id'])
	if task_name != "pretraining":
		with open(task_configures[task['task']], "r") as oc:
			optim_config = json.load(oc)
		ckpt = task["ckpts"]
		fusion = task["fusion"]
		fine_tune_config = {"fusion": fusion}
		if task_name == "regression":
			fine_tune_config["pred_len"] = task["pred_len"]
		if task_name == "imputation":
			fine_tune_config["i_ratio"] = task["i_ratio"]
		# if task_name == "anomaly_detection":
		# 	fine_tune_config["pred_len"] = 1e7
		model_name = None
		hp_path = None
		p_path = None
	elif task_name == "pretraining":
		with open(optim_configures[task['model_name']], "r") as oc:
			optim_config = json.load(oc)
		ckpt = None
		fusion = None
		model_name = task['model_name']
		hp_path = task['model_hp_path']
		p_path = task['model_path']
		if p_path is None or p_path == "":
			p_path = model_configures[model_name]
		fine_tune_config = None

	data_configs = []
	for i in range(len(task["data_name"])):
		data_configs.append({
			"filepath": task["data_file_list"][i],
			"dsid": task["data_name"][i],
			"train_ratio": 1,
			"test_ratio": 1
		})
	print(ckpt)
	start_time = time.strftime("%m_%d_%H_%M_%S", time.localtime()) 
	
	if task_name == "pretraining":
		task_summary = "_".join(task["data_name"]) + "_" + model_name
		save_name = start_time + "_" + task_summary
	else:
		task_summary = "_".join(task["data_name"]) + "_" + ".".join([c.split("/")[-1] for c in ckpt])
		save_name = start_time + "_" + task_summary
	if task_name == "classification":
		save_path = os.path.join("classification_ckpt", save_name)
	elif task_name == "clustering":
		save_path = os.path.join("clustering_ckpt", save_name)
	elif task_name == "regression":
		save_path = os.path.join("forcasting_ckpt", save_name)
	elif task_name == "anomaly_detection":
		save_path = os.path.join("anomaly_detection_ckpt", save_name)
	elif task_name == "imputation":
		save_path = os.path.join("imputation_ckpt", save_name)
	else:
		save_path = os.path.join("ckpts", save_name)
	best_predictions_path = "best_predictions.npz"
	per_batch_path = os.path.join(save_path, best_predictions_path)
	epoch_losses_path = "epoch_losses.json"
	epoch_losses_path = os.path.join(save_path, epoch_losses_path)

	with pbls_lock:
		per_batch_load_states[task_name][task_id] = {
			"file_lock": file_lock, 
			"per_batch_path": per_batch_path,
			"epoch_losses_path": epoch_losses_path
			}

	os.makedirs(save_path, exist_ok=True)
	logger = setup_logger("__main__." + save_name, os.path.join(save_path, "run.log"))
	try:
		print(data_configs)
		print(ckpt)
		print(save_name)
		print(save_path)
		if task_name == "anomaly_detection":
			key_metric = "auc"
		else:
			key_metric = "loss"
		# 判断当前是否有可用的显卡
		if torch.cuda.is_available():
			# 设置使用的显卡编号，可以使用多个显卡，编号从0开始
			device = torch.device("cuda:0")
			# 设置在显卡上运行
			torch.cuda.set_device(device)
			print("Using CUDA")
		else:
			# 如果没有可用的显卡，就在 CPU 上运行
			None
		device = torch.device("cpu")
		print("Using CPU")
		trainer = Trainer(data_configs, model_name, p_path, 
					device=device, task=task_name, optim_config=optim_config, fine_tune_config=fine_tune_config, logger=logger, ckpt_paths=ckpt)
		# save the loss value for each epoch
		epoch_losses = {
				"train_loss": [],
				"valid_loss": []
			}
		just_valid = False
		if optim_config['epochs'] == 0:
			optim_config['epochs'] = 1
			just_valid = True
		for epoch in range(optim_config['epochs']):
			epoch_metrics = None

			if not just_valid : epoch_metrics = trainer.train_epoch(epoch_num=epoch)
			aggr_metrics, best_metrics, best_value = trainer.validate(epoch, key_metric, save_path, best_predictions_path)
			if epoch_metrics is not None:
				epoch_losses["train_loss"].append(epoch_metrics["loss"])
				epoch_losses["valid_loss"].append(aggr_metrics["loss"])
				with pbls_lock, file_lock:
					with open(epoch_losses_path, "w") as f:
						json.dump(epoch_losses, f)
			best_value = float(best_value)
			if task_name == "classification":
				state = {
					"task_id": task_id,
					"progress": int((epoch + 1) / optim_config["epochs"] * 100),
					"fusion": task["fusion"],
					"alg": "; ".join(ckpt),
					"loss": best_value,
					"alg_title": "\n".join(ckpt),
					"data": "/".join(task["data_name"]),
					"report": best_metrics["report"] 
				}
			elif task_name == "pretraining":
				# logger.info("best Loss: " + str(best_value))
				state = {
					"task_id": task_id,
					"progress": int((epoch + 1) / optim_config["epochs"] * 100),
					"loss": float(best_value),
					"alg": task['model_name'],
					"data": "/".join(task["data_name"])
				}
			elif task_name == "clustering":
				state = {
					"task_id": task_id,
					"progress": int((epoch + 1) / optim_config["epochs"] * 100),
					"fusion": task["fusion"],
					"alg": "; ".join(ckpt),
					"loss": str(best_value),
					"alg_title": "\n".join(ckpt),
					"data": "/".join(task["data_name"]),
					"report": best_metrics["report"] 
				}
			elif task_name == "regression":
				state = {
					"task_id": task_id,
					"progress": int((epoch + 1) / optim_config["epochs"] * 100),
					"fusion": task["fusion"],
					"pred_len": task["pred_len"],
					"alg": "; ".join(ckpt),
					"loss": best_value,
					"alg_title": "\n".join(ckpt),
					"data": "/".join(task["data_name"]),
					"report": best_metrics["report"] 
				}
			elif task_name == "anomaly_detection":
				state = {
					"task_id": task_id,
					"progress": int((epoch + 1) / optim_config["epochs"] * 100),
					"fusion": task["fusion"],
					"alg": "; ".join(ckpt),
					"loss": best_value,
					"alg_title": "\n".join(ckpt),
					"data": "/".join(task["data_name"]),
					"report": best_metrics["report"] 
				}
			elif task_name == "imputation":
				state = {
					"task_id": task_id,
					"progress": int((epoch + 1) / optim_config["epochs"] * 100),
					"fusion": task["fusion"],
					"i_ratio": task["i_ratio"],
					"alg": "; ".join(ckpt),
					"loss": best_value,
					"alg_title": "\n".join(ckpt),
					"data": "/".join(task["data_name"]),
					"report": best_metrics["loss"] 
				}
			else:
				raise NotImplementedError()
			lock.acquire()
			if task_name == "classification":
				states = classification_states
			elif task_name == "pretraining":
				states = training_states
			elif task_name == "clustering":
				states = clustering_states
			elif task_name == "regression":
				states = forecasting_states
			elif task_name == "anomaly_detection":
				states = anomaly_detection_states
			elif task_name == "imputation":
				states = imputation_states
			else:
				raise NotImplementedError()			
			states[task_id] = state
			# print(states)
			eel.get_new_state(states, task_name)
			lock.release()
		
	except Exception:
		trace = traceback.format_exc()
		logger.error(trace)
