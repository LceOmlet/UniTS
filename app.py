import myeel as eel
from random import randint
import torch
import torch.multiprocessing as mp
import threading
import logging
import time
import random
import tkinter as tk 
from tkinter import filedialog 
from ts_url.controller import process_training_tasks
logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


root = tk.Tk()
root.withdraw()

tasklist = mp.Manager().list()
eel.init("web")

# Exposing the random_python function to javascript
@eel.expose	
def random_python():
	print("Random function running")
	return randint(1,100)
	

lock = threading.Lock()

def fake_train(task):
	for i in range(100):
		state = {
			"task_id": task['task_id'],
			"progress": i + 1 ,
			"loss": random.random()
		}
		lock.acquire()
		eel.test_expose(state)
		lock.release()
		print(f"process: {task}")
		time.sleep(10)

def task_invoker():
	while True:
		# print(tasklist)
		if len(tasklist):
			lock.acquire()
			task = tasklist.pop(0)
			lock.release()
			### do task
			process_training_tasks(task)
			print(f"task done: {task}")

simple = threading.Thread(target = task_invoker)
simple.daemon = True
simple.start()
"""
task = {
	'data_file_list': [string, ...],
	'data_name': string,
	'model_name': string,
	'model_path': string,
	'model_hp_path': string
}
"""
@eel.expose						 # Expose this function to Javascript
def start_training(data_model):
	print(data_model)
	task = data_model["task"]
	data_file_list =  [data['sFile'] for data in  data_model['data']]
	data_name_list = [data['name'] for data in  data_model['data']]
	if task == "pretraining":
		task = {
			'data_file_list': data_file_list,
			'data_name': data_name_list,
			'model_name': data_model['name'],
			'model_hp_path': data_model['sFile'],
			"model_path": data_model['pFile'],
			'task_id': data_model["task_id"],
			"task": task
		}

	elif task == "classification":
		task = {
			"ckpts": [model["ckpt"] for model in data_model['model']],
			'data_file_list': data_file_list,
			'data_name': data_name_list,
			"task": task,
			'fusion': data_model["fusion"],
			'task_id': data_model["task_id"]
		}
	elif task == "clustering":
		task = {
			"ckpts": [model["ckpt"] for model in data_model['model']],
			'data_file_list': data_file_list,
			'data_name': data_name_list,
			"task": task,
			'fusion': data_model["fusion"],
			'task_id': data_model["task_id"]
		}
	elif task == "regression":
		pred_len = int(data_model["pred_len"])
		task = {
			"ckpts": [model["ckpt"] for model in data_model['model']],
			'data_file_list': data_file_list,
			'data_name': data_name_list,
			"task": task,
			'fusion': data_model["fusion"],
			"pred_len": pred_len,
			'task_id': data_model["task_id"]
		}
	elif task == "anomaly_detection":
		task = {
			"ckpts": [model["ckpt"] for model in data_model['model']],
			'data_file_list': data_file_list,
			'data_name': data_name_list,
			"task": task,
			'fusion': data_model["fusion"],
			'task_id': data_model["task_id"]
		}
	elif task == "imputation":
		task = {
			"ckpts": [model["ckpt"] for model in data_model['model']],
			'data_file_list': data_file_list,
			'data_name': data_name_list,
			"task": task,
			'fusion': data_model["fusion"],
			'i_ratio': float(data_model["i_ratio"]),
			'task_id': data_model["task_id"]
		}
	else:
		raise NotImplementedError("No such task: " + str(task))
	print(f"task queued: {task}")
	lock.acquire()
	tasklist.append(task)
	lock.release()
	

# eel.say_hello_js('Python World!') 

# training_process = mp.Process(target=task_invoker, args=())
# training_process.start()


# eel.say_hello_js('Python World!') 
# Start the index.html file
eel.start("html/pretrain.models.html", size=(1024, 760))
exit(0)
# training_process.close()