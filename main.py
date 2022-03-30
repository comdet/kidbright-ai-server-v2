from flask import Flask, render_template, request, copy_current_request_context, jsonify, send_file
from flask_socketio import SocketIO, send, emit
from flask_cors import CORS
import threading, queue
import ctypes

import sys, json, os, time, logging, random, shutil
import numpy as np
import cv2
import utils.helper as helper
sys.path.append(".")

from models.custom_classifier_model import create_classifier
from models.custom_yolo_model import create_yolo, get_dataset_labels

#logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "./"
PROJECT_FILENAME = "project.json"
TRAIN_FOLDER = "train"
TEST_FOLDER = "test"
VALIDATE_FOLDER = "valid"
DATASET_FOLDER = "dataset"
RAW_DATASET_FOLDER = "raw_dataset"
OUTPUT_FOLDER = "output"
TEMP_FOLDER = "temp"

STAGE = 0 #0 none, 1 = prepare dataset, 2 = training, 3 = trained, 4 = converting, 5 converted
report_queue = queue.Queue()
train_task = None
report_task = None

app = Flask(__name__)
#Set this argument to``'*'`` to allow all origins, or to ``[]`` to disable CORS handling.
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins = "*")
CORS(app)

@app.route('/', methods=["GET"])
def index():
    return jsonify({"result":"OK"})

@app.route('/upload', methods=["POST"])
def upload():
    if request.method == "POST":
        files = request.files.getlist("dataset")
        project_id = request.form['project_id']
        dataset_raw_path = os.path.join(PROJECT_PATH, project_id, RAW_DATASET_FOLDER)
        for file in files:
            target_path = os.path.join(dataset_raw_path,file.filename)
            file.save(target_path)
    return jsonify({"result":"OK"})

@app.route("/sync_project", methods=["POST"])
def sync_project():
    data = request.get_json() #content = request.get_json(silent=True)
    # ======== init project =======#
    project = data['project']['project']
    # check project dir exists
    project_path = os.path.join(PROJECT_PATH, project['id'])
    helper.create_not_exist(project_path)
    # write project file
    project_file = os.path.join(project_path,PROJECT_FILENAME)
    with open(project_file, 'w') as json_file:
        json.dump(data, json_file)
    # ========= sync project ======#
    # sync dataset
    RAW_PROJECT_DATASET = os.path.join(project_path,RAW_DATASET_FOLDER)
    helper.create_not_exist(RAW_PROJECT_DATASET) 
    dataset = data["dataset"]["dataset"]

    needed_filename = [i["id"]+"."+i["ext"] for i in dataset["data"]]
    needed_files = helper.sync_files(RAW_PROJECT_DATASET, needed_filename)
    res = "OK" if len(needed_files) == 0 else "SYNC"
    return jsonify({"result" : res, "needed" : needed_files})
    # =========================== #

def training_task(data, q):
    global STAGE, current_model
    try:
        # 1 ========== prepare project ========= #
        STAGE = 1
        q.put({"time":time.time(), "event": "initial", "msg" : "Start training step 1 ... prepare dataset"})
        project_id = data["project_id"]
        project_file = os.path.join(PROJECT_PATH, project_id, PROJECT_FILENAME)
        project = helper.read_json_file(project_file)
        model = project["project"]["project"]["model"]
        cmd_code = model["code"]
        config = helper.parse_json(cmd_code)
        q.put({"time":time.time(), "event": "initial", "msg" : "target project id : "+project_id})
        # 2 ========== prepare dataset ========= #
        train_type = config["model"]
        raw_dataset_path = os.path.join(PROJECT_PATH, project_id, RAW_DATASET_FOLDER)
        dataset = project["dataset"]["dataset"]["data"]
        
        #remove corrupted file
        
        # q.put({"time":time.time(), "event": "initial", "msg" : "check corrupted files ..."})
        # print("step 0 check corrupted files")
        # corrupted_file = helper.check_and_remove_corrupted_image(raw_dataset_path)
        # q.put({"time":time.time(), "event": "initial", "msg" : "found corrupted image file : " + str(len(corrupted_file))})
        # print("found corrupted image file : ")
        # print(corrupted_file)
        # if len(corrupted_file) > 0:
        #     dataset = [el for el in dataset if (el["id"]+"."+el["ext"]) not in corrupted_file]
        # q.put({"time":time.time(), "event": "initial", "msg" : "corrupted file has been removed from dataset"})
        
        random.shuffle(dataset)
        #################################################################
        # WARNING !!! we didn't distributed train/test split each class #
        #################################################################
        q.put({"time":time.time(), "event": "initial", "msg" : "prepare dataset ... contain " + str(len(dataset)) + " files" })
        train, valid = np.split(dataset,[int(len(dataset) * (config["train_rate"]/100.0))])
        print("training length : " + str(len(train)))
        print("validate length : " + str(len(valid)))
        q.put({"time":time.time(), "event": "initial", "msg" : "train length : " + str(len(train)) })
        q.put({"time":time.time(), "event": "initial", "msg" : "validate length : " + str(len(valid))})
        train_dataset_path = os.path.join(PROJECT_PATH, project_id, DATASET_FOLDER, TRAIN_FOLDER)
        valid_dataset_path = os.path.join(PROJECT_PATH, project_id, DATASET_FOLDER, VALIDATE_FOLDER)
        shutil.rmtree(train_dataset_path, ignore_errors=True)
        shutil.rmtree(valid_dataset_path, ignore_errors=True)
        
        if train_type == "mobilenet":
            q.put({"time":time.time(), "event": "initial", "msg" : "training type : classification(mobilenet)"})
            print("step 1 prepare dataset")
            #create folder with label
            shutil.rmtree(train_dataset_path, ignore_errors=True)
            shutil.rmtree(valid_dataset_path, ignore_errors=True)
            labels = helper.move_dataset_file_to_folder(train, raw_dataset_path, train_dataset_path)
            print("train data moved to : " + train_dataset_path)
            q.put({"time":time.time(), "event": "initial", "msg" : "train data moved to : " + train_dataset_path})
            helper.move_dataset_file_to_folder(valid, raw_dataset_path, valid_dataset_path)
            print("validate data moved to : " + valid_dataset_path)
            q.put({"time":time.time(), "event": "initial", "msg" : "validate data moved to : " + valid_dataset_path})
            print("labels : ")
            print(labels)
            q.put({"time":time.time(), "event": "initial", "msg" : "train label : " + ",".join(labels)})
            
            cmd_lines = cmd_code.split("\n")
            current_model, input_conf, output_conf = create_classifier(cmd_lines, labels)
            output_folder_path = os.path.join(PROJECT_PATH, project_id, OUTPUT_FOLDER)
            shutil.rmtree(output_folder_path, ignore_errors=True)
            helper.create_not_exist(output_folder_path)
            
            # stringlist = []
            # model.network.summary(print_fn=lambda x: stringlist.append(x))
            # model_summary = "\n".join(stringlist)
            # q.put({"time":time.time(), "event": "initial", "msg" : "model network : \n" + model_summary})
            STAGE = 2
            current_model.train(
                train_dataset_path,
                input_conf["epochs"],
                output_folder_path,
                batch_size = input_conf["batch_size"],
                augumentation = True,
                learning_rate = input_conf["learning_rate"], 
                train_times = input_conf["train_times"],
                valid_times = input_conf["valid_times"],
                valid_img_folder = valid_dataset_path,
                first_trainable_layer = None,
                metrics = output_conf["save_on"],
                callback_q = q,
                callback_sleep = None)
            STAGE = 3
            print("finish traing")

        elif train_type == "yolo":
            q.put({"time":time.time(), "event": "initial", "msg" : "training type : Yolo object detection"})
            # get label
            labels = get_dataset_labels(train)
            print("labels : ")
            print(labels)
            q.put({"time":time.time(), "event": "initial", "msg" : "train label : " + ",".join(labels)})
            
            cmd_lines = cmd_code.split("\n")

            current_model, input_conf, output_conf, anchors = create_yolo(cmd_lines,train, labels)
            
            output_folder_path = os.path.join(PROJECT_PATH, project_id, OUTPUT_FOLDER)
            shutil.rmtree(output_folder_path, ignore_errors=True)
            helper.create_not_exist(output_folder_path)
            
            STAGE = 2

            current_model.train(train,
                raw_dataset_path,
                valid,
                raw_dataset_path,
                input_conf["epochs"],
                output_folder_path,
                batch_size = input_conf["batch_size"],
                jitter = True, #augumentation
                learning_rate = input_conf["learning_rate"], 
                train_times = input_conf["train_times"],
                valid_times = input_conf["valid_times"],
                metrics = output_conf["save_on"],
                callback_q = q,
                callback_sleep = None)
            STAGE = 3
            q.put({"time":time.time(), "event": "train_end", "msg" : "Train ended", "matric" : None})
            
            # print("finish traing")



    finally:
        print("Thread ended")

def kill_thread(target_thread):
    target_thread_id = None
    if hasattr(target_thread, '_thread_id'):
        target_thread_id = target_thread._thread_id
    else:
        for id, thread in threading._active.items():
            if thread is target_thread:
                target_thread_id = id
    if target_thread_id != None: 
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(target_thread_id,ctypes.py_object(SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(target_thread_id, 0)
            print('Exception raise failure')

@app.route("/start_training", methods=["POST"])
def start_training():
    global train_task, report_queue
    print("start training process")
    data = request.get_json()
    train_task = threading.Thread(target=training_task, args=(data,report_queue,))
    train_task.start()
    return jsonify({"result" : "OK"})

@app.route("/terminate_training", methods=["POST"])
def terminate_training():
    global train_task, report_task, report_queue
    print("terminate current training process")
    if train_task and train_task.is_alive():
        kill_thread(train_task)
        print("send kill command")
    #if report_task and report_task.is_alive():
    #    report_queue.put({"time": time.time(), "event" : "terminate", "msg":"Training terminated"})
    time.sleep(3)
    return jsonify({"result" : "OK"})

@app.route("/download_model", methods=["GET"])
def handle_download_model():
    print("download model file")
    project_id = request.args.get("project_id")
    if not project_id:
        return "Fail"
    if not os.path.exists(project_id):
        return "Fail"
    output_path = os.path.join(project_id,"output")
    files = [os.path.join(output_path,f) for f in os.listdir(output_path) if f.endswith(".h5")]
    if len(files) <= 0:
        return "Fail"
    return send_file(files[0], as_attachment=True)
        
@app.route("/inference_image", methods=["POST"])
def handle_inference_model():
    global STAGE, current_model
    if 'image' not in request.files:
        return "No image"
    if STAGE < 3:
        return "Training not success yet :" + str(STAGE)
    
    tmp_img = request.files['image']
    project_id = request.form['project_id']
    if not tmp_img:
        return "Image null or something"
    
    target_file_path = os.path.join(project_id,TEMP_FOLDER)
    helper.create_not_exist(target_file_path) 
    target_file = os.path.join(target_file_path, tmp_img.filename)
    tmp_img.save(target_file)
    orig_image, img = helper.prepare_image(target_file, current_model, current_model.input_size)
    elapsed_ms, prob, prediction = current_model.predict(img)
    return jsonify({"result" : "OK","prediction":prediction, "prob":np.float64(prob)})

@socketio.on('connect')
def client_connect():
    global report_task, report_queue
    print("client connected")
    # check prev thread are alive
    if report_task and report_task.is_alive():
        report_queue.put({"time": time.time(), "event" : "terminate", "msg":"terminate client"})
        time.sleep(1)
    @copy_current_request_context
    def train_callback(q):
        while True:
            try:
                report = q.get(block=True, timeout=None)
                if report["event"] == "train_end":
                    print("traning end")
                    #break
                if report["event"] == "terminate":
                    print("terminate thread called")
                    break
                else:
                    emit("training_progress",report)
            except Queue.Empty:
                continue
            except:
                print("Unknow Error")
                break 
    report_task = threading.Thread(target=train_callback, args=(report_queue,))
    report_task.start()

@socketio.on('disconnect')
def client_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app,debug=True)
    #data = {"project_id" : "project-sdfsdf-B51tiO"}
    #training_task(data,report_queue)