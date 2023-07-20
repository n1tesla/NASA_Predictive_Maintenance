import os

def make_dir(path:str):
    try:
        os.mkdir(path)
    except Exception as e:
        print(e)

def config_reader():
    import json
    f=open("config.json")
    config=json.load(f)
    return config

def make_paths(config):
    import datetime
    window_size = config["DATA"]["window_size"]
    stride_size = config["DATA"]["stride_size"]
    cwd = os.getcwd()
    architecture = 'lstm'
    observation = ''
    observation_name = f"{architecture}w{window_size}s{stride_size}{observation}"
    saved_models_path = cwd + "\\saved_models"
    observation_path = saved_models_path + "\\" + observation_name + "\\"

    if not os.path.exists(saved_models_path):
        make_dir(saved_models_path)

    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    make_dir(observation_path)
    time_path = observation_path + start_time
    make_dir(time_path)
    return time_path,observation_name,start_time
