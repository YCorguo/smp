import subprocess
import os

result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value

from datasets import load_dataset

from huggingface_hub import login
login(token="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")

# Load Train dataset
ds_train_posts = load_dataset("smpchallenge/SMP-Video", 'posts')['train']
ds_train_users = load_dataset("smpchallenge/SMP-Video", 'users')['train']
ds_train_videos = load_dataset("smpchallenge/SMP-Video", 'videos')['train']
ds_train_labels = load_dataset("smpchallenge/SMP-Video", 'labels')['train']

# Load test dataset
ds_test_posts = load_dataset("smpchallenge/SMP-Video", 'posts')['test']
ds_test_users = load_dataset("smpchallenge/SMP-Video", 'users')['test']
ds_test_videos = load_dataset("smpchallenge/SMP-Video", 'videos')['test']

import tqdm
import types
import dill

def save_workspace_variables(filename, variables):
    """
    将工作区变量保存到指定文件，过滤掉不可序列化的对象，如模块、函数等。

    参数:
    filename -- 保存的文件路径
    variables -- 当前工作区的变量字典，包含所有变量
    
    返回:
    无
    """
    
    # 过滤掉无法序列化的对象（模块、函数、方法等）
    workspace_variables = {key: value for key, value in variables.items() 
                           if not (key.startswith("__")  # 排除私有变量
                                   or isinstance(value, types.ModuleType)  # 排除模块
                                   or isinstance(value, types.FunctionType)  # 排除函数
                                   or isinstance(value, types.MethodType))  # 排除方法
                           and is_serializable(value)}  # 只保留可序列化的对象
    
    # 打开文件并使用 dill 库序列化保存工作区变量
    with open(filename, "wb") as file:
        dill.dump(workspace_variables, file)

def load_workspace_variables(filename):
    """从指定文件加载工作区变量并更新到全局命名空间"""
    with open(filename, "rb") as file:
        # 使用 dill 库反序列化对象并加载
        workspace_variables = dill.load(file)
    # 将加载的工作区变量更新到全局命名空间
    globals().update(workspace_variables)

def is_serializable(obj):
    """尝试序列化对象，若失败则返回False"""
    try:
        # 使用 dill 库尝试序列化对象
        dill.dumps(obj)
    except (TypeError, dill.PicklingError):
        # 如果对象无法序列化，捕获异常并返回 False
        return False
    # 如果序列化成功，返回 True
    return True

low_level_feature_file = "video_data.pkl"
save_workspace_variables(low_level_feature_file, globals())
# globals().update(dill.load(open(low_level_feature_file, "rb")))
