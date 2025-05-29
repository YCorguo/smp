# 操作系统相关功能，例如路径处理、文件操作等
import os

# 随机数生成器，用于生成随机数据或进行随机抽样
import random

# 内存管理工具，帮助手动释放内存
import gc

# 深度学习框架 PyTorch 的核心库
import torch

# PIL (Python Imaging Library) 用于图像处理的库
import PIL
from PIL import Image  # 从 PIL 中导入 Image 用于加载和操作图像文件

# 用于生成和验证数据的哈希值，常用于文件验证或加密
import hashlib

# 数据分析库，提供高效的数据结构和操作功能
import pandas as pd

# 数学和数值计算库，提供常见的数学函数
import numpy as np

# 处理 JSON 数据（如读取和写入 JSON 格式的数据）
import json

# 用于加载深度学习模型的 mmpretrain 库
from mmpretrain import get_model

# PyTorch 的数据集和数据加载器，用于处理数据集
from torch.utils.data import Dataset, DataLoader

# 图像转换相关功能，包括标准化、裁剪、旋转等
import torchvision.transforms as transforms
from torchvision.transforms import functional as F  # 提供直接的图像操作函数

# 用于显示进度条的库，通常与循环结合使用
import tqdm

# OpenCV 库，用于图像和视频处理，常用于计算机视觉任务
import cv2

# skimage (scikit-image) 提供了多种图像处理功能
from skimage.feature import graycomatrix, graycoprops  # 提供灰度共生矩阵计算和特征提取
from skimage import img_as_ubyte  # 用于将图像转换为无符号字节格式（适用于保存图像）

# SciPy 提供统计学和数值优化功能
import scipy.stats as stats  # 提供各种统计检验和分布函数

# Pandas 用于处理数据框（DataFrame）和系列（Series）
import pandas as pd

# Seaborn 是基于 matplotlib 的数据可视化库，简化了复杂图表的创建
import seaborn as sns

# Matplotlib 用于绘制各种类型的图表，如折线图、散点图、柱状图等
import matplotlib.pyplot as plt

# sklearn 中用于数据预处理和特征缩放的类和函数
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

# statsmodels 提供统计建模功能，常用于线性回归、时间序列分析等
import statsmodels.api as sm

# scikit-learn 中用于模型训练和评估的工具
from sklearn.model_selection import train_test_split  # 用于数据集的分割
from sklearn.metrics import mean_squared_error, r2_score  # 用于模型性能评估

# datetime 模块提供日期和时间的操作功能
from datetime import datetime
from datetime import datetime, timedelta  # 用于日期的加减和格式化

# scipy.stats 提供卡方检验和 t 检验等统计检验功能
from scipy.stats import chi2_contingency, ttest_ind

# Counter 是一个字典子类，用于计数操作，常用于统计元素出现的次数
from collections import Counter

# pickle 库用于将 Python 对象序列化为字节流，或将字节流反序列化为对象
import pickle

# dill 是 pickle 的增强版，提供更强大的序列化功能，支持更多数据类型
import dill

# types 模块提供对 Python 类型的访问和管理
import types

# warnings 模块用于控制警告信息
import warnings

# ks_2samp 进行 Kolmogorov-Smirnov 检验，常用于比较两个样本是否来自同一分布
from scipy.stats import ks_2samp

# concurrent.futures 提供并发编程的工具，支持线程池和进程池，便于多任务并行执行
import concurrent.futures

import re
import pandas as pd
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, f_oneway
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import shap
from sklearn.metrics import mean_squared_error

import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

from scipy.stats import spearmanr


# 忽略特定类型的警告
warnings.filterwarnings("ignore", category=RuntimeWarning)  # 忽略运行时警告
warnings.filterwarnings("ignore", category=UserWarning)  # 忽略用户警告
warnings.simplefilter(action='ignore', category=FutureWarning)  # 忽略未来警告

# 函数：将字符串按逗号分隔并返回最后一个部分
def leave_str_only_last_part_behind_comma(strr):
    # 如果传入的字符串为空，返回 None
    if not strr:
        return None
    # 如果字符串中有逗号，则按逗号分割，获取最后一部分
    if ',' in strr:
        strr = strr.split(',')[-1]
    # 去除字符串两端的空白字符并返回
    return strr.strip()

# 函数：处理日期时间字符串，将其转换为时间戳
def process_date_time(time_str):
        # 输入示例： "2008-02-11 17:18:31"
        # 使用 strptime 将字符串转换为 datetime 对象
        dt_object = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        # 获取该日期时间对象的 Unix 时间戳（秒数）
        timestamp_seconds = int(dt_object.timestamp())
        # 返回时间戳
        return timestamp_seconds

# 创建一个自定义的数据集类，继承自 PyTorch 的 Dataset 类
class ConstructDataset(Dataset):
    # 初始化方法，接受包含图像数据的 DataFrame
    def __init__(self, img_data):
        # 将图像数据保存为 DataFrame 属性
        self.dataframe = img_data
        # 定义图像转换（如调整大小、转换为 Tensor 等）
        img_classification_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 将图像调整为 224x224 的大小
            transforms.ToTensor(),  # 将图像转换为 Tensor 格式
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 可选：标准化图像
        ])
        # 保存图像转换操作
        self.transform = img_classification_transform

    # 定义该数据集的长度，即图像数据的总数
    def __len__(self):
        return len(self.dataframe)

    # 根据索引获取图像及其标签
    def __getitem__(self, idx):
        # 根据索引获取图像路径
        img_path = self.dataframe.loc[idx, 'image_path']
        # 使用 PIL 打开图像并转换为 RGB 模式（3 通道）
        image = Image.open(img_path).convert('RGB')
        # 根据索引获取标签（如分类标签）
        label = self.dataframe.loc[idx, 'label']

        # 如果定义了图像转换操作，则进行转换（如调整大小、转为 Tensor 等）
        if self.transform:
            image = self.transform(image)

        # 返回图像和标签
        return image, label
    
# classical statistic features

# 定义一个图像分类的类
class ImageClassification():
    # 初始化方法，接受图像数据集
    def __init__(self, img_data):
        # 设置 ImageNet 1K 类别标签文件的路径
        self.imagenet_1k_labels_file = os.path.join(data_path, "imagenet-simple-labels.json")
        # 载入 ImageNet 1K 类别标签文件
        self.imagenet_1k_labels = json.load(open(self.imagenet_1k_labels_file))
        # 存储传入的图像数据集
        self.img_dataset = img_data
        # 创建数据加载器，用于批量加载图像数据集
        data_loader = DataLoader(self.img_dataset, batch_size=32, shuffle=False)
        # 获取预训练的 EfficientNet-B0 图像分类模型并加载到设备上
        img_classification_model = get_model('efficientnet-b0_3rdparty_8xb32_in1k', pretrained=True).to(device)
        # 设置模型为评估模式
        img_classification_model.eval()
        outputs = []  # 存储预测结果
        # 禁用梯度计算，避免计算图的创建，节省内存和计算
        with torch.no_grad():
            # 遍历数据加载器，获取每个批次的图像和标签
            for image, labels in tqdm.tqdm(data_loader):
                image = image.to(device)  # 将图像数据移动到设备（GPU/CPU）
                results = img_classification_model(image)  # 获取模型的预测结果
                _, preds = torch.max(results, 1)  # 获取最大概率的类索引
                
                outputs.append(preds.cpu())  # 将预测结果移回 CPU，并保存到输出列表
        # 将所有批次的预测结果拼接成一个大张量
        self.result = torch.cat(outputs, dim=0)
    
    # 显示图像分类的预测结果
    def show_img1k_result(self):
        result = self.result.tolist()  # 将预测结果转换为列表
        # 将预测结果转换为对应的类标签
        result = [self.imagenet_1k_labels[class_idx] for class_idx in result]
        print(result)  # 打印最终的分类结果

    # 显示模型的结构
    def show_model(self):
        # 获取并打印预训练的 EfficientNet-B0 模型
        img_classification_model = get_model('efficientnet-b0_3rdparty_8xb32_in1k', pretrained=True).to(device)
        print(img_classification_model)

# 定义一个类来计算图像的 RGB 值（均值和标准差）
class RGBValue():
    # 初始化方法，接受图像数据集
    def __init__(self, img_data):
        # 存储传入的图像数据集
        self.img_dataset = img_data
        # 创建数据加载器，用于批量加载图像数据集
        data_loader = DataLoader(self.img_dataset, batch_size=32, shuffle=False)
        outputs = []  # 存储 RGB 值结果
        # 遍历数据加载器，获取每个批次的图像和标签
        for image, labels in tqdm.tqdm(data_loader):
            # 计算每个通道的均值
            mean_per_channel = torch.mean(image, dim=(2, 3))
            # 计算每个通道的标准差
            std_per_channel = torch.std(image, dim=(2, 3))
            # 将均值和标准差合并成一个张量
            rgb_values = torch.cat((mean_per_channel, std_per_channel), dim=1)
            outputs.append(rgb_values)  # 将结果添加到输出列表
        # 将所有批次的结果拼接成一个大张量
        self.result = torch.cat(outputs, dim=0)

# HSV值计算类
class HSVValue():
    # 初始化方法，接受图像数据集
    def __init__(self, img_data):
        # 存储传入的图像数据集
        self.img_dataset = img_data
        # 创建数据加载器，批量加载图像数据集
        data_loader = DataLoader(self.img_dataset, batch_size=32, shuffle=False)
        outputs = []  # 存储 HSV 值的结果

        # 遍历数据加载器中的每个批次
        for image, labels in tqdm.tqdm(data_loader):
            # 将图像转换为 numpy 数组并进行处理
            image = image.numpy()
            image = image * 255  # 将图像像素值从 [0, 1] 扩展到 [0, 255]
            image = image.astype(np.uint8)  # 将像素值转换为无符号整数
            image = np.transpose(image, (0, 2, 3, 1))  # 调整维度顺序为 (batch_size, height, width, channels)

            # 将图像转换为 HSV 色彩空间
            for i in range(len(image)):
                image[i] = cv2.cvtColor(image[i], cv2.COLOR_RGB2HSV)
                
            image = np.transpose(image, (0, 3, 1, 2))  # 将维度恢复为 (batch_size, channels, height, width)
            image = torch.tensor(image, dtype=torch.float32)  # 将图像数据转换为 PyTorch tensor

            # 计算每个通道的均值和标准差
            mean_per_channel = torch.mean(image, dim=(2, 3))
            std_per_channel = torch.std(image, dim=(2, 3))

            # 将均值和标准差拼接成一个 tensor
            hsv_values = torch.cat((mean_per_channel, std_per_channel), dim=1)
            outputs.append(hsv_values)  # 将结果添加到输出列表

        # 将所有批次的结果拼接成一个大的 tensor
        self.result = torch.cat(outputs, dim=0)

# 纹理特征计算类
class Texture():
    # 初始化方法，接受图像数据集
    def __init__(self, img_data):
        # 存储传入的图像数据集
        self.img_dataset = img_data
        # 创建数据加载器，批量加载图像数据集
        data_loader = DataLoader(self.img_dataset, batch_size=32, shuffle=False)
        outputs = []  # 存储纹理特征的结果

        # 遍历数据加载器中的每个批次
        for image, labels in tqdm.tqdm(data_loader):
            # 将图像转换为 numpy 数组并进行处理
            image = image.numpy()
            image = image * 255  # 将图像像素值从 [0, 1] 扩展到 [0, 255]
            image = image.astype(np.uint8)  # 将像素值转换为无符号整数
            image = np.transpose(image, (0, 2, 3, 1))  # 调整维度顺序为 (batch_size, height, width, channels)

            # 将图像转换为灰度图像并计算纹理特征
            for i in range(len(image)):
                temp_image = cv2.cvtColor(image[i], cv2.COLOR_RGB2GRAY)  # 转换为灰度图像
                # 计算灰度共生矩阵（GLCM）特征
                contrast, homogeneity, energy, entropy, correlation, ASM, dissimilarity = self.calculate_glcm_features(temp_image)
                textile_values = torch.tensor([contrast, homogeneity, energy, entropy, correlation, ASM, dissimilarity])
                outputs.append(textile_values)  # 将纹理特征添加到输出列表

        # 将所有批次的纹理特征拼接成一个大的 tensor
        self.result = torch.stack(outputs)

    # 计算灰度共生矩阵（GLCM）特征
    def calculate_glcm_features(self, image, distances=[1], angles=[0]):
        # 将灰度图像转换为 8-bit 的无符号字节格式
        gray_image = img_as_ubyte(image)
        # 计算灰度共生矩阵
        glcm = graycomatrix(gray_image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
        
        # 计算 GLCM 特征
        contrast = graycoprops(glcm, 'contrast')[0, 0]  # 对比度
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]  # 同质性
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]  # 不相似度
        ASM = graycoprops(glcm, 'ASM')[0, 0]  # 能量（Angular Second Moment）
        energy = graycoprops(glcm, 'energy')[0, 0]  # 能量
        entropy = -np.sum(glcm * np.log2(glcm + (glcm == 0)))  # 熵
        correlation = graycoprops(glcm, 'correlation')[0, 0]  # 相关性

        # 返回所有计算的特征
        return contrast, homogeneity, energy, entropy, correlation, ASM, dissimilarity

# 深度学习特征提取类
class ClassificationEmbedding():
    # 初始化方法，接受图像数据集
    def __init__(self, img_data):
        # 加载 ImageNet 1K 类别标签文件的路径
        self.imagenet_1k_labels_file = os.path.join(data_path, "imagenet-simple-labels.json")
        # 载入 ImageNet 1K 类别标签文件
        self.imagenet_1k_labels = json.load(open(self.imagenet_1k_labels_file))
        # 存储传入的图像数据集
        self.img_dataset = img_data
        # 创建数据加载器，批量加载图像数据集
        data_loader = DataLoader(self.img_dataset, batch_size=32, shuffle=False)
        # 加载预训练的 EfficientNet-B0 模型并设置为评估模式
        img_classification_model = get_model('efficientnet-b0_3rdparty_8xb32_in1k', pretrained=True).to(device)
        img_classification_model.eval()
        outputs = []  # 存储模型的输出

        # 禁用梯度计算，避免计算图的创建
        with torch.no_grad():
            # 遍历数据加载器中的每个批次
            for image, labels in tqdm.tqdm(data_loader):
                image = image.to(device)  # 将图像数据移动到设备（GPU/CPU）
                results = img_classification_model(image)  # 获取模型的预测结果
                outputs.append(results.cpu())  # 将结果移回 CPU 并保存

        # 将所有批次的预测结果拼接成一个大的 tensor
        self.result = torch.cat(outputs, dim=0)
    
    # 显示分类结果
    def show_img1k_result(self):
        result = self.result.tolist()  # 将预测结果转换为列表
        # 将预测结果转换为对应的类标签
        result = [self.imagenet_1k_labels[class_idx] for class_idx in result]
        print(result)  # 打印最终的分类结果

    # 显示模型结构
    def show_model(self):
        # 获取并打印预训练的 EfficientNet-B0 模型
        img_classification_model = get_model('efficientnet-b0_3rdparty_8xb32_in1k', pretrained=True).to(device)
        print(img_classification_model)

# 分割嵌入类，用于使用 FastSAM 模型进行图像分割
class SegmentationEmbedding():
    # 初始化方法，接受图像数据集
    def __init__(self, img_data):
        # 设置 FastSAM 模型路径并将其加入到 sys.path 中
        # https://github.com/CASIA-IVA-Lab/FastSAM
        sys.path[-3] = os.path.abspath(os.path.join(scripts_path, "models", "FastSAM"))
        
        # 从 fastsam 库导入 FastSAM 模型
        from fastsam import FastSAM
        import torch
        from PIL import Image
        
        # 恢复 sys.path 到原状态
        sys.path[-3] = ''
        
        # 存储传入的图像数据集
        self.img_dataset = img_data
        
        # 创建数据加载器，批量加载图像数据集
        data_loader = DataLoader(self.img_dataset, batch_size=32, shuffle=False)
        
        # 加载模型路径并初始化 FastSAM 模型
        model_path = os.path.join(scripts_path, 'models', 'FastSAM', 'weights', 'FastSAM-s.pt')
        self.segmentation_model = FastSAM(model_path)
        
        # 设置一些分割模型的参数
        retina_masks = True  # 使用视网膜掩膜
        imgsz = 224  # 图像尺寸
        conf = 0.4  # 置信度阈值
        iou = 0.9  # IoU 阈值

        outputs = []  # 存储分割结果

        # 在禁用梯度计算的环境中进行推理，避免计算图的创建
        with torch.no_grad():
            # 遍历数据加载器中的每个批次
            for image, labels in tqdm.tqdm(data_loader):
                image = image.numpy()  # 将图像从 Tensor 转换为 numpy 数组
                image = np.transpose(image, (0, 2, 3, 1))  # 转换图像维度顺序为 (batch_size, height, width, channels)
                image = torch.tensor(image)  # 将 numpy 数组转换为 PyTorch Tensor
                image = image.to(device)  # 将图像数据移动到设备（GPU/CPU）
                
                # 使用 FastSAM 模型进行图像分割
                results = self.segmentation_model(
                    image,
                    device=device,
                    retina_masks=retina_masks,
                    imgsz=imgsz,
                    conf=conf,
                    iou=iou
                )
                
                # 打印分割结果的框和掩膜（可选，调试时查看）
                print(results[0].boxes)
                print(results[0].masks)
                print(results[0].masks.shape)
                break  # 调试时仅处理第一批数据并停止

                # 将每个批次的分割结果添加到输出列表
                outputs.append(results)

        # 将所有批次的分割结果拼接成一个大 tensor
        self.result = torch.cat(outputs, dim=0)

    # 显示模型的结构
    def show_model(self):
        print(self.segmentation_model)

    # 后处理分割结果，将其掩膜进行整理
    def postprocess(self, results):
        # 提取每个结果的掩膜部分
        results = [result.masks for result in results]
        # 遍历每个结果的掩膜，确保掩膜的长度统一
        for i in range(len(results)):
            print(results[i].shape)  # 打印掩膜的形状（调试）
            results[i] = torch.tensor(results[i])  # 将掩膜转换为 PyTorch Tensor
            # 如果掩膜长度小于 5，则填充零
            while len(results[i]) < 5:
                results[i] = torch.cat([results[i], torch.zeros_like(results[i])], dim=0)
            # 如果掩膜长度大于 5，则裁剪到前 5 个
            if len(results[i]) > 5:
                results[i] = results[i][:5]
        return results

# 添加低级特征到数据中
def add_low_level_feature(data):
    # 构建图像数据集
    dataset = ConstructDataset(data)
    
    # 计算 RGB 特征（均值和标准差）
    rgb_value = RGBValue(dataset)
    rgb_df = pd.DataFrame(rgb_value.result.numpy(), 
                          columns=['red_mean', 'green_mean', 'blue_mean', 
                                   'red_std', 'green_std', 'blue_std'])
    
    # 计算 HSV 特征（均值和标准差）
    hsv_value = HSVValue(dataset)
    hsv_df = pd.DataFrame(hsv_value.result.numpy(), 
                          columns=['hue_mean', 'saturation_mean', 'value_mean', 
                                   'hue_std', 'saturation_std', 'value_std'])
    
    # 计算纹理特征（如对比度、同质性等）
    texture = Texture(dataset)
    texture_df = pd.DataFrame(texture.result.numpy(), 
                              columns=['contrast', 'homogeneity', 'energy', 
                                       'entropy', 'correlation', 'ASM', 'dissimilarity'])
    
    # 将原始数据和特征拼接在一起
    new_data = pd.concat([data, rgb_df, hsv_df, texture_df], axis=1)
    
    return new_data

# 添加深度学习特征到数据中
def add_deep_learning_feature(data):
    # 创建一个图像数据集对象
    dataset = ConstructDataset(data)
    
    # 使用分类嵌入模型提取深度学习特征
    classification_embedding = ClassificationEmbedding(dataset)
    # 将分类嵌入的结果添加到数据中
    data['classification_embedding'] = classification_embedding.result.numpy()
    
    # 使用图像分类模型提取分类特征
    classification = ImageClassification(dataset)
    # 将图像分类的结果添加到数据中
    data['classification'] = classification.result.numpy()
    
    # 使用分割嵌入模型提取图像分割特征
    segmentation_embedding = SegmentationEmbedding(dataset)
    # 将分割嵌入的结果添加到数据中
    data['segmentation_embedding'] = segmentation_embedding.result.numpy()
    
    # 返回包含深度学习特征的数据
    return data

# 保存分析结果的函数
def analysis_feature(data, data_path, analysis_name):
    # 设置结果保存路径
    imgs_path = os.path.join(data_path, analysis_name)
    
    # 如果 analysis_name 以 "smp_" 开头，去掉前缀
    if analysis_name[:4] in ["smp_"]:
        analysis_name = analysis_name[4:]
    
    # 创建数据副本并去除缺失值
    data_analysis = data.copy()
    data_analysis = data_analysis.dropna()
    data_analysis = data_analysis.reset_index(drop=True)

    significant_results = []  # 存储显著结果
    
    # Spearman 相关性分析
    spearman_corr_matrix = data_analysis.corr(method='spearman')
    # 创建存储 t 值和 p 值的矩阵
    t_matrix = pd.DataFrame(index=spearman_corr_matrix.index, columns=spearman_corr_matrix.columns)
    p_matrix = pd.DataFrame(index=spearman_corr_matrix.index, columns=spearman_corr_matrix.columns)
    num_samples = len(data_analysis)
    
    # 计算 Spearman 相关性以及 t 值和 p 值
    for col1 in spearman_corr_matrix.columns:
        for col2 in spearman_corr_matrix.columns:
            if col1 != col2:
                r = spearman_corr_matrix.loc[col1, col2]
                if abs(r) == 1:
                    t = np.inf
                else:
                    t = r * np.sqrt((num_samples-2) / (1-r**2))
                p = 2 * (1 - stats.t.cdf(abs(t), num_samples-2))
                t_matrix.loc[col1, col2] = t
                p_matrix.loc[col1, col2] = p
                # 如果 p 值小于等于 0.05 且相关的变量是 'label'，记录该结果
                if p <= 0.05 and col1 == 'label':
                    significant_results.append({
                        'Analysis': 'Spearman Correlation',
                        'Variable1': col1,
                        'Variable2': col2,
                        'Correlation Coefficient': r,
                        'P-value': p
                    })
    
    # 创建目录来保存分析图像
    if not os.path.exists(imgs_path):
        os.makedirs(imgs_path)
    
    # 处理矩阵数据，确保数据是数值型
    p_matrix = p_matrix.apply(pd.to_numeric, errors='coerce')
    t_matrix = t_matrix.apply(pd.to_numeric, errors='coerce')
    
    # 绘制 Spearman 相关性矩阵、t 值矩阵和 p 值矩阵的热图
    plt.figure(figsize=(90, 30))
    plt.subplot(1, 3, 1)
    sns.heatmap(spearman_corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('corr matrix', fontsize=16)
    plt.subplot(1, 3, 2)
    sns.heatmap(t_matrix, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title('t value matrix', fontsize=16)
    plt.subplot(1, 3, 3)
    sns.heatmap(p_matrix, annot=True, cmap='YlGnBu', fmt='.4f')
    plt.title('p value matrix', fontsize=16)
    # 保存结果为图像文件
    plt.savefig(os.path.join(imgs_path, f"spearman_correlation_analysis_{analysis_name}.png"))
    plt.close()
    
    # 将 Spearman 相关性矩阵和 p 值矩阵保存为 CSV 文件
    spearman_corr_matrix_path = os.path.join(imgs_path, f"spearman_corr_matrix_{analysis_name}.csv")
    spearman_corr_matrix.to_csv(spearman_corr_matrix_path, index=True)
    p_matrix_path = os.path.join(imgs_path, f"spearman_p_matrix_{analysis_name}.csv")
    p_matrix.to_csv(p_matrix_path, index=True)
    
    # 进行 Pearson 相关性分析
    pearson_corr_matrix = data_analysis.corr(method='pearson')
    t_matrix = pd.DataFrame(index=pearson_corr_matrix.index, columns=pearson_corr_matrix.columns)
    p_matrix = pd.DataFrame(index=pearson_corr_matrix.index, columns=pearson_corr_matrix.columns)
    num_samples = len(data_analysis)
    
    # 计算 Pearson 相关性以及 t 值和 p 值
    for col1 in pearson_corr_matrix.columns:
        for col2 in pearson_corr_matrix.columns:
            if col1 != col2:
                r = pearson_corr_matrix.loc[col1, col2]
                if abs(r) == 1:
                    t = np.inf
                else:
                    t = r * np.sqrt((num_samples-2) / (1-r**2))
                p = 2 * (1 - stats.t.cdf(abs(t), num_samples-2))
                t_matrix.loc[col1, col2] = t
                p_matrix.loc[col1, col2] = p
                if p <= 0.05 and col1 == 'label':
                    significant_results.append({
                        'Analysis': 'Pearson Correlation',
                        'Variable1': col1,
                        'Variable2': col2,
                        'Correlation Coefficient': r,
                        'P-value': p
                    })
    
    # 处理矩阵数据，确保数据是数值型
    p_matrix = p_matrix.apply(pd.to_numeric, errors='coerce')
    t_matrix = t_matrix.apply(pd.to_numeric, errors='coerce')
    
    # 绘制 Pearson 相关性矩阵、t 值矩阵和 p 值矩阵的热图
    plt.figure(figsize=(90, 30))
    plt.subplot(1, 3, 1)
    sns.heatmap(pearson_corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('corr matrix', fontsize=16)
    plt.subplot(1, 3, 2)
    sns.heatmap(t_matrix, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title('t value matrix', fontsize=16)
    plt.subplot(1, 3, 3)
    sns.heatmap(p_matrix, annot=True, cmap='YlGnBu', fmt='.4f')
    plt.title('p value matrix', fontsize=16)
    plt.savefig(os.path.join(imgs_path, f"pearson_correlation_analysis_{analysis_name}.png"))
    plt.close()
    
    # 保存 Pearson 相关性矩阵和 p 值矩阵为 CSV 文件
    pearson_corr_matrix_path = os.path.join(imgs_path, f"pearson_corr_matrix_{analysis_name}.csv")
    pearson_corr_matrix.to_csv(pearson_corr_matrix_path, index=True)
    p_matrix_path = os.path.join(imgs_path, f"pearson_p_matrix_{analysis_name}.csv")
    p_matrix.to_csv(p_matrix_path, index=True)
    
    # 进行 Kendall 相关性分析
    kendall_corr_matrix = data_analysis.corr(method='kendall')
    t_matrix = pd.DataFrame(index=kendall_corr_matrix.index, columns=kendall_corr_matrix.columns)
    p_matrix = pd.DataFrame(index=kendall_corr_matrix.index, columns=kendall_corr_matrix.columns)
    num_samples = len(data_analysis)
    
    # 计算 Kendall 相关性以及 t 值和 p 值
    for col1 in kendall_corr_matrix.columns:
        for col2 in kendall_corr_matrix.columns:
            if col1 != col2:
                r = kendall_corr_matrix.loc[col1, col2]
                if abs(r) == 1:
                    t = np.inf
                else:
                    t = r * np.sqrt((num_samples-2) / (1-r**2))
                p = 2 * (1 - stats.t.cdf(abs(t), num_samples-2))
                t_matrix.loc[col1, col2] = t
                p_matrix.loc[col1, col2] = p
                if p <= 0.05 and col1 == 'label':
                    significant_results.append({
                        'Analysis': 'Kendall Correlation',
                        'Variable1': col1,
                        'Variable2': col2,
                        'Correlation Coefficient': r,
                        'P-value': p
                    })
    
    # 处理矩阵数据，确保数据是数值型
    p_matrix = p_matrix.apply(pd.to_numeric, errors='coerce')
    t_matrix = t_matrix.apply(pd.to_numeric, errors='coerce')
    
    # 绘制 Kendall 相关性矩阵、t 值矩阵和 p 值矩阵的热图
    plt.figure(figsize=(90, 30))
    plt.subplot(1, 3, 1)
    sns.heatmap(kendall_corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('corr matrix', fontsize=16)
    plt.subplot(1, 3, 2)
    sns.heatmap(t_matrix, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title('t value matrix', fontsize=16)
    plt.subplot(1, 3, 3)
    sns.heatmap(p_matrix, annot=True, cmap='YlGnBu', fmt='.4f')
    plt.title('p value matrix', fontsize=16)
    plt.savefig(os.path.join(imgs_path, f"kendall_correlation_analysis_{analysis_name}.png"))
    plt.close()
    
    # 保存 Kendall 相关性矩阵和 p 值矩阵为 CSV 文件
    kendall_corr_matrix_path = os.path.join(imgs_path, f"kendall_corr_matrix_{analysis_name}.csv")
    kendall_corr_matrix.to_csv(kendall_corr_matrix_path, index=True)
    p_matrix_path = os.path.join(imgs_path, f"kendall_p_matrix_{analysis_name}.csv")
    p_matrix.to_csv(p_matrix_path, index=True)

    # ---------------------------------------
    # 线性回归分析
    # ---------------------------------------

    # 创建特征和目标变量的副本，避免修改原始数据
    X_linear = data_analysis.drop('label', axis=1).copy()
    y_linear = np.exp(data_analysis['label'])  # 对目标变量 'label' 应用指数转换

    # 给预测变量添加常数项
    X_linear = sm.add_constant(X_linear)

    # 将数据拆分为训练集和测试集
    X_train_linear, X_test_linear, y_train_linear, y_test_linear = train_test_split(
        X_linear, y_linear, test_size=0.2, random_state=42)

    # 构建线性回归模型
    linear_model = sm.OLS(y_train_linear, X_train_linear)
    linear_results = linear_model.fit()

    # 保存线性回归结果到文本文件
    with open(os.path.join(imgs_path, f"linear_regression_analysis_{analysis_name}.txt"), "w") as f:
        f.write(linear_results.summary().as_text())
        coefficients = linear_results.params
        p_values = linear_results.pvalues
        t_values = linear_results.tvalues
        results_df = pd.DataFrame({
            'Coefficient': coefficients,
            'P-value': p_values,
            'T-value': t_values
        })
        f.write("\nCoefficient, P-value, and T-value:\n")
        f.write(results_df.to_string())
        y_pred_linear = linear_results.predict(X_test_linear)
        mse_linear = mean_squared_error(y_test_linear, y_pred_linear)
        r2_linear = r2_score(y_test_linear, y_pred_linear)
        f.write(f"\nMean Squared Error (MSE): {mse_linear}")
        f.write(f"\nR-squared: {r2_linear}")
        # 构建回归方程
        equation = "y = "
        for i, column in enumerate(X_linear.columns):
            coef = coefficients.iloc[i]
            sign = " + " if coef >= 0 else " - "
            term = f"{abs(coef):.4f} * {column}" if column != 'const' else f"{coef:.4f}"
            equation += sign + term if i > 0 else term
        f.write(f"\nLinear Regression Equation:\n")
        f.write(equation)

    for index, row in results_df.iterrows():
        if index != 'const' and row['P-value'] <= 0.05:
            significant_results.append({
                'Analysis': 'Linear Regression',
                'Variable': index,
                'Coefficient': row['Coefficient'],
                'P-value': row['P-value'],
                'T-value': row['T-value']
            })

    # 绘制回归系数和 p 值的热图
    plt.figure(figsize=(30, 15))

    # 绘制回归系数的热图
    plt.subplot(1, 2, 1)
    sns.heatmap(results_df[['Coefficient']].sort_values(by='Coefficient', ascending=False),
                annot=True, cmap='coolwarm', center=0)
    plt.title('Coefficient')

    # 绘制 p 值的热图
    plt.subplot(1, 2, 2)
    sns.heatmap(results_df[['P-value']].sort_values(by='P-value'),
                annot=True, cmap='YlOrRd')
    plt.title('P-value')

    plt.tight_layout()
    plt.savefig(os.path.join(imgs_path, f"linear_regression_analysis_plots_{analysis_name}.png"))
    plt.close()

    # ---------------------------------------
    # 泊松回归分析
    # ---------------------------------------

    # 创建特征和目标变量的副本，避免修改原始数据
    X_poisson = data_analysis.drop('label', axis=1).copy()
    y_poisson = np.exp(data_analysis['label'])  # 对目标变量 'label' 应用指数转换

    # 将目标变量转换为非负整数（泊松回归要求）
    y_poisson = y_poisson.round().astype(int)

    # 检查目标变量中是否包含负值
    if (y_poisson < 0).any():
        raise ValueError("The target variable contains negative values, which are invalid for Poisson regression.")

    # 给预测变量添加常数项
    X_poisson = sm.add_constant(X_poisson)

    # 将数据拆分为训练集和测试集
    X_train_poisson, X_test_poisson, y_train_poisson, y_test_poisson = train_test_split(
        X_poisson, y_poisson, test_size=0.2, random_state=42)

    # 使用广义线性模型（GLM）构建泊松回归模型
    poisson_model = sm.GLM(y_train_poisson, X_train_poisson, family=sm.families.Poisson())
    poisson_results = poisson_model.fit()

    # 保存泊松回归结果到文本文件
    with open(os.path.join(imgs_path, f"poisson_regression_analysis_{analysis_name}.txt"), "w") as f:
        f.write(poisson_results.summary().as_text())
        coefficients = poisson_results.params
        p_values = poisson_results.pvalues
        t_values = poisson_results.tvalues
        results_df = pd.DataFrame({
            'Coefficient': coefficients,
            'P-value': p_values,
            'T-value': t_values
        })
        f.write("\nCoefficient, P-value, and T-value:\n")
        f.write(results_df.to_string())
        y_pred_poisson = poisson_results.predict(X_test_poisson)
        mse_poisson = mean_squared_error(y_test_poisson, y_pred_poisson)
        r2_poisson = r2_score(y_test_poisson, y_pred_poisson)
        f.write(f"\nMean Squared Error (MSE): {mse_poisson}")
        f.write(f"\nR-squared: {r2_poisson}")
        # 构建回归方程
        equation = "y = "
        for i, column in enumerate(X_poisson.columns):
            coef = coefficients.iloc[i]
            sign = " + " if coef >= 0 else " - "
            term = f"{abs(coef):.4f} * {column}" if column != 'const' else f"{coef:.4f}"
            equation += sign + term if i > 0 else term
        f.write(f"\nPoisson Regression Equation:\n")
        f.write(equation)

    for index, row in results_df.iterrows():
        if index != 'const' and row['P-value'] <= 0.05:
            significant_results.append({
                'Analysis': 'Poisson Regression',
                'Variable': index,
                'Coefficient': row['Coefficient'],
                'P-value': row['P-value'],
                'T-value': row['T-value']
            })

    # 绘制回归系数和 p 值的热图
    plt.figure(figsize=(30, 15))

    # 绘制回归系数的热图
    plt.subplot(1, 2, 1)
    sns.heatmap(results_df[['Coefficient']].sort_values(by='Coefficient', ascending=False),
                annot=True, cmap='coolwarm', center=0)
    plt.title('Coefficient')

    # 绘制 p 值的热图
    plt.subplot(1, 2, 2)
    sns.heatmap(results_df[['P-value']].sort_values(by='P-value'),
                annot=True, cmap='YlOrRd')
    plt.title('P-value')

    plt.tight_layout()
    plt.savefig(os.path.join(imgs_path, f"poisson_regression_analysis_plots_{analysis_name}.png"))
    plt.close()

    # 卡方检验和 t 检验
    chi2_results = {}
    t_results = {}
    X = data_analysis.drop('label', axis=1)
    y = data_analysis['label']
    for column in X.columns:
        # 卡方检验
        contingency_table = pd.crosstab(y, X[column])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        if p <= 0.05:
            significant_results.append({
                'Analysis': 'Chi-square Test',
                'Variable': column,
                'Chi2': chi2,
                'P-value': p,
                'Degrees of Freedom': dof
            })
        chi2_results[column] = {'chi2': chi2, 'p': p, 'dof': dof}
        # t 检验
        t_stat, p_value = ttest_ind(y, X[column])
        t_results[column] = {'t_stat': t_stat, 'p_value': p_value}
        if p_value <= 0.05:
            significant_results.append({
                'Analysis': 't-test',
                'Variable': column,
                't_stat': t_stat,
                'P-value': p_value
            })
    # 保存卡方检验和 t 检验结果
    with open(os.path.join(imgs_path, f"chi2_test_results_{analysis_name}.txt"), "w") as f:
        f.write(str(chi2_results))
    with open(os.path.join(imgs_path, f"t_test_results_{analysis_name}.txt"), "w") as f:
        f.write(str(t_results))

    return significant_results

def get_smp_data(scripts_path, data_path, configs):
    # 获取数据目录的绝对路径，指定数据源
    smp_data_dir = os.path.abspath(os.path.join(scripts_path, "..", "data", "smp_2019"))

    # 加载图像路径文件
    with open(os.path.join(smp_data_dir, "train_img_filepath.txt")) as f:
        smp_image_path_files = f.readlines()
        # 处理路径，去掉"train/"并拼接完整路径
        smp_image_path_files = [os.path.join(smp_data_dir, path.replace("train/", "").strip()) 
                                for path in smp_image_path_files]

    # 加载标签文件
    with open(os.path.join(smp_data_dir, "train_label.txt")) as f:
        smp_labels = [float(label.strip()) for label in f]

    # 加载分类信息
    with open(os.path.join(data_path, "smp_2019", "train_category.json")) as f:
        smp_category_dict = json.load(f)
        smp_category = [item['Category'] for item in smp_category_dict]  # 主类别
        smp_subcategory = [item['Subcategory'] for item in smp_category_dict]  # 子类别
        smp_concept = [item['Concept'] for item in smp_category_dict]  # 概念

    # 合并类别和子类别信息
    smp_category_n_subcategory = [f"{cat}:{subcat}" for cat, subcat in zip(smp_category, smp_subcategory)]

    # 计算每个类别出现的次数
    smp_subcategory_2_counter = Counter(smp_category_n_subcategory)

    # 加载额外信息（如媒体状态、路径别名、是否公开等）
    with open(os.path.join(smp_data_dir, "train_additional_information.json")) as f:
        smp_additional_info = json.load(f)
        smp_Mediastatus = [info['Mediastatus'] for info in smp_additional_info]
        smp_Pathalias = [info['Pathalias'] for info in smp_additional_info]
        smp_Ispublic = [info['Ispublic'] for info in smp_additional_info]
        smp_Pid = [info['Pid'] for info in smp_additional_info]
        smp_Uid = [info['Uid'] for info in smp_additional_info]

    # 加载时间和空间信息（如发布日期、经度、纬度等）
    with open(os.path.join(smp_data_dir, "train_temporalspatial_information.json")) as f:
        smp_temporalspatial_information = json.load(f)
        smp_Postdate = [info['Postdate'] for info in smp_temporalspatial_information]
        smp_Longitude = [info['Longitude'] for info in smp_temporalspatial_information]
        smp_Geoaccuracy = [info['Geoaccuracy'] for info in smp_temporalspatial_information]
        smp_Latitude = [info['Latitude'] for info in smp_temporalspatial_information]

    # 加载用户数据（如用户描述、照片数量、时区等）
    with open(os.path.join(smp_data_dir, "train_user_data.json")) as f:
        smp_user_data = json.load(f)
        n = len(smp_user_data['Pid'])
        smp_photo_firstdate = [smp_user_data['photo_firstdate'][str(i)] for i in range(n)]
        smp_photo_count = [smp_user_data['photo_count'][str(i)] for i in range(n)]
        smp_ispro = [smp_user_data['ispro'][str(i)] for i in range(n)]
        smp_timezone_offset = [smp_user_data['timezone_offset'][str(i)] for i in range(n)]
        smp_photo_firstdatetaken = [smp_user_data['photo_firstdatetaken'][str(i)] for i in range(n)]
        smp_timezone_id = [smp_user_data['timezone_id'][str(i)] for i in range(n)]
        smp_user_description = [smp_user_data['user_description'][str(i)] for i in range(n)]
        smp_location_description = [smp_user_data['location_description'][str(i)] for i in range(n)]

    # 加载文本数据（如所有标签、媒体类型、标题等）
    with open(os.path.join(smp_data_dir, "train_text.json")) as f:
        smp_text_data = json.load(f)
        smp_all_tags = [item["Alltags"] for item in smp_text_data]
        smp_media_type = [item["Mediatype"] for item in smp_text_data]
        smp_title = [item["Title"] for item in smp_text_data]

    # 创建包含所有信息的 DataFrame
    smp_data = pd.DataFrame({
        "image_path": smp_image_path_files,
        "label": smp_labels,
        "category": smp_category,
        "subcategory": smp_subcategory,
        "Mediastatus": smp_Mediastatus,
        "Pathalias": smp_Pathalias,
        "Ispublic": smp_Ispublic,
        "Pid": smp_Pid,
        "Uid": smp_Uid,
        "Postdate": smp_Postdate,
        "Longitude": smp_Longitude,
        "Geoaccuracy": smp_Geoaccuracy,
        "Latitude": smp_Latitude,
        "photo_firstdate": smp_photo_firstdate,
        "photo_count": smp_photo_count,
        "ispro": smp_ispro,
        "timezone_offset": smp_timezone_offset,
        "photo_firstdatetaken": smp_photo_firstdatetaken,
        "timezone_id": smp_timezone_id,
        "user_description": smp_user_description,
        "location_description": smp_location_description,
        "concept": smp_concept,
        "all_tags": smp_all_tags,
        "media_type": smp_media_type,
        "title": smp_title,
    })

    # 处理 location_description 字段，保留逗号后面的最后一部分
    smp_data["location_description"] = smp_data["location_description"].apply(leave_str_only_last_part_behind_comma)

    # 如果配置要求进行下采样，随机选择部分图像数据
    if configs.smp_subset_test:
        if not configs.smp_img_random:
            random.seed(42)
        random_images_indices = random.sample(range(len(smp_image_path_files)), 
                                            min(len(smp_image_path_files), configs.smp_img_num))
        random_images_indices = [idx for idx in random_images_indices 
                                if os.path.getsize(smp_image_path_files[idx]) > 0]
        smp_data = smp_data.iloc[random_images_indices].reset_index(drop=True)

    # 打印数据列信息
    print(smp_data.columns)

    return smp_data

def data_normalization(data, column_name):
    # 复制原始数据，避免修改原始数据
    data_analysis = data.copy()

    # 只保留 'photo' 类型的媒体数据
    data_analysis = data_analysis[data_analysis['media_type'] == 'photo']

    # 删除与分析无关的列（如路径、用户信息、地理位置、类别等）
    for column in ['image_path', 'Pid', 'Uid', 'category', 'subcategory', 'Mediastatus', 'Pathalias', 'user_description', 'location_description', 'concept', 'all_tags', 'media_type', 'title', 'timezone_id', 'Longitude', 'photo_firstdate', 'photo_firstdatetaken']:
        if column in data_analysis.columns:  # 确保列存在
            data_analysis = data_analysis.drop(columns=[column])  # 删除该列

    # 保存初始的行数，用于之后比较
    initial_line_count = len(data_analysis)

    # 检查每列的缺失值数量（注释掉了这部分，避免冗余输出）
    # for column in data_analysis.columns:
    #     print(f"Missing values in {column} column: {data_analysis[column].isna().sum()}")

    # 删除包含 NaN 值的行
    data_analysis = data_analysis.dropna()
    data_analysis = data_analysis.reset_index(drop=True)  # 重置索引，删除之前的索引信息

    # 将 'timezone_offset' 列从字符串格式转换为分钟数
    def time_str_to_minutes(time_str):
        # 将时区偏移字符串转换为时间对象
        time_obj = datetime.strptime(time_str, "%z")
        # 获取时区偏移量，并计算总分钟数
        offset = time_obj.utcoffset()
        total_minutes = int(offset.total_seconds() / 60)
        return total_minutes

    # 将 'timezone_offset' 列中的时区信息转换为分钟数
    data_analysis['timezone_offset'] = data_analysis['timezone_offset'].apply(time_str_to_minutes)

    # 删除 'timezone_offset' 列中仍为 NaN 的行
    data_analysis = data_analysis.dropna(subset=['timezone_offset'])

    # 注释掉打印列值的部分，避免冗余
    # for column in data_analysis.columns:
    #     print(column, data_analysis[column])

    # 再次删除所有缺失值的行
    data_analysis = data_analysis.dropna()

    # 去除异常值（基于四分位距法，IQR）
    def remove_outliers_all_columns(df, exclude_columns=['label'], factor=1.5):
        df = df.dropna()  # 先去除 NaN 值
        df = df.reset_index(drop=True)  # 重置索引
        df_cleaned = df.copy()
        # 遍历所有列，排除标签列
        columns_to_process = [col for col in df.columns if col not in exclude_columns]
        for column in columns_to_process:
            if np.issubdtype(df[column].dtype, np.number):  # 如果是数值型列
                # 计算四分位数（Q1 和 Q3）
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1  # 四分位距
                lower_bound = Q1 - factor * IQR  # 下界
                upper_bound = Q3 + factor * IQR  # 上界
                # 将异常值（超出上下界的值）设为 NaN
                df_cleaned.loc[(df_cleaned[column] < lower_bound) | (df_cleaned[column] > upper_bound), column] = np.nan
        df_cleaned = df_cleaned.dropna()  # 删除 NaN 值
        return df_cleaned

    # 数据归一化的标准化方法
    def sklearn_normalize(df, method='minmax', exclude_columns=['label']):
        df_normalized = df.copy()  # 复制数据
        # 获取需要进行归一化的列，排除标签列
        columns_to_normalize = [col for col in df.columns if col not in exclude_columns]
        # 根据不同的归一化方法进行选择
        if method == 'minmax':
            scaler = MinMaxScaler()  # 使用最小-最大归一化
        elif method == 'zscore':
            scaler = StandardScaler()  # 使用 z-score 标准化
        elif method == 'maxabs':
            scaler = MaxAbsScaler()  # 使用最大绝对值标准化
        else:
            raise ValueError("Unsupported method. Choose 'minmax', 'zscore', or 'maxabs'.")
        
        # 对数据进行归一化处理
        df_normalized[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
        return df_normalized

    # 去除异常值并进行数据归一化
    data_analysis = remove_outliers_all_columns(data_analysis)
    data_analysis = sklearn_normalize(data_analysis, method='zscore')

    # 确保所有列可以转换为浮动类型
    for column in data_analysis.columns:
        if not np.issubdtype(data_analysis[column].dtype, np.number):
            data_analysis[column] = pd.to_numeric(data_analysis[column], errors='coerce')

    # 返回处理后的数据
    return data_analysis

def show_subset_of_data(data, column_name, filter):
    # 从数据中筛选出符合过滤条件的子集
    subset_data = data[data[column_name] == filter]
    # 打印显示筛选后的子集数据的列名和前五行数据
    print(column_name, "show:")
    print(subset_data.head())
    # 打印该子集的行数（数据量）
    print(f"\nTravel data volume: {len(subset_data)}")

    # 随机选择 20 个样本的索引
    random_indices = random.sample(range(len(subset_data)), 20)
    
    # 创建 2 行 10 列的子图（显示图片的布局）
    fig, axes = plt.subplots(2, 10, figsize=(15, 5))
    
    # 遍历随机选择的 20 个样本
    for i, idx in enumerate(random_indices):
        # 获取当前图像的路径
        img_path = subset_data.iloc[idx]['image_path']
        
        # 打开图像文件
        img = Image.open(img_path)
        
        # 计算当前图片在子图中的行和列位置
        row = i // 10
        col = i % 10
        
        # 在对应的子图位置显示图片
        axes[row, col].imshow(img)
        # 关闭坐标轴显示
        axes[row, col].axis('off')
        # 设置子图的标题
        axes[row, col].set_title(f"travel images {i+1}")

    # 调整子图的间距，使其不会重叠
    plt.tight_layout()
    # 显示绘制的图形
    plt.show()

def split_data(input, column_name, method="one-hot", intervals_num=None, other=0.01):
    # 复制输入数据，避免修改原始数据
    data = input.copy()
    
    # 方法 1: one-hot 编码方式，将数据按指定列的不同值进行分割
    if method == "one-hot":
        # 获取该列中所有唯一值
        unique_values = data[column_name].unique()
        # 按每个唯一值将数据分割成多个 DataFrame
        split_dataframes = [data[data[column_name] == value].reset_index(drop=True) for value in unique_values]
        
        # 如果 'other' 参数大于 1，抛出异常，因为它应该小于等于 1
        if other > 1:
            raise ValueError("one_hot_thr must be less than or equal to 1.")
        
        # 过滤掉行数较少的 DataFrame，只保留那些长度大于总数据集的 'other' 百分比的子集
        split_dataframes = [split_dataframe for split_dataframe in split_dataframes if len(split_dataframe) >= len(data)*other]
    
    # 方法 2: 基于分位数（quantile）进行数据分割
    elif method == "quantile":
        # 将指定列转换为数值型数据（以确保没有非数值型数据）
        data[column_name] = pd.to_numeric(data[column_name], errors='coerce')
        # 按指定列对数据进行排序
        sorted_data = data.sort_values(by=column_name).reset_index(drop=True)
        
        # 计算 0.05 和 0.95 分位数
        lower_quantile = sorted_data[column_name].quantile(0.05)
        upper_quantile = sorted_data[column_name].quantile(0.95)
        
        # 筛选出位于这两个分位数之间的数据
        filtered_data = sorted_data[(sorted_data[column_name] >= lower_quantile) & (sorted_data[column_name] <= upper_quantile)]
        
        # 如果没有指定分割的区间数或区间数小于等于 0，抛出异常
        if intervals_num is None or intervals_num <= 0:
            raise ValueError("intervals_num must be a positive integer for quantile method.")
        
        # 将筛选后的数据分割成指定数量的区间
        split_dataframes = np.array_split(filtered_data, intervals_num)
        
        # 重置每个分割子集的索引
        split_dataframes = [df.reset_index(drop=True) for df in split_dataframes]
    
    # 方法 3: 根据文本中的词汇数量进行分割
    elif method == "texts-word-count":
        # 替换列中所有的 None 为一个空字符串
        data[column_name] = data[column_name].fillna("")
        # 将该列文本的单词数作为新的列数据
        data[column_name] = data[column_name].apply(lambda x: len(x.split()))
        
        # 按照文本的单词数进行排序
        sorted_data = data.sort_values(by=column_name).reset_index(drop=True)
        
        # 计算 0.05 和 0.95 分位数
        lower_quantile = sorted_data[column_name].quantile(0.05)
        upper_quantile = sorted_data[column_name].quantile(0.95)
        
        # 筛选出位于这两个分位数之间的数据
        filtered_data = sorted_data[(sorted_data[column_name] >= lower_quantile) & (sorted_data[column_name] <= upper_quantile)]
        
        # 如果没有指定分割的区间数或区间数小于等于 0，抛出异常
        if intervals_num is None or intervals_num <= 0:
            raise ValueError("intervals_num must be a positive integer for quantile method.")
        
        # 将筛选后的数据分割成指定数量的区间
        split_dataframes = np.array_split(filtered_data, intervals_num)
        
        # 重置每个分割子集的索引
        split_dataframes = [df.reset_index(drop=True) for df in split_dataframes]
    else:
        # 如果方法无效，抛出异常
        raise ValueError(f"Invalid method: {method}. Supported methods are 'one-hot', 'quantile', 'texts-word-count'.")

    # 获取每个子集的第一行数据
    first_rows = [df.iloc[0] for df in split_dataframes]
    # 合并第一行数据并重置索引
    first_rows = pd.concat(first_rows, axis=1).T.reset_index(drop=True)
    
    # 返回第一行数据和每个子集的 DataFrame 列表
    return first_rows, split_dataframes

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

def load_workspace_variables(filename):
    """从指定文件加载工作区变量并更新到全局命名空间"""
    with open(filename, "rb") as file:
        # 使用 dill 库反序列化对象并加载
        workspace_variables = dill.load(file)
    # 将加载的工作区变量更新到全局命名空间
    globals().update(workspace_variables)

def analysis_features_using_configs(smp_data, config, data_path):
    """
    根据配置分析特征，支持不同的数据切割方法，并执行相关的统计分析

    参数:
    smp_data -- 数据集
    config -- 配置字典，包括切割方法和相关参数
    data_path -- 数据路径，用于存储分析结果

    返回:
    valid_records -- 存储分析结果的列表
    """
    valid_records = []  # 初始化有效记录列表
    
    if config["column_name"] == "total":
        # 如果配置的列名为 'total'，则进行全量数据的标准化分析
        smp_data_analysis = data_normalization(data=smp_data, column_name=config["column_name"])
        valid_records += analysis_feature(smp_data_analysis, data_path, "smp_total")
    else:
        # 根据配置的切割方法（如 "one-hot"），对数据进行分割
        smp_splited_data_first_rows, smp_splited_data = split_data(input=smp_data, **config)
        
        # 对分割后的数据进行标准化处理，并保存标准化后的数据
        smp_splited_normalized_data = [
            (data[config["column_name"]], data_normalization(data, config["column_name"] + "|" + str(data[config["column_name"]].iloc[0]))) 
            for data in smp_splited_data
        ]
        
        # 进行 Kolmogorov-Smirnov (KS) 测试
        valid_records += ks_test_analysis(smp_splited_normalized_data, data_path, analysis_column='label', split_column_name=config["column_name"])
        
        # 对每个分割后的数据集进行特征分析
        for column, data in tqdm.tqdm(smp_splited_normalized_data):
            save_name = "smp_" + config["column_name"] + "_" + str(column.iloc[0])
            valid_records += analysis_feature(data, data_path, save_name)
    
    return valid_records

def smp_analysis_texts(smp_data, most_common_num=20):
    """
    分析文本数据（如 'user_description', 'all_tags', 'title'）中的常见词汇

    参数:
    smp_data -- 数据集
    most_common_num -- 要输出的最常见词汇数量

    返回:
    user_cter, tags_cter, tit_cter -- 分别为用户描述、标签、标题的词频计数结果
    """
    user_cter = Counter()  # 用于存储用户描述的词频
    for user in smp_data["user_description"]:
        if user is not None:
            user_cter.update(user.lower().split())  # 将用户描述文本转换为小写，并统计词频
    
    # 删除常见的无意义词汇
    user_cter["and"] = user_cter["I"] = user_cter["the"] = user_cter["href"] = user_cter["of"] = \
        user_cter["<a"] = user_cter["to"] = user_cter["in"] = user_cter["rel"] = user_cter["nofollow"] = \
        user_cter["class"] = user_cter["on"] = user_cter["for"] = user_cter["is"] = user_cter["-"] = \
        user_cter["m"] = user_cter["><img"] = user_cter["alt="] = user_cter["with"] = user_cter["/></a>"] = \
        user_cter["at"] = user_cter["are"] = user_cter["or"] = user_cter["as"] = user_cter["width="] = \
        user_cter["that"] = user_cter["height="] = user_cter["by"] = user_cter["s"] = user_cter["be"] = \
        user_cter["from"] = user_cter["title="] = user_cter["an"] = user_cter["&amp;"] = user_cter["am"] = \
        user_cter["if"] = user_cter["this"] = user_cter["de"] = user_cter["do"] = user_cter["will"] = \
        user_cter["src="] = user_cter["href="] = user_cter["a"] = user_cter["rel="] = user_cter["class="] = 0
    
    print("user_cter:", user_cter.most_common(most_common_num))  # 打印出用户描述中最常见的词汇

    tags_cter = Counter()  # 用于存储标签文本的词频
    for tag in smp_data["all_tags"]:
        if tag is not None:
            tags_cter.update(tag.lower().split())  # 统计标签文本中的词频
    tags_cter['??'] = tags_cter['2015'] = tags_cter['????'] = 0
    print("tags_cter:", tags_cter.most_common(most_common_num))  # 打印出标签文本中最常见的词汇

    tit_cter = Counter()  # 用于存储标题文本的词频
    for title in smp_data["title"]:
        if title is not None:
            tit_cter.update(title.lower().split())  # 统计标题文本中的词频
    tit_cter['-'] = tit_cter['the'] = tit_cter['2015'] = tit_cter['of'] = tit_cter['and'] = tit_cter['in'] = \
        tit_cter['&'] = tit_cter['2016'] = tit_cter['at'] = tit_cter['to'] = tit_cter['on'] = tit_cter['de'] = \
        tit_cter['|'] = tit_cter['with'] = tit_cter['?'] = tit_cter['????'] = tit_cter['1'] = tit_cter['a'] = \
        tit_cter['@'] = tit_cter['2'] = tit_cter['from'] = tit_cter['is'] = tit_cter['la'] = tit_cter['/'] = 0
    print("tit_cter:", tit_cter.most_common(most_common_num))  # 打印出标题文本中最常见的词汇

    return user_cter.most_common(most_common_num), tags_cter.most_common(most_common_num), \
        tit_cter.most_common(most_common_num)

def smp_analysis_texts_existence(smp_data, data_path, user_cter, tags_cter, tit_cter):
    """
    根据用户描述、标签、标题中的常见词汇，分析每个文本字段是否包含这些词汇

    参数:
    smp_data -- 数据集
    data_path -- 数据路径，用于存储分析结果
    user_cter, tags_cter, tit_cter -- 用户描述、标签和标题的词频计数结果

    返回:
    None -- 该函数主要是执行分析操作并输出结果
    """
    # 对每个用户描述中的常见词汇进行分析
    for user_text in tqdm.tqdm(user_cter):
        text = user_text[0]
        # 在数据集的 'user_description' 列中检查是否包含该词汇
        smp_user_text_data = pd.concat([smp_data.copy(), smp_data["user_description"].apply(
            lambda x: text in x if x else False
        ).to_frame("user_description_" + text)], axis=1)
        config = {
            "column_name": "user_description_" + text,
            "method": "one-hot",
            "intervals_num": None,
            "other": 0.01,
        }
        # 使用配置分析特征
        analysis_features_using_configs(smp_user_text_data, config, data_path)
    
    # 对每个标签中的常见词汇进行分析
    for tag_text in tqdm.tqdm(tags_cter):
        text = tag_text[0]
        smp_tag_text_data = pd.concat([smp_data.copy(), smp_data["all_tags"].apply(
            lambda x: text in x if x else False
        ).to_frame("all_tags_" + text)], axis=1)
        config = {
            "column_name": "all_tags_" + text,
            "method": "one-hot",
            "intervals_num": None,
            "other": 0.01,
        }
        analysis_features_using_configs(smp_tag_text_data, config, data_path)
    
    # 对每个标题中的常见词汇进行分析
    for user_text in tqdm.tqdm(tit_cter):
        text = user_text[0]
        smp_title_text_data = pd.concat([smp_data.copy(), smp_data["title"].apply(
            lambda x: text in x if x else False
        ).to_frame("title_" + text)], axis=1)
        config = {
            "column_name": "title_" + text,
            "method": "one-hot",
            "intervals_num": None,
            "other": 0.01,
        }
        analysis_features_using_configs(smp_title_text_data, config, data_path)

def ks_test_analysis(splitted_data, data_path, analysis_column, split_column_name):
    """
    进行 Kolmogorov-Smirnov (KS) 检验，比较分割后的多个数据集之间的差异性。
    
    参数:
    splitted_data -- 分割后的数据，包含多个 DataFrame，每个 DataFrame 包含相同列名，但在指定的列上有不同的取值。
    data_path -- 数据存储路径，用于保存 KS 检验结果。
    analysis_column -- 用于分析的列名，通常为目标列，例如标签列。
    split_column_name -- 用于分割数据的列名。
    
    返回:
    significant_results -- 包含显著性分析结果的列表，记录了每组数据的 KS 检验统计量和 p-value。
    """
    
    # 从分割后的数据中提取列名和数据框列表
    columns = [data[0] for data in splitted_data]  # 每个 DataFrame 的列名
    df_list = [data[1] for data in splitted_data]  # 每个 DataFrame
    
    # 初始化 KS-stat 和 p-value 矩阵，存储不同组之间的检验统计量和 p-value
    num_dfs = len(df_list)
    ks_stat_matrix = np.zeros((num_dfs, num_dfs))  # 存储 KS 统计量
    p_value_matrix = np.zeros((num_dfs, num_dfs))  # 存储 p-value
    
    significant_results = []  # 用于保存显著的结果
    
    # 计算两两 DataFrame 的 KS-stat 和 p-value
    for i in range(num_dfs):
        for j in range(i, num_dfs):
            if i == j:
                # 当比较自身时，随机将数据分成两部分
                label_column = df_list[i][analysis_column]
                split_index = random.sample(range(len(label_column)), len(label_column) // 2)  # 随机选择一半的数据
                part1 = label_column.iloc[split_index]  # 第一个部分
                part2 = label_column.drop(label_column.index[split_index])  # 第二个部分
                ks_stat, p_value = ks_2samp(part1, part2)  # 进行 KS 检验
            else:
                # 比较不同 DataFrame 之间的 KS 检验
                a_label = df_list[i][analysis_column]
                b_label = df_list[j][analysis_column]
                ks_stat, p_value = ks_2samp(a_label, b_label)  # 进行 KS 检验

            # 填充矩阵
            ks_stat_matrix[i, j] = ks_stat
            ks_stat_matrix[j, i] = ks_stat
            p_value_matrix[i, j] = p_value
            p_value_matrix[j, i] = p_value

            # 如果 p-value <= 0.05，则认为该对比结果显著
            if i <= j and p_value <= 0.05:
                # 构建可读的组名
                group_i_name = f"{split_column_name}_{columns[i].iloc[0]}" if hasattr(columns[i], 'iloc') else str(columns[i])
                group_j_name = f"{split_column_name}_{columns[j].iloc[0]}" if hasattr(columns[j], 'iloc') else str(columns[j])
                
                # 将显著的结果记录到列表中
                significant_results.append({
                    "Group1": group_i_name,
                    "Group2": group_j_name,
                    "KS-stat": ks_stat,
                    "P-value": p_value
                })

    # 将 KS-stat 和 p-value 矩阵转化为 DataFrame 格式，便于可视化和保存
    ks_stat_df = pd.DataFrame(ks_stat_matrix, columns=[split_column_name + "_" + \
        str(column.iloc[0]) for column in columns], index=[split_column_name + "_" + \
        str(column.iloc[0]) for column in columns])
    
    p_value_df = pd.DataFrame(p_value_matrix, columns=[split_column_name + "_" + \
        str(column.iloc[0]) for column in columns], index=[split_column_name + "_" + \
        str(column.iloc[0]) for column in columns])

    # 画 KS-stat 和 p-value 的热力图
    plt.figure(figsize=(24, 10))

    # KS-stat 矩阵的热力图
    plt.subplot(1, 2, 1)
    sns.heatmap(ks_stat_df, annot=True, cmap='Blues', square=True, cbar=True)
    plt.title('KS-stat Matrix')

    # p-value 矩阵的热力图
    plt.subplot(1, 2, 2)
    sns.heatmap(p_value_df, annot=True, cmap='Reds', square=True, cbar=True)
    plt.title('p-value Matrix')

    # 创建保存结果的目录
    save_path = os.path.join(data_path, "smp_" + split_column_name + "_ks_test")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 调整布局，保存图像和矩阵
    plt.tight_layout()
    fig_filename = os.path.join(save_path, 'ks_pvalue_heatmaps.png')
    plt.savefig(fig_filename)
    ks_stat_df_filename = os.path.join(save_path, 'ks_stat_matrix.csv')
    p_value_df_filename = os.path.join(save_path, 'p_value_matrix.csv')
    ks_stat_df.to_csv(ks_stat_df_filename)
    p_value_df.to_csv(p_value_df_filename)

    return significant_results


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

# 定义一个函数来计算文本特征
def calculate_text_features(text):
    # 如果文本为空或者不是字符串，返回默认值
    if text is pd.NA or not text or not isinstance(text, str):
        return {
            'char_count': 0,
            'word_count': 0,
            'avg_word_length': 0.0,
            'punctuation_count': 0,
            'uppercase_count': 0,
            'digit_count': 0,
            'stopword_ratio': 0.0,
            'noun_ratio': 0.0,
            'verb_ratio': 0.0,
            'unique_word_ratio': 0.0,
            'avg_sentence_length': 0.0
        }
    
    # 字符总数
    char_count = len(text)
    
    # 单词总数
    words = word_tokenize(text)
    word_count = len(words)
    
    # 平均单词长度
    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0.0
    
    # 标点符号总数
    punctuation_count = len(re.findall(r'[^\w\s]', text))
    
    # 大写字母总数
    uppercase_count = sum(1 for c in text if c.isupper())
    
    # 数字字符总数
    digit_count = sum(1 for c in text if c.isdigit())
    
    # 停用词比例
    stop_words = set(stopwords.words('english'))
    stopword_count = sum(1 for word in words if word.lower() in stop_words)
    stopword_ratio = stopword_count / word_count if word_count > 0 else 0.0
    
    # 词性标注及比例
    pos_tags = pos_tag(words)
    pos_counts = Counter(tag for word, tag in pos_tags)
    total_pos_tags = sum(pos_counts.values())
    noun_ratio = (pos_counts.get('NN', 0) + pos_counts.get('NNS', 0)) / total_pos_tags if total_pos_tags > 0 else 0.0
    verb_ratio = (pos_counts.get('VB', 0) + pos_counts.get('VBD', 0)) / total_pos_tags if total_pos_tags > 0 else 0.0
    
    # 独特词汇比例
    unique_words = set(words)
    unique_word_ratio = len(unique_words) / word_count if word_count > 0 else 0.0
    
    # 平均句子长度
    sentences = re.split(r'[.!?]+', text)
    sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences if sentence.strip()]
    avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0.0
    
    return {
        'char_count': char_count,
        'word_count': word_count,
        'avg_word_length': avg_word_length,
        'punctuation_count': punctuation_count,
        'uppercase_count': uppercase_count,
        'digit_count': digit_count,
        'stopword_ratio': stopword_ratio,
        'noun_ratio': noun_ratio,
        'verb_ratio': verb_ratio,
        'unique_word_ratio': unique_word_ratio,
        'avg_sentence_length': avg_sentence_length
    }

def add_text_numeric_feat(smp_data):
    # 对每一列应用calculate_text_features函数，并收集特征
    for col in ['title', 'user_description', 'all_tags']:
        print(f"Processing {col}...")
        features = smp_data[col].apply(calculate_text_features)
        features_df = pd.DataFrame(features.tolist())
        # 在特征名之前加上列名作为前缀
        features_df.columns = [f"{col}_{name}" for name in features_df.columns]
        smp_data = pd.concat([smp_data, features_df], axis=1)
    return smp_data

# 🔹 通道注意力 SEBlock
class SEBlock(nn.Module):
    def __init__(self, input_dim, reduction=8):
        super(SEBlock, self).__init__()
        reduced_dim = max(1, input_dim // reduction)
        self.fc1 = nn.Linear(input_dim, reduced_dim)
        self.fc2 = nn.Linear(reduced_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = x.mean(dim=0, keepdim=True)  # 对 batch 进行全局池化
        attn = self.fc1(attn)
        attn = nn.ReLU()(attn)
        attn = self.fc2(attn)
        attn = self.sigmoid(attn)  # 计算通道注意力
        return x * attn  # 重新加权拼接后的 embedding

# 🔹 Transformer 编码器
class TransformerEncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        residual = x
        x = self.self_attn(x, x, x)[0]
        x = self.dropout1(x)
        x = self.norm1(x + residual)
        
        residual = x
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout2(self.linear2(x))
        x = self.norm2(x + residual)
        return x

# 🔹 MLP
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)

# 🔹 Transformer + MLP
class TransformerMLPModel(nn.Module):
    def __init__(self, input_dim=216, num_heads=3, transformer_hidden_dim=512, 
                 mlp_hidden_dims=[128, 64, 32, 8], output_dim=1, dropout=0.1):
        super(TransformerMLPModel, self).__init__()
        num_heads = 10
        while input_dim % num_heads:
            num_heads -= 1

        self.se_block = SEBlock(input_dim)  # 通道注意力模块
        self.transformer = TransformerEncoderBlock(input_dim, num_heads, transformer_hidden_dim, dropout)
        self.mlp = MLP(input_dim, mlp_hidden_dims, output_dim)  # 最终预测

    def forward(self, x):
        x = self.se_block(x)  # 先调整 A+B 特征的权重
        x = x.unsqueeze(0)  # Transformer 输入要求 (seq_len, batch_size, input_dim)
        x = self.transformer(x)
        x = x.squeeze(0)  # 变回 (batch_size, input_dim)
        return self.mlp(x)

# 🔹 AllinOneMLP（最终版本）
class AllinOneMLP(nn.Module):
    def __init__(self, image_dim, text_dim, user_dim, mask, dropout_rate=0.2):
        super(AllinOneMLP, self).__init__()

        self.mask = mask
        dims = [image_dim, text_dim, user_dim]
        total_dims = sum([dims[i] for i in range(len(dims)) if mask >> i & 1])
        
        self.trans_mlp = TransformerMLPModel(input_dim=total_dims)
        self.residual = nn.Linear(total_dims, 1)  # 残差连接，确保 B 影响力不变
        self.dropout = nn.Dropout(p=dropout_rate)  # Feature Dropout

    def forward(self, x):
        x = self.dropout(x)  # Dropout 让模型不依赖 A
        return self.trans_mlp(x) + self.residual(x)  # 残差连接，确保 B 信息不会丢失

# 消融实验

# 1. 去掉dropout
# 2. 去掉残差连接
# 3. 去掉transformermlp
# 4. 去掉transformer
# 5. 去掉mlp（保留transformer+一个fc）
# 6. 去掉SEBlock

# 🔹 Transformer + MLP
class TransformerMLPModelAblation(nn.Module):
    def __init__(self, input_dim=216, num_heads=3, transformer_hidden_dim=512, 
                 mlp_hidden_dims=[128, 64, 32, 8], output_dim=1, dropout=0.1,
                 need_se_block=True, need_transformer=True, need_mlp=True):
        super(TransformerMLPModelAblation, self).__init__()
        num_heads = 10
        while input_dim % num_heads:
            num_heads -= 1

        self.se_block = SEBlock(input_dim)  # 通道注意力模块
        self.transformer = TransformerEncoderBlock(input_dim, num_heads, transformer_hidden_dim, dropout)
        self.need_se_block = need_se_block
        self.need_transformer = need_transformer
        self.need_mlp = need_mlp
        if self.need_mlp:
            self.mlp = MLP(input_dim, mlp_hidden_dims, output_dim)  # 最终预测
        else:
            self.mlp = nn.Linear(input_dim, output_dim)  # 最终预测
        self.need_se_block = need_se_block
        self.need_transformer = need_transformer
        self.need_mlp = need_mlp

    def forward(self, x):
        if self.need_se_block:
            x = self.se_block(x)  # 先调整 A+B 特征的权重
        if self.need_transformer:
            x = x.unsqueeze(0)  # Transformer 输入要求 (seq_len, batch_size, input_dim)
            x = self.transformer(x)
            x = x.squeeze(0)  # 变回 (batch_size, input_dim)
        return self.mlp(x)

class WithoutDropout(nn.Module):
    def __init__(self, image_dim, text_dim, user_dim, mask, dropout_rate=0):
        super(WithoutDropout, self).__init__()

        self.mask = mask
        dims = [image_dim, text_dim, user_dim]
        total_dims = sum([dims[i] for i in range(len(dims)) if mask >> i & 1])
        
        self.trans_mlp = TransformerMLPModel(input_dim=total_dims, dropout=dropout_rate)
        self.residual = nn.Linear(total_dims, 1)  # 残差连接，确保 B 影响力不变
        self.dropout = nn.Dropout(p=dropout_rate)  # Feature Dropout

    def forward(self, x):
        x = self.dropout(x)  # Dropout 让模型不依赖 A
        return self.trans_mlp(x) + self.residual(x)  # 残差连接，确保 B 信息不会丢失

class WithoutResidual(nn.Module):
    def __init__(self, image_dim, text_dim, user_dim, mask, dropout_rate=0.2):
        super(WithoutResidual, self).__init__()

        self.mask = mask
        dims = [image_dim, text_dim, user_dim]
        total_dims = sum([dims[i] for i in range(len(dims)) if mask >> i & 1])
        
        self.trans_mlp = TransformerMLPModel(input_dim=total_dims)
        self.residual = nn.Linear(total_dims, 1)  # 残差连接，确保 B 影响力不变
        self.dropout = nn.Dropout(p=dropout_rate)  # Feature Dropout

    def forward(self, x):
        x = self.dropout(x)  # Dropout 让模型不依赖 A
        return self.trans_mlp(x) # + self.residual(x)  # 残差连接，确保 B 信息不会丢失

class WithoutTransformerMLP(nn.Module):
    def __init__(self, image_dim, text_dim, user_dim, mask, dropout_rate=0.2):
        super(WithoutTransformerMLP, self).__init__()

        self.mask = mask
        dims = [image_dim, text_dim, user_dim]
        total_dims = sum([dims[i] for i in range(len(dims)) if mask >> i & 1])
        
        self.trans_mlp = TransformerMLPModel(input_dim=total_dims)
        self.residual = nn.Linear(total_dims, 1)  # 残差连接，确保 B 影响力不变
        self.dropout = nn.Dropout(p=dropout_rate)  # Feature Dropout

    def forward(self, x):
        x = self.dropout(x)  # Dropout 让模型不依赖 A
        return self.residual(x)

class WithoutTransformer(nn.Module):
    def __init__(self, image_dim, text_dim, user_dim, mask, dropout_rate=0.2):
        super(WithoutTransformer, self).__init__()

        self.mask = mask
        dims = [image_dim, text_dim, user_dim]
        total_dims = sum([dims[i] for i in range(len(dims)) if mask >> i & 1])
        
        self.trans_mlp = TransformerMLPModelAblation(input_dim=total_dims, need_transformer=False)
        self.residual = nn.Linear(total_dims, 1)  # 残差连接，确保 B 影响力不变
        self.dropout = nn.Dropout(p=dropout_rate)  # Feature Dropout

    def forward(self, x):
        x = self.dropout(x)  # Dropout 让模型不依赖 A
        return self.trans_mlp(x) + self.residual(x)  # 残差连接，确保 B 信息不会丢失

class WithoutMLP(nn.Module):
    def __init__(self, image_dim, text_dim, user_dim, mask, dropout_rate=0.2):
        super(WithoutMLP, self).__init__()

        self.mask = mask
        dims = [image_dim, text_dim, user_dim]
        total_dims = sum([dims[i] for i in range(len(dims)) if mask >> i & 1])
        
        self.trans_mlp = TransformerMLPModelAblation(input_dim=total_dims, need_mlp=False)
        self.residual = nn.Linear(total_dims, 1)  # 残差连接，确保 B 影响力不变
        self.dropout = nn.Dropout(p=dropout_rate)  # Feature Dropout

    def forward(self, x):
        x = self.dropout(x)  # Dropout 让模型不依赖 A
        return self.trans_mlp(x) + self.residual(x)  # 残差连接，确保 B 信息不会丢失

class WithoutSEBlock(nn.Module):
    def __init__(self, image_dim, text_dim, user_dim, mask, dropout_rate=0.2):
        super(WithoutSEBlock, self).__init__()

        self.mask = mask
        dims = [image_dim, text_dim, user_dim]
        total_dims = sum([dims[i] for i in range(len(dims)) if mask >> i & 1])
        
        self.trans_mlp = TransformerMLPModelAblation(input_dim=total_dims, need_se_block=False)
        self.residual = nn.Linear(total_dims, 1)  # 残差连接，确保 B 影响力不变
        self.dropout = nn.Dropout(p=dropout_rate)  # Feature Dropout

    def forward(self, x):
        x = self.dropout(x)  # Dropout 让模型不依赖 A
        return self.trans_mlp(x) + self.residual(x)  # 残差连接，确保 B 信息不会丢失

def train_and_evaluate(
    model_class, image_dim, text_dim, user_dim, train_loader, test_loader,
    device, save_file_prefix, model_name, masks, X_test_tensor, y_test_tensor, 
    ckpt_gap, num_epochs=500, lr=1e-4, weight_decay=5e-4, stop_threshold=100, 
    clip_and_norm=False, min_eps=0.000001, 
):
    criterion = nn.MSELoss()
    shap_df_final = None
    results = []
    
    for mask in masks:
        def get_input(input_data, mask):
            res = []
            if mask >> 0 & 1:
                res.append(input_data[:, :image_dim])
            if mask >> 1 & 1:
                res.append(input_data[:, image_dim:image_dim+text_dim])
            if mask >> 2 & 1:
                res.append(input_data[:, image_dim+text_dim:])
            return torch.cat(res, dim=1)
        
        model = model_class(image_dim, text_dim, user_dim, mask).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        print("mask: ", mask)

        epoch_losses = []
        test_losses = []
        min_train_loss = float('inf')
        min_test_loss = float('inf')
        last_refresh_train = 0
        last_refresh_test = 0

        progress_bar = tqdm.tqdm(range(num_epochs), desc=f"Training model {model_name}")
        for epoch in progress_bar:
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(get_input(inputs, mask))
                loss = criterion(outputs, labels)
                loss.backward()
                if clip_and_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()
                running_loss += loss.item()
            
            epoch_losses.append(running_loss / len(train_loader))
            
            model.eval()
            with torch.no_grad():
                total_loss = 0.0
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(get_input(inputs, mask))
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                test_losses.append(total_loss / len(test_loader))
            
            if epoch_losses[-1] < min_train_loss:
                min_train_loss = epoch_losses[-1]
                last_refresh_train = epoch
                torch.save(model.state_dict(), f'{model_name}_training_{save_file_prefix}_mask_{mask}.pt')
            if test_losses[-1] < min_test_loss:
                min_test_loss = test_losses[-1]
                last_refresh_test = epoch
                torch.save(model.state_dict(), f'{model_name}_testing_{save_file_prefix}_mask_{mask}.pt')
            if (epoch+1) % ckpt_gap == 0:
                torch.save(model.state_dict(), f'{model_name}_ckpt_{epoch}_{save_file_prefix}_mask_{mask}.pt')
            
            progress_bar.set_postfix(train_loss=f"{epoch_losses[-1]:.4f}", test_loss=f"{test_losses[-1]:.4f}")
        
        model.load_state_dict(torch.load(f'{model_name}_testing_{save_file_prefix}_mask_{mask}.pt'))
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epoch + 2), epoch_losses, color='b', label=f'Training Loss as {model_name}')
        plt.plot(range(1, epoch + 2), test_losses, color='r', label=f'Testing Loss as {model_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Epoch vs Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        output_y_labels = y_test_tensor.clone()
        output_y_preds = model(get_input(X_test_tensor, mask))
        
        correlation, _ = spearmanr(output_y_labels.cpu().detach().numpy().flatten(), output_y_preds.cpu().detach().numpy().flatten())
        mae = torch.mean(torch.abs(output_y_labels - output_y_preds)).item()
        
        print(f'Correlation between actual labels and predicted values at {model_name}: {correlation}')
        print(f'Mean Absolute Error (MAE) at {model_name}: {mae}')

        results.append([epoch_losses, test_losses])

    return results

def get_input(input_data, mask):
    res = []
    if mask >> 0 & 1:
        res.append(input_data[:, :image_dim])
    if mask >> 1 & 1:
        res.append(input_data[:, image_dim:image_dim+text_dim])
    if mask >> 2 & 1:
        res.append(input_data[:, image_dim+text_dim:])
    return torch.cat(res, dim=1)
