import os
import gc

data_path = '/root/autodl-tmp/data/'

image_data_path = os.path.join(data_path, 'image')
image_test_data_path = os.path.join(image_data_path, 'test')
image_train_data_path = os.path.join(image_data_path, 'train')

video_data_path = os.path.join(data_path, 'video_file')
video_test_data_path = os.path.join(video_data_path, 'test')
video_train_data_path = os.path.join(video_data_path, 'train')


import json
import pandas as pd
def read_image_data(data_path, data_type):
    with open(os.path.join(data_path, data_type + '_additional_information.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)
        data = pd.DataFrame(data)
        data['Pid'] = data['Pid'].astype(str)
    with open(os.path.join(data_path, data_type + '_category.json'), 'r', encoding='utf-8') as f:
        tmp_data = json.load(f)
        tmp_data = pd.DataFrame(tmp_data)
        tmp_data['Pid'] = tmp_data['Pid'].astype(str)
        data = pd.merge(data, tmp_data, on=['Pid', 'Uid'])
    with open(os.path.join(data_path, data_type + '_temporalspatial_information.json'), 'r', encoding='utf-8') as f:
        tmp_data = json.load(f)
        tmp_data = pd.DataFrame(tmp_data)
        tmp_data['Pid'] = tmp_data['Pid'].astype(str)
        data = pd.merge(data, tmp_data, on=['Pid', 'Uid'])
    with open(os.path.join(data_path, data_type + '_text.json'), 'r', encoding='utf-8') as f:
        tmp_data = json.load(f)
        tmp_data = pd.DataFrame(tmp_data)
        tmp_data['Pid'] = tmp_data['Pid'].astype(str)
        data = pd.merge(data, tmp_data, on=['Pid', 'Uid'])
    with open(os.path.join(data_path, data_type + '_user_data.json'), 'r', encoding='utf-8') as f:
        tmp_data = json.load(f)
        tmp_data = pd.DataFrame(tmp_data)
        tmp_data['Pid'] = tmp_data['Pid'].astype(str)
        data = pd.merge(data, tmp_data, on=['Pid', 'Uid'])
    with open(os.path.join(data_path, data_type + '_img_filepath.txt'), 'r', encoding='utf-8') as f:
        tmp_data = f.readlines()
        tmp_data = [x.strip() for x in tmp_data]
        tmp_data = pd.DataFrame(tmp_data, columns=['img_path'])
        data = pd.concat([tmp_data, data], axis=1)
    if data_type == "train":
        with open(os.path.join(image_train_data_path, 'train' + '_label.txt'), 'r', encoding='utf-8') as f:
            tmp_data = f.readlines()
            tmp_data = [float(x.strip()) for x in tmp_data]
            tmp_data = pd.DataFrame(tmp_data, columns=['label'])
            tmp_data = pd.concat([tmp_data, data], axis=1)
    return data

train_data = read_image_data(image_train_data_path, "train")
test_data = read_image_data(image_test_data_path, "test")
# print("Train data:")
# print(train_data.head(3))
# print("Test data:")
# print(test_data.head(3))


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


import pandas as pd

df_train_posts = pd.DataFrame(ds_train_posts)
df_train_users = pd.DataFrame(ds_train_users)
df_train_videos = pd.DataFrame(ds_train_videos)
df_train_labels = pd.DataFrame(ds_train_labels)
df_test_posts = pd.DataFrame(ds_test_posts)
df_test_users = pd.DataFrame(ds_test_users)
df_test_videos = pd.DataFrame(ds_test_videos)
df_users = pd.concat([df_train_users, df_test_users])
df_users = df_users.drop_duplicates(subset="uid", keep="first")

video_train_data = pd.merge(
    df_train_posts,
    df_users,
    on="uid",
    how="left"
)

video_train_data = pd.merge(
    video_train_data,
    df_train_videos,
    on=["uid", "pid"],
    how="left"
)

video_train_data = pd.merge(
    video_train_data,
    df_train_labels,
    on=["uid", "pid"],
    how="left"
)

video_test_data = pd.merge(
    df_test_posts,
    df_users,
    on="uid",
    how="left"
)

video_test_data = pd.merge(
    video_test_data,
    df_test_videos,
    on=["uid", "pid"],
    how="left"
)

del ds_train_posts, ds_train_users, ds_train_videos, ds_train_labels, \
    df_train_posts, df_train_users, df_train_videos, df_train_labels, \
    ds_test_posts, ds_test_users, ds_test_videos, \
    df_test_posts, df_test_users, df_test_videos 
gc.collect()


def process_video_data(data):
    tmp_data = data.copy()
    tmp_data['uid'] = data['uid'].str.replace('USER', '').astype(int)
    tmp_data['pid'] = data['pid'].str.replace('POST', '').astype(int)
    tmp_data['video_ratio'] = data['video_ratio'].str.replace('p', '').astype(int)
    tmp_data['vid'] = data['vid'].str.replace('VIDEO', '').astype(int)
    tmp_data["post_time"] = (
        pd.to_datetime(data["post_time"])
        .astype('int') // 10**9
    )
    return tmp_data

processed_video_train_data = process_video_data(video_train_data)
processed_video_test_data = process_video_data(video_test_data)

# processed_video_test_data


video_train_path = processed_video_train_data[['video_path']]
video_test_path = processed_video_test_data[['video_path']]

video_train_data = processed_video_train_data.drop(columns=['video_path'])
video_test_data = processed_video_test_data.drop(columns=['video_path'])

video_train_numeric_data = video_train_data.select_dtypes(include=['number'])
video_test_numeric_data = video_test_data.select_dtypes(include=['number'])

video_train_text_data = video_train_data.select_dtypes(exclude=['number'])
video_test_text_data = video_test_data.select_dtypes(exclude=['number'])

# video_train_text_data


import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from decord import VideoReader, cpu
from torchvision import transforms
from torchvision.models import resnet50

# -----------------------
# 全局关闭 cuDNN，但仍使用 GPU
# -----------------------
torch.backends.cudnn.enabled = False

class VideoTemporalEncoder(nn.Module):
    def __init__(self,
                 num_frames: int = 16,
                 img_size: int = 224,
                 cnn_feat_dim: int = 2048,
                 lstm_hidden: int = 1024,
                 lstm_layers: int = 2,
                 out_dim: int = 1024,
                 device: str = 'cuda'):
        super().__init__()
        self.device = device
        self.num_frames = num_frames

        # 1) 2D‑CNN 特征提取（不加载在线预训练，改为手动加载本地权重）
        cnn = resnet50(pretrained=False)
        state_dict = torch.load("resnet50-0676ba61.pth", map_location=device)
        cnn.load_state_dict(state_dict)
        cnn.fc = nn.Identity()
        self.cnn = cnn.to(device)

        # 2) 双向多层 LSTM 做时序建模
        self.lstm = nn.LSTM(
            input_size=cnn_feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        ).to(device)

        # 3) 投影到更高维度（只接最后一层双向输出，所以 in_features = 2 * lstm_hidden）
        self.proj = nn.Linear(lstm_hidden * 2, out_dim).to(device)

        # 帧预处理
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def encode(self, video_path: str) -> torch.Tensor:
        # 尝试打开视频，fault_tol=1 遇到一帧错就跳过
        try:
            vr = VideoReader(video_path, ctx=cpu(0), fault_tol=1)
        except:
            print(f"[WARN] 无法打开视频 {video_path}")
            # 返回全零向量，长度与投影层输出一致
            return torch.zeros(self.proj.out_features, dtype=torch.float32)
    
        total = len(vr)
        if total < 1:
            print(f"[WARN] 视频无帧可读 {video_path}")
            return torch.zeros(self.proj.out_features, dtype=torch.float32)
    
        # 等距抽帧
        idx = np.linspace(0, total - 1, self.num_frames, dtype=int)
        try:
            frames = vr.get_batch(idx).asnumpy()  # (T, H, W, 3)
        except:
            print(f"[WARN] 解码帧失败 {video_path}")
            return torch.zeros(self.proj.out_features, dtype=torch.float32)
    
        # 逐帧提取特征
        feats = []
        for f in frames:
            img = Image.fromarray(f[..., ::-1])  # BGR → RGB
            x = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat2d = self.cnn(x)
            feats.append(feat2d)
        seq = torch.cat(feats, dim=0).unsqueeze(0)  # (1, T, D)
    
        # LSTM 聚合
        _, (h_n, _) = self.lstm(seq)
        fwd, bwd = h_n[-2], h_n[-1]
        last = torch.cat([fwd, bwd], dim=1)
        out = self.proj(last)
        return out.squeeze(0).cpu()


# video_train_encoding = video_train_encoding.reshape(4000, -1)
# video_test_encoding = video_test_encoding.reshape(2000, -1)

import gc, torch, pandas as pd, numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder

torch.cuda.empty_cache(); gc.collect()

model_name = "intfloat/e5-large-v2"

text_encoder = SentenceTransformer(model_name)
dim = text_encoder.get_sentence_embedding_dimension()

def encode_text_column(series: pd.Series, is_multi_token=False) -> np.ndarray:
    vecs = []
    for txt in series.fillna("").astype(str):
        txt = txt.strip()
        if not txt or txt in {"〖 〗", "[]"}:
            vecs.append(np.zeros(dim)); continue

        if is_multi_token:
            tokens = [t.strip() for t in txt.strip("[]").split(",") if t.strip()]
            emb = text_encoder.encode(tokens, batch_size=16, convert_to_numpy=True)
            vecs.append(emb.mean(axis=0))
        else:
            vecs.append(text_encoder.encode(txt, convert_to_numpy=True))
    return np.vstack(vecs)

train_content_vec = encode_text_column(video_train_text_data["post_content"])
test_content_vec  = encode_text_column(video_test_text_data["post_content"])

train_suggest_vec = encode_text_column(video_train_text_data["post_suggested_words"],
                                       is_multi_token=True)
test_suggest_vec  = encode_text_column(video_test_text_data["post_suggested_words"],
                                       is_multi_token=True)

cat_cols = ["post_location", "post_text_language", "video_format", "music_title"]
cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
cat_encoder.fit(pd.concat([video_train_text_data[cat_cols],
                           video_test_text_data[cat_cols]]))
train_cat = cat_encoder.transform(video_train_text_data[cat_cols])
test_cat  = cat_encoder.transform(video_test_text_data[cat_cols])

final_train = np.hstack([train_content_vec,
                         train_suggest_vec,
                         # video_train_encoding,
                         train_cat,
                         video_train_numeric_data.values])

final_test  = np.hstack([test_content_vec,
                         test_suggest_vec,
                         # video_test_encoding,
                         test_cat,
                         video_test_numeric_data.values])

print("训练集维度:", final_train.shape)
print("测试集维度:", final_test.shape)

# split_tab_transformer_reg.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =============================================================================
# 数据集
# =============================================================================
class TabDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray | None = None):
        self.X = X.astype(np.float32)
        self.y = None if y is None else y.astype(np.float32)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        if self.y is None:
            return x
        return x, torch.tensor(self.y[idx], dtype=torch.float32)


# =============================================================================
# 模型
# =============================================================================
class GlobalBranch(nn.Module):
    def __init__(self, in_dim: int, d_model: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, d_model)
        )

    def forward(self, x):        # x [B, in_dim]
        return self.net(x)


class MAPELoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        ape = torch.abs((y_true - y_pred) / (torch.abs(y_true) + self.epsilon))
        return torch.mean(ape) * 100

def safe_mape(y, p, eps=1e-4):
    mask = np.abs(y) > eps
    if mask.sum() == 0:
        return np.nan
    return 100 * np.mean(np.abs((y[mask] - p[mask]) / y[mask]))


class SplitTabTransformerReg(nn.Module):
    def __init__(
        self,
        n_c_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_ff: int = 256,
        scale_init: float = 0.3,
        dropout_ab: float = 0.0,
    ):
        super().__init__()
        # A/B/C 三个全连接分支
        self.branch_a = GlobalBranch(1024, d_model)
        self.branch_b = GlobalBranch(1024, d_model)
        self.branch_c = GlobalBranch(2048, d_model)

        # 可学习缩放
        self.use_scale = scale_init is not None
        if self.use_scale:
            self.scale_a = nn.Parameter(torch.tensor(scale_init))
            self.scale_b = nn.Parameter(torch.tensor(scale_init))
            self.scale_c = nn.Parameter(torch.tensor(scale_init))

        # dropout
        self.drop_ab = nn.Dropout(dropout_ab)

        # D（剩余特征）嵌入为 token
        self.d_embed = nn.Parameter(torch.empty(n_c_features, d_model))
        nn.init.xavier_uniform_(self.d_embed, gain=0.1)

        # Transformer 编码器
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, n_layers)

        # Head：LayerNorm + 展平 + MLP -> 1
        total_tokens = n_c_features + 3   # A,B,C 各 1 token + D 段 n_c_features tokens
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Flatten(),
            nn.Linear(total_tokens * d_model, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):                        # x [B,  ?  ]
        # 切分四段：A(0:1024), B(1024:2048), C(2048:4096), D(4096:)
        a = x[:, :1024]
        b = x[:, 1024:2048]
        c = x[:, 2048:4096]
        d = x[:, 4096:]

        # 三个分支
        tok_a = self.branch_a(a)
        tok_b = self.branch_b(b)
        tok_c = self.branch_c(c)
        if self.use_scale:
            tok_a = self.scale_a * tok_a
            tok_b = self.scale_b * tok_b
            tok_c = self.scale_c * tok_c

        # dropout + 加上序列维度
        tok_a = self.drop_ab(tok_a).unsqueeze(1)   # [B,1,d_model]
        tok_b = self.drop_ab(tok_b).unsqueeze(1)
        tok_c = self.drop_ab(tok_c).unsqueeze(1)

        # D 段直接做 embedding token
        # d: [B, n_c_features] -> [B, n_c_features, 1] * [n_c_features, d_model]
        tok_d = d.unsqueeze(-1) * self.d_embed    # [B, n_c_features, d_model]

        # 拼接所有 token
        tokens = torch.cat([tok_a, tok_b, tok_c, tok_d], dim=1)  # [B, 3+n_c_features, d_model]

        # LayerNorm 稳定数值
        tokens = nn.functional.layer_norm(tokens, tokens.shape[-1:])

        # Transformer 编码
        h = self.encoder(tokens)

        # Head 输出
        return self.head(h).squeeze(-1)            # [B]


# =============================================================================
# 训练 + 预测
# =============================================================================
def train_model(
    train_arr: np.ndarray,
    epochs: int = 60,
    batch: int = 128,
    lr: float = 3e-4,
    log_y: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    # 特征 Z‑score
    X = train_arr[:, :-1]
    y = train_arr[:, -1]
    mu, std = X.mean(0, keepdims=True), X.std(0, keepdims=True)
    std[std == 0] = 1.0
    X = (X - mu) / std

    # 标签 log1p
    if log_y:
        y_tr = np.log1p(np.clip(y, 0, None))
        inv_fn = lambda t: np.expm1(t)
    else:
        y_tr, inv_fn = y, lambda t: t

    train_ds = TabDataset(X, y_tr)
    dl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=2)

    # 注意这里 n_c_features = X.shape[1] - 4096
    model = SplitTabTransformerReg(
        n_c_features=X.shape[1] - 4096
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, 10, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler()
    mape_loss = MAPELoss()

    for ep in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for Xb, yb in dl:
            Xb, yb = Xb.to(device), yb.to(device)
            if not torch.isfinite(Xb).all():
                raise ValueError("NaN/Inf in input")
            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                pred = model(Xb)
                loss = mape_loss(pred, yb)
            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError("NaN/Inf loss")

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            epoch_loss += loss.item() * len(Xb)

        scheduler.step()
        epoch_loss /= len(train_ds)

        # —— 计算 MAPE ——（在 CPU, 原尺度）
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for Xb, yb in dl:
                pb = model(Xb.to(device)).cpu().numpy()
                ys.append(inv_fn(yb.numpy()))
                ps.append(inv_fn(pb))
        mape = safe_mape(np.concatenate(ys), np.concatenate(ps))

        print(f"Epoch {ep:02d} | loss={epoch_loss:.4f} | "
              f"MAPE={mape:.2f}% | lr={scheduler.get_last_lr()[0]:.1e}")

    return model, (mu, std), inv_fn


@torch.no_grad()
def predict(
    model: nn.Module,
    test_arr: np.ndarray,
    mu_std: tuple[np.ndarray, np.ndarray],
    inv_fn,
    batch: int = 256,
):
    mu, std = mu_std
    X_te = (test_arr - mu) / std
    ds = TabDataset(X_te, y=None)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2)

    device = next(model.parameters()).device
    model.eval()
    preds = []
    for Xb in dl:
        pb = model(Xb.to(device)).cpu().numpy()
        preds.append(pb)
    preds = np.concatenate(preds)
    return inv_fn(preds)        # 原始标签尺度


import numpy as np
import pandas as pd

# ---------- 超参数 ----------
EPS   = 1e-4   # 标签 0 平滑
IQR_K = 1.5    # winsorize 强度

# ---------- 清洗工具 ----------
def _fill_nan_inf(df: pd.DataFrame):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df.fillna(df.median(numeric_only=True))

def _winsorize(df: pd.DataFrame, k: float = IQR_K):
    q1, q3 = df.quantile(0.25), df.quantile(0.75)
    low, high = q1 - k*(q3-q1), q3 + k*(q3-q1)
    return df.clip(lower=low, upper=high, axis=1)

def _minmax(df: pd.DataFrame):
    min_v, rng = df.min(), df.max()-df.min()
    rng[rng == 0] = 1.0
    return (df - min_v) / rng, (min_v, rng)

def clean_numpy_arrays(train_arr: np.ndarray,
                       test_arr: np.ndarray,
                       return_scaler: bool = False):
    """
    train_arr shape = (N_train, n_features+1)  (最后一列是标签)
    test_arr  shape = (N_test,  n_features)
    """

    # --- 1. 拆分成 DataFrame 便于列操作 ---
    dfX_tr = pd.DataFrame(train_arr[:, :-1])
    y_tr   = train_arr[:, -1].astype(np.float32)
    dfX_te = pd.DataFrame(test_arr)

    # --- 2. NaN / Inf 填充 ---
    dfX_tr = _fill_nan_inf(dfX_tr)
    dfX_te = _fill_nan_inf(dfX_te)

    # --- 3. Winsorize 异常值 ---
    dfX_tr = _winsorize(dfX_tr)
    dfX_te = _winsorize(dfX_te)

    # --- 4. 统一归一化 ---
    dfX_tr, scaler = _minmax(dfX_tr)
    min_v, rng = scaler
    dfX_te      = (dfX_te - min_v) / rng      # 用同一 scaler

    # --- 5. 标签 0 平滑 ---
    y_tr = np.where(np.abs(y_tr) < EPS, EPS, y_tr)

    # --- 6. 拼回 numpy ---
    train_clean = np.hstack([dfX_tr.values, y_tr.reshape(-1, 1)]).astype(np.float32)
    test_clean  = dfX_te.values.astype(np.float32)

    if return_scaler:
        return train_clean, test_clean, scaler
    return train_clean, test_clean

import numpy as np

video_train_arr = final_train.copy()
video_test_arr  = final_test.copy()

# ========== 直接调用 ==========
video_train_arr_clean, video_test_arr_clean = clean_numpy_arrays(
    video_train_arr, video_test_arr
)

print("train_clean:", video_train_arr_clean.shape,
      "| test_clean:", video_test_arr_clean.shape)


model, mu_std, inv_fn = train_model(
    video_train_arr,
    epochs=200,
    batch=512,
    lr=3e-3,
    log_y=True
)

y_pred = predict(model, video_test_arr, mu_std, inv_fn)
np.save("test_pred.npy", y_pred)

prediction = pd.DataFrame(y_pred, columns=['polularity_score'])
prediction = pd.concat([pd.DataFrame(load_dataset("smpchallenge/SMP-Video", 'posts')['test'])['pid'], prediction], axis=1)
prediction.to_csv(
    "result.csv",
    index=False
)
