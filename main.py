# main.py (修改后)


import torch, time, os, shutil
import models, utils
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import FECGDataset
from config import config
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import argparse
from clinical_evaluator import robust_detect_r_peaks, calculate_qrs_performance, calculate_fhr_error
SAMPLING_RATE = 200

# 【新增】导入计算 R² 和 MSE 的库
from sklearn.metrics import r2_score, mean_squared_error

import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_DEVICE_OEDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(41)
torch.cuda.manual_seed(41)

# 保存模型的函数保持不变
def save_ckpt(state, is_best, model_save_dir):
    current_w = os.path.join(model_save_dir, config.current_w)
    best_w = os.path.join(model_save_dir, config.best_w)
    torch.save(state, current_w)
    if is_best: shutil.copyfile(current_w, best_w)

# train_epoch 函数保持不变
def train_epoch(model, optimizer, criterion, scheduler, train_dataloader, show_interval=100):
    model.train()
    losses = []
    total = 0
    tbar = tqdm(train_dataloader)
    for i, (inputs, target) in enumerate(tbar):      
        data = inputs.to(device)
        labelt = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output,labelt.to(torch.float32))
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    tbar.close()       
    for i in range(len(losses)):
        total = total + losses[i]
    total /= len(losses)
    # 【修改】移除无意义的f1分数
    return total

# val_epoch 函数保持不变
def val_epoch(model, optimizer, criterion, scheduler, val_dataloader, show_interval=100):
    model.eval()
    losses = []
    total = 0
    tbar = tqdm(val_dataloader)
    for i, (inputs, target) in enumerate(tbar):     
        data = inputs.to(device)
        labelt = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output,labelt.to(torch.float32))
        losses.append(loss.item())
    for i in range(len(losses)):
        total = total + losses[i]
    total /= len(losses)
    # 【修改】移除无意义的f1分数
    return total

# 【新增】独立的测试函数
def test_epoch(model, test_dataloader, model_name):
    """
    在独立的测试集上评估最终模型的性能（包含信号级和临床级指标）
    """
    model.eval()
    all_targets = []
    all_outputs = []
    
    print("\n" + "="*20)
    print("开始在独立测试集上进行最终评估...")
    print(f"测试数据文件: r10.edf")
    print("="*20)

    with torch.no_grad():
        tbar = tqdm(test_dataloader)
        for i, (inputs, target) in enumerate(tbar):
            data = inputs.to(device, dtype=torch.float32)
            labelt = target.to(device, dtype=torch.float32)
            output = model(data)
            all_targets.append(labelt.cpu().numpy())
            all_outputs.append(output.cpu().numpy())

    # --- 计算并打印最终的评估指标 ---
    # 拼接成一个大数组
    all_targets_np = np.vstack(all_targets)
    all_outputs_np = np.vstack(all_outputs)
    targets_flat = all_targets_np.flatten()
    outputs_flat = all_outputs_np.flatten()

    # 信号级指标
    mse = mean_squared_error(targets_flat, outputs_flat)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets_flat, outputs_flat)

    # 临床级指标
    true_peaks = robust_detect_r_peaks(targets_flat, sampling_rate=SAMPLING_RATE)
    pred_peaks = robust_detect_r_peaks(outputs_flat, sampling_rate=SAMPLING_RATE)
    qrs_performance = calculate_qrs_performance(true_peaks, pred_peaks, tolerance_ms=20, sampling_rate=SAMPLING_RATE)
    fhr_mae = calculate_fhr_error(true_peaks, pred_peaks, sampling_rate=SAMPLING_RATE, tolerance_ms=100)

    print("\n--- ✅ 最终测试结果 ---")
    print("--- 信号级指标 ---")
    print(f"均方误差 (MSE): {mse:.6f}")
    print(f"均方根误差 (RMSE): {rmse:.6f}")
    print(f"决定系数 (R²): {r2:.6f}")
    print("\n--- 临床级指标 ---")
    print(f"QRS F1-Score: {qrs_performance['f1_score']:.4f} (Se={qrs_performance['sensitivity']:.4f}, P+={qrs_performance['precision']:.4f})")
    print(f"胎心率误差 (FHR MAE): {fhr_mae:.4f} BPM")
    print("------------------------\n")


def weights_init_normal(m):
    # ... (此函数无变化) ...
    if isinstance(m, nn.Conv1d):
        tanh_gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(m.weight.data, gain=tanh_gain)

def train(args):
    model = getattr(models, config.model_name)(output_size=300)
    args.ckpt = None
    if args.ckpt and not args.resume:
        state = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(state['state_dict'])
        print('train with pretrained weight')
   
    model.apply(weights_init_normal)
    model = model.to(device)
    
    # --- 【核心修改】数据加载 ---
    # 定义清晰的数据文件索引
    TRAIN_FILES = [1, 2, 3] # r04, r07, r08
    VAL_FILE = [0]          # r01
    TEST_FILE = [4]         # r10
    
    print("--- 正在准备数据集 ---")
    # 创建训练数据集
    train_dataset = FECGDataset(data_path=config.train_dir, file_indices=TRAIN_FILES)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    print(f"训练集: {len(train_dataset)} 个样本, 来自文件索引 {TRAIN_FILES}")

    # 创建验证数据集
    val_dataset = FECGDataset(data_path=config.val_dir, file_indices=VAL_FILE)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=0)
    print(f"验证集: {len(val_dataset)} 个样本, 来自文件索引 {VAL_FILE}")

    # 创建测试数据集
    test_dataset = FECGDataset(data_path=config.test_dir, file_indices=TEST_FILE)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=0)
    print(f"测试集: {len(test_dataset)} 个样本, 来自文件索引 {TEST_FILE}")
    print("----------------------\n")
 
    optimizer = optim.Adam(model.parameters(), lr=config.lr)  
    criterion = nn.MSELoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.1)
    
    model_save_dir = '%s/%s' % (config.ckpt, config.model_name)
    if args.ex: model_save_dir += args.ex
    
    # 【修改】移除了 best_f1, 使用 min_loss 作为标准
    min_loss = float('inf')
    lr = config.lr
    start_epoch = 1
    stage = 1
    
    print("--- 🚀 开始训练 ---")
    for epoch in range(start_epoch, config.max_epoch + 1):
        since = time.time()
        
        # 【修改】函数返回值已更新
        train_loss = train_epoch(model, optimizer, criterion, exp_lr_scheduler, train_dataloader, show_interval=epoch)
        val_loss = val_epoch(model, optimizer, criterion, exp_lr_scheduler, val_dataloader, show_interval=epoch)
        
        print('\n')
        # 【修改】更新打印信息，移除f1
        print('#epoch:%02d stage:%d train_loss:%.4e val_loss:%0.4e time:%s\n'
              % (epoch, stage, train_loss, val_loss, utils.print_time_cost(since)))
              
        is_best = val_loss < min_loss
        min_loss = min(val_loss, min_loss)
        
        # 【修改】state字典中移除f1
        state = {"state_dict": model.state_dict(), "epoch": epoch, "loss": val_loss, 'lr': lr, 'stage': stage}
        save_ckpt(state, is_best, model_save_dir)

    print("--- 🎉 训练完成 ---")

    # --- 【新增】训练后自动开始测试 ---
    # 加载训练过程中保存的最佳模型
    # --- 【新增】训练后自动开始测试 ---
    best_w_path = os.path.join(model_save_dir, config.best_w)
    if os.path.exists(best_w_path):
        print(f"\n加载性能最佳的模型 '{config.best_w}' 用于最终测试...")
        best_model_state = torch.load(best_w_path, map_location=device)
        model.load_state_dict(best_model_state['state_dict'])
        # 【修改】调用更新后的测试函数
        test_epoch(model, test_dataloader, config.model_name)
    else:
        print("未找到最佳模型文件，跳过测试。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', help='仅支持 "train" 命令')
    parser.add_argument("--ex", type=str, help="experience name")
    # 以下参数在当前逻辑中未使用，但保留
    parser.add_argument("--ckpt", type=str, help="the path of model weight file")
    parser.add_argument("--resume", action='store_true', default=False)
    args = parser.parse_args()

    if args.command == 'train':
        train(args)
    else:
        print("无效命令，请使用 'train'")