# test.py

import torch
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error

# 导入项目内的模块
import models
import dataset
from config import config

# 忽略一些不必要的警告
import warnings
warnings.filterwarnings("ignore")
from clinical_evaluator import (
    robust_detect_r_peaks,
    plot_rpeaks,
    calculate_qrs_performance,
    calculate_fhr_error
)

# 设置设备 (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAMPLING_RATE = 200  # 采样率，单位Hz
def test(model, test_dataloader, model_name, file_name):
    """
    测试函数，用于评估模型在测试集上的性能（包含信号级和临床级指标）
    """
    model.eval()
    all_targets = []
    all_outputs = []

    print("步骤 1/4: 模型推理，获取预测信号...")
    with torch.no_grad():
        tbar = tqdm(test_dataloader)
        for i, (inputs, target) in enumerate(tbar):
            data = inputs.to(device, dtype=torch.float32)
            labelt = target.to(device, dtype=torch.float32)
            output = model(data)
            all_targets.append(labelt.cpu().numpy())
            all_outputs.append(output.cpu().numpy())

    # --- 信号级评估 (MSE, R²) ---
    print("\n--- 信号级指标 (Signal-level Metrics) ---")
    all_targets_np = np.vstack(all_targets)
    all_outputs_np = np.vstack(all_outputs)
    targets_flat = all_targets_np.flatten()
    outputs_flat = all_outputs_np.flatten()

    mse = mean_squared_error(targets_flat, outputs_flat)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets_flat, outputs_flat)

    print(f"均方误差 (MSE): {mse:.6f}")
    print(f"均方根误差 (RMSE): {rmse:.6f}")
    print(f"决定系数 (R-squared, R²): {r2:.6f}")

    # --- 临床级评估 (Clinical-level Metrics) ---
    print("\n--- 临床级指标 (Clinical-level Metrics) ---")
    
    # 步骤 2/4: R波检测
    # 使用 flatten() 后的1D长信号进行R波检测
    print("步骤 2/4: 在真实信号和预测信号上检测R波...")
    true_peaks = robust_detect_r_peaks(targets_flat, sampling_rate=SAMPLING_RATE)
    pred_peaks = robust_detect_r_peaks(outputs_flat, sampling_rate=SAMPLING_RATE)
    print(f"检测到 {len(true_peaks)} 个真实R波, {len(pred_peaks)} 个预测R波。")
    
    # 步骤 3/4: (可选) 可视化R波检测结果以供调试
    # 创建 'results/rpeak_debug' 文件夹
    debug_dir = f'results/{model_name}/rpeak_debug/'
    os.makedirs(debug_dir, exist_ok=True)
    plot_rpeaks(
        targets_flat, 
        true_peaks, 
        title=f"R-Peak Detection on Ground Truth ({file_name})",
        save_path=os.path.join(debug_dir, f"true_{file_name}.png")
    )
    plot_rpeaks(
        outputs_flat,
        pred_peaks,
        title=f"R-Peak Detection on Model Prediction ({file_name})",
        save_path=os.path.join(debug_dir, f"pred_{file_name}.png")
    )
    
    # 步骤 4/4: 计算临床指标
    print("步骤 4/4: 计算QRS检测性能和胎心率误差...")
    # QRS 检测性能 (使用20ms容差)
    qrs_performance = calculate_qrs_performance(true_peaks, pred_peaks, tolerance_ms=20, sampling_rate=SAMPLING_RATE)
    
    # FHR 误差 (使用100ms容差)
    fhr_mae = calculate_fhr_error(true_peaks, pred_peaks, sampling_rate=SAMPLING_RATE, tolerance_ms=100)

     # --- 【新增】以更详细的表格格式打印最终评估结果 ---
    print("\n--- ✅ 最终临床评估详细报告 ---")
    print("-" * 60)
    print(f"{'Metric':<35} | {'Value'}")
    print("-" * 60)
    
    # R波检测统计
    print(f"{'Actual R Waves (Ground Truth)':<35} | {len(true_peaks)}")
    print(f"{'Predicted R Waves (Model Output)':<35} | {len(pred_peaks)}")
    print("-" * 60)
    
    # QRS 检测性能分解
    print(f"{'True Positives (TP)':<35} | {qrs_performance['TP']}")
    print(f"{'False Positives (FP)':<35} | {qrs_performance['FP']}")
    print(f"{'False Negatives (FN)':<35} | {qrs_performance['FN']}")
    print("-" * 60)
    
    # 最终性能指标
    # 注意：我们在这里也打印了Se和P+，因为它们对于理解F1分数很有帮助
    print(f"{'Sensitivity (Se)':<35} | {qrs_performance['sensitivity']:.4f}")
    print(f"{'Precision (P+)':<35} | {qrs_performance['precision']:.4f}")
    print(f"{'F1 Score':<35} | {qrs_performance['f1_score']:.4f}")
    print("-" * 60)
    
    # 胎心率误差
    print(f"{'Fetal Heart Rate MAE (BPM)':<35} | {fhr_mae:.4f}")
    print("-" * 60)
    print("\n")


    # 保存原始信号的功能仍然保留
    results_dir = f'results/{model_name}'
    np.save(os.path.join(results_dir, f'target_{file_name}.npy'), all_targets_np)
    np.save(os.path.join(results_dir, f'output_{file_name}.npy'), all_outputs_np)
    print(f"原始信号已保存至 '{results_dir}' 文件夹。")


def main():
    # --- 解析命令行参数 ---
    parser = argparse.ArgumentParser(description='模型测试脚本')
    # 添加 --test_file_index 参数，用于选择测试文件，默认为0 (r01.edf)
    parser.add_argument('--test_file_index', type=int, default=0, help='选择测试文件的索引 (0-4)')
    args = parser.parse_args()

    # 检查索引是否有效
    if not (0 <= args.test_file_index < 5):
        print(f"错误: test_file_index 必须在 0 到 4 之间。")
        return

    # --- 加载模型 ---
    # 定义模型权重文件的路径
    model_save_dir = '%s/%s' % (config.ckpt, config.model_name)
    best_w_path = os.path.join(model_save_dir, config.best_w)

    if not os.path.exists(best_w_path):
        print(f"错误: 找不到模型权重文件 {best_w_path}")
        print("请先运行 main.py 进行训练，以生成模型文件。")
        return

    # 动态加载模型架构
    model = getattr(models, config.model_name)(output_size=300)
    model.to(device)

    # 加载训练好的最佳权重
    state = torch.load(best_w_path, map_location=device)
    model.load_state_dict(state['state_dict'])
    print(f"成功加载模型权重: {best_w_path}")
    print(f"该模型在验证集上的最优损失为: {state['loss']:.6f}")

    # --- 加载数据集 ---
    # 核心步骤: 在实例化数据集之前，设置要加载的文件索引
    dataset.select_index = args.test_file_index
    
    test_dataset = dataset.FECGDataset(data_path=config.test_dir, train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=0, shuffle=False)
    selected_file_name = test_dataset.fileNames[dataset.select_index]
    print(f"已选择测试文件: {selected_file_name}")

    # --- 运行测试 ---
    # 【修改】调用更新后的test函数
    test(model, test_dataloader, config.model_name, selected_file_name.replace('.edf',''))



if __name__ == '__main__':
    main()