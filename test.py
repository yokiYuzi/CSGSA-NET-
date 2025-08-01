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

# 设置设备 (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(model, test_dataloader):
    """
    测试函数，用于评估模型在测试集上的性能

    参数:
    model (torch.nn.Module): 加载了权重的模型
    test_dataloader (DataLoader): 测试数据的DataLoader
    """
    # 将模型设置为评估模式
    model.eval()

    # 用于存储所有样本的真实标签和模型预测值
    all_targets = []
    all_outputs = []

    print("开始测试...")
    # 在测试过程中不计算梯度
    with torch.no_grad():
        # 使用tqdm显示进度条
        tbar = tqdm(test_dataloader)
        for i, (inputs, target) in enumerate(tbar):
            # 将数据移动到指定设备
            data = inputs.to(device, dtype=torch.float32)
            labelt = target.to(device, dtype=torch.float32)

            # 模型前向传播
            output = model(data)

            # 将当前批次的真实值和预测值存入列表
            # 使用 .cpu() 将数据移回CPU，并使用 .numpy() 转换为NumPy数组
            all_targets.append(labelt.cpu().numpy())
            all_outputs.append(output.cpu().numpy())

    # 将列表中所有的批次数据拼接成一个大的NumPy数组
    # all_targets 和 all_outputs 的形状将是 (样本总数, 1, 信号长度)
    all_targets = np.vstack(all_targets)
    all_outputs = np.vstack(all_outputs)

    # 展平数组以便于计算指标，形状变为 (样本总数 * 信号长度)
    targets_flat = all_targets.flatten()
    outputs_flat = all_outputs.flatten()

    # --- 计算评估指标 ---
    mse = mean_squared_error(targets_flat, outputs_flat)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets_flat, outputs_flat)

    print("\n--- 测试结果 ---")
    print(f"均方误差 (MSE): {mse:.6f}")
    print(f"均方根误差 (RMSE): {rmse:.6f}")
    print(f"决定系数 (R-squared, R²): {r2:.6f}")
    print("------------------\n")

    # --- 保存预测结果 ---
    # 创建一个 'results' 文件夹 (如果不存在)
    if not os.path.exists('results'):
        os.makedirs('results')

    # 定义保存文件的路径
    # file_index 用于标识当前测试的是哪个文件的数据
    file_index = dataset.select_index
    file_name = test_dataloader.dataset.fileNames[file_index]
    save_path_target = os.path.join('results', f'target_{file_name}.npy')
    save_path_output = os.path.join('results', f'output_{file_name}.npy')

    # 保存真实信号和模型输出信号
    np.save(save_path_target, all_targets)
    np.save(save_path_output, all_outputs)
    print(f"测试结果已保存至 'results' 文件夹:")
    print(f" - 真实信号: {save_path_target}")
    print(f" - 模型预测信号: {save_path_output}")


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
    model = getattr(models, config.model_name)(output_size=128)
    model.to(device)

    # 加载训练好的最佳权重
    state = torch.load(best_w_path, map_location=device)
    model.load_state_dict(state['state_dict'])
    print(f"成功加载模型权重: {best_w_path}")
    print(f"该模型在验证集上的最优损失为: {state['loss']:.6f}")

    # --- 加载数据集 ---
    # 核心步骤: 在实例化数据集之前，设置要加载的文件索引
    dataset.select_index = args.test_file_index
    
    # 实例化测试数据集
    test_dataset = dataset.FECGDataset(data_path=config.test_dir, train=False)
    # 创建DataLoader
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=0, shuffle=False)
    
    selected_file = test_dataset.fileNames[dataset.select_index]
    print(f"已选择测试文件: {selected_file}")

    # --- 运行测试 ---
    test(model, test_dataloader)


if __name__ == '__main__':
    main()