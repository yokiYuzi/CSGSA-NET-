# clinical_evaluator.py (增强版)
# 新增功能：
# 1. 修正了 biosppy R波检测的bug。
# 2. 增加了基于 Scipy 的通用峰值检测器作为备用方案。
# 3. 增加了可视化函数，用于调试R波检测结果。

import numpy as np
import matplotlib.pyplot as plt
from biosppy.signals import ecg
from scipy.signal import find_peaks

def detect_r_peaks_biosppy(signal, sampling_rate):
    """
    使用 biosppy 库检测ECG信号中的R波峰值点 (已修正bug)。
    """
    try:
        # --- 这是修正的地方：去掉了变量名后面的逗号 ---
        rpeaks = ecg.ecg(signal=signal.ravel(), sampling_rate=sampling_rate, show=False)['rpeaks']
        # ------------------------------------------
        return rpeaks
    except Exception:
        # biosppy在信号质量差时可能会失败，返回空数组
        return np.array([], dtype=int)

def detect_r_peaks_scipy(signal, sampling_rate):
    """
    使用 scipy.signal.find_peaks 进行通用的峰值检测。
    这个方法更稳健，尤其适用于模型生成的、可能被平滑的信号。
    """
    # -- 参数可调 --
    # height: 峰值的最小高度。可以设置为信号标准差的一半，以滤除噪声
    # distance: 相邻峰之间的最小距离（样本数）。例如，心率不应高于240BPM (0.25秒)
    min_distance = int(sampling_rate * 0.25) 
    min_height = 0.5 * np.std(signal)
    
    peaks, _ = find_peaks(signal.ravel(), height=min_height, distance=min_distance)
    return peaks

def robust_detect_r_peaks(signal, sampling_rate):
    """
    一个稳健的R波检测流程：
    1. 优先尝试专用的 biosppy 检测器。
    2. 如果失败或检测到的峰值过少，则使用通用的 scipy 检测器作为后备。
    """
    # 优先使用 biosppy
    rpeaks = detect_r_peaks_biosppy(signal, sampling_rate)
    
    # 如果 biosppy 效果不佳（例如，对于整个测试集只检测到少于5个心跳），则启用后备方案
    if len(rpeaks) < 5:
        print("[提示] biosppy 检测效果不佳，正在尝试使用更通用的 scipy.find_peaks 作为后备...")
        rpeaks = detect_r_peaks_scipy(signal, sampling_rate)
        
    return rpeaks
    
def plot_rpeaks(signal, rpeaks, title, save_path, num_points_to_plot=2000):
    """
    一个新的可视化函数，用于绘制信号和其上检测到的R波，方便调试。
    """
    plt.figure(figsize=(15, 7))
    # 绘制部分信号
    plot_len = min(num_points_to_plot, len(signal))
    plt.plot(signal[:plot_len], label="Signal")
    
    # 在信号上标记R波位置
    rpeaks_to_plot = rpeaks[rpeaks < plot_len]
    plt.plot(rpeaks_to_plot, signal[rpeaks_to_plot], 'ro', markersize=8, label="Detected R-Peaks")
    
    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close() # 关闭图像，防止在Jupyter等环境中直接显示
    print(f"[INFO] R-peak-debug plot saved to {save_path}")


# --- calculate_qrs_performance 和 calculate_fhr_error 函数保持不变 ---
# (请将您原来的这两个函数复制到这里)
def calculate_qrs_performance(true_peaks, pred_peaks, tolerance_ms, sampling_rate):
    # ... 您原来的正确代码 ...
    # 将容差从毫秒转换为采样点数
    tolerance_samples = int(tolerance_ms * sampling_rate / 1000)
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # 复制一份预测峰值点列表，用于匹配后移除
    pred_peaks_copy = list(pred_peaks)

    # 寻找真阳性（TP）和假阴性（FN）
    for true_peak in true_peaks:
        # 在容差范围内寻找匹配的预测峰值点
        matches = [p for p in pred_peaks_copy if abs(p - true_peak) <= tolerance_samples]
        
        if matches:
            # 找到匹配，认为是TP
            true_positives += 1
            # 找到最近的一个匹配点并从列表中移除，防止重复匹配
            best_match = min(matches, key=lambda p: abs(p - true_peak))
            pred_peaks_copy.remove(best_match)
        else:
            # 未找到匹配，认为是FN
            false_negatives += 1

    # 剩余未匹配的预测峰值点即为假阳性（FP）
    false_positives = len(pred_peaks_copy)

    # 计算指标
    sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    return {
        "TP": true_positives,
        "FP": false_positives,
        "FN": false_negatives,
        "sensitivity": sensitivity, # 灵敏度 (Se)
        "precision": precision,     # 阳性预测率 (P+)
        "f1_score": f1_score
    }


def calculate_fhr_error(true_peaks, pred_peaks, sampling_rate, tolerance_ms=100):
    # ... 您上次修正后的正确代码 ...
    tolerance_samples = int(tolerance_ms * sampling_rate / 1000)
    
    matched_rr_true = []
    matched_rr_pred = []

    pred_peaks_copy = list(pred_peaks)
    for i in range(len(true_peaks) - 1):
        current_true_peak = true_peaks[i]
        next_true_peak = true_peaks[i+1]
        
        current_match = [p for p in pred_peaks_copy if abs(p - current_true_peak) <= tolerance_samples]
        next_match = [p for p in pred_peaks_copy if abs(p - next_true_peak) <= tolerance_samples]

        if current_match and next_match:
            best_current = min(current_match, key=lambda p: abs(p - current_true_peak))
            best_next = min(next_match, key=lambda p: abs(p - next_true_peak))
            
            rr_true_samples = next_true_peak - current_true_peak
            rr_pred_samples = best_next - best_current
            
            if rr_pred_samples > 0:
                matched_rr_true.append(rr_true_samples)
                matched_rr_pred.append(rr_pred_samples)

            if best_current in pred_peaks_copy:
                pred_peaks_copy.remove(best_current)

    if not matched_rr_true:
        return np.nan
    
    fhr_true_bpm = (sampling_rate * 60) / np.array(matched_rr_true)
    fhr_pred_bpm = (sampling_rate * 60) / np.array(matched_rr_pred)

    fhr_mae = np.mean(np.abs(fhr_true_bpm - fhr_pred_bpm))
    
    return fhr_mae