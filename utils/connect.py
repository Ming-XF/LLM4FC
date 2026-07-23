import torch
import torch.nn.functional as F
import torch.fft as fft
import numpy as np
from scipy import stats
import antropy as ant


def segment_eeg_data(data, epoch_length, sfreq):
    """将EEG数据分段为不重叠的epochs"""
    B, C, N = data.shape
    n_samples = int(epoch_length * sfreq)
    n_epochs = N // n_samples
    
    if n_epochs == 0:
        # 使用最小可用长度
        n_samples = N
        n_epochs = 1
    
    # 简单截取
    data = data[:, :, :n_epochs * n_samples]
    segmented = data.reshape(B, C, n_epochs, n_samples)
    segmented = segmented.transpose(1, 2).contiguous()
    
    return segmented, n_epochs, n_samples


def bandpass_filter(data, f_low, f_high, sfreq):
    """频域带通滤波"""
    N = data.shape[-1]
    device = data.device
    
    data_fft = fft.fft(data, dim=-1)
    freqs = fft.fftfreq(N, 1/sfreq).abs().to(device)
    
    mask = ((freqs >= f_low) & (freqs <= f_high)).float().view(1, 1, -1)
    filtered_fft = data_fft * mask
    
    return fft.ifft(filtered_fft, dim=-1).real


def hilbert_transform(data):
    """希尔伯特变换计算解析信号"""
    N = data.shape[-1]
    device = data.device
    
    data_fft = fft.fft(data, dim=-1)
    
    # 向量化的希尔伯特滤波器
    h = torch.zeros(N, device=device)
    if N % 2 == 0:
        h[0] = h[N//2] = 1
        h[1:N//2] = 2
    else:
        h[0] = 1
        h[1:(N+1)//2] = 2
    
    h = h.view(1, 1, -1)
    analytic_fft = data_fft * h
    
    return fft.ifft(analytic_fft, dim=-1)


def calculate_connectivity_matrices(filtered_data, analytic_signal):
    """计算三种连接矩阵"""
    B, C, N = filtered_data.shape
    device = filtered_data.device
    
    # 获取相位和振幅
    phase = torch.angle(analytic_signal)
    amplitude = torch.abs(analytic_signal)
    
    # 预分配矩阵
    ispc_matrices = torch.zeros((B, C, C), device=device)
    wpli_matrices = torch.zeros((B, C, C), device=device)
    aec_matrices = torch.zeros((B, C, C), device=device)
    
    # 批量计算所有通道对
    for i in range(C):
        # 获取第i个通道的数据
        phase_i = phase[:, i:i+1, :]  # (B, 1, N)
        analytic_i = analytic_signal[:, i:i+1, :]
        amp_i = amplitude[:, i:i+1, :]
        
        for j in range(i+1, C):
            # 获取第j个通道的数据
            phase_j = phase[:, j:j+1, :]
            analytic_j = analytic_signal[:, j:j+1, :]
            amp_j = amplitude[:, j:j+1, :]
            
            # 1. 计算ISPC（相位同步）
            phase_diff = phase_i - phase_j
            ispc_val = torch.abs(torch.mean(torch.exp(1j * phase_diff), dim=-1))
            
            # 2. 计算wPLI
            cross_spectrum = analytic_i * torch.conj(analytic_j)
            imag_part = torch.imag(cross_spectrum)
            
            numerator = torch.mean(torch.abs(imag_part) * torch.sign(imag_part), dim=-1)
            denominator = torch.mean(torch.abs(imag_part), dim=-1)
            wpli_val = numerator / denominator
            wpli_val = torch.nan_to_num(wpli_val, nan=0.0)
            
            # 3. 计算AEC（振幅包络相关性）
            amp_i_centered = amp_i - torch.mean(amp_i, dim=-1, keepdim=True)
            amp_j_centered = amp_j - torch.mean(amp_j, dim=-1, keepdim=True)
            
            cov = torch.sum(amp_i_centered * amp_j_centered, dim=-1)
            var_i = torch.sum(amp_i_centered**2, dim=-1)
            var_j = torch.sum(amp_j_centered**2, dim=-1)
            aec_val = cov / torch.sqrt(var_i * var_j)
            aec_val = torch.nan_to_num(aec_val, nan=0.0)
            
            # 填充对称矩阵
            ispc_matrices[:, i, j] = ispc_val.squeeze()
            ispc_matrices[:, j, i] = ispc_val.squeeze()
            
            wpli_matrices[:, i, j] = wpli_val.squeeze()
            wpli_matrices[:, j, i] = wpli_val.squeeze()
            
            aec_matrices[:, i, j] = aec_val.squeeze()
            aec_matrices[:, j, i] = aec_val.squeeze()
    
    # 填充对角线
    diag_idx = torch.arange(C, device=device)
    ispc_matrices[:, diag_idx, diag_idx] = 1.0
    wpli_matrices[:, diag_idx, diag_idx] = 1.0
    aec_matrices[:, diag_idx, diag_idx] = 1.0
    
    return ispc_matrices, wpli_matrices, aec_matrices


def extract_statistical_features(matrices):
    """从连接矩阵中提取4个统计特征"""
    B, C, _ = matrices.shape
    device = matrices.device
    
    # 只提取上三角（不包括对角线）
    triu_indices = torch.triu_indices(C, C, offset=1)
    features = torch.zeros((B, 4), device=device)
    
    for b in range(B):
        values = matrices[b][triu_indices[0], triu_indices[1]]
        
        if len(values) < 2:
            continue
        
        # 1. 均值
        features[b, 0] = torch.mean(values)
        
        # 2. 标准差（比方差更稳定）
        features[b, 1] = torch.std(values, unbiased=False)
        
        # 3. 偏度（使用三阶矩近似）
        mean_val = torch.mean(values)
        std_val = torch.std(values, unbiased=False) + 1e-8
        skew_val = torch.mean(((values - mean_val) / std_val) ** 3)
        features[b, 2] = skew_val
        
        # 4. 近似熵（如果可用则使用原始方法，否则使用简化方法）
        if len(values) >= 100:  # 足够的数据点使用antropy
            values_np = values.cpu().numpy()
            try:
                entropy_val = ant.app_entropy(values_np, order=2)
                features[b, 3] = torch.tensor(entropy_val, device=device)
            except:
                # 简化的熵估计
                hist = torch.histc(values, bins=min(10, len(values)))
                prob = hist / len(values) + 1e-10
                entropy_val = -torch.sum(prob * torch.log(prob))
                features[b, 3] = entropy_val
        else:
            # 简化的熵估计
            hist = torch.histc(values, bins=min(10, len(values)))
            prob = hist / len(values) + 1e-10
            entropy_val = -torch.sum(prob * torch.log(prob))
            features[b, 3] = entropy_val
    
    return features


def build_feature_maps(all_features, C, device):
    """从特征构建特征图"""
    B = all_features.shape[0]
    feature_maps = torch.zeros((B, 4, C, C), device=device)
    
    # 获取上三角索引
    triu_indices = torch.triu_indices(C, C, offset=1, device=device)
    
    for b in range(B):
        # 对所有频率带和连接度量取平均
        avg_features = torch.mean(all_features[b], dim=0)  # (4,)
        
        for f_idx in range(4):
            # 创建对称矩阵
            mat = torch.zeros((C, C), device=device)
            mat[triu_indices[0], triu_indices[1]] = avg_features[f_idx]
            mat = mat + mat.T
            mat.fill_diagonal_(1.0)
            
            feature_maps[b, f_idx] = mat
    
    return feature_maps


def extract_eeg_connectivity_features(eeg_data, sfreq=500, epoch_length=2, use_all_bands=True):
    """
    主函数：从EEG数据中提取连接性特征图
    
    参数:
        eeg_data: torch.Tensor，形状为 (B, C, N)
        sfreq: 采样频率，默认500Hz
        epoch_length: 分段长度（秒），默认2秒
        use_all_bands: 是否使用所有频率带，False时只使用Alpha和Beta
    
    返回:
        feature_maps: torch.Tensor，形状为 (B, 4, C, C)
                      4个特征图：均值、标准差、偏度、熵
    """
    
    device = eeg_data.device
    B, C, N = eeg_data.shape
    
    # 1. 分段处理
    segmented, n_epochs, epoch_samples = segment_eeg_data(eeg_data, epoch_length, sfreq)
    
    # 2. 定义频率带
    if use_all_bands:
        freq_bands = [(0.5, 4), (4, 8), (8, 13), (13, 25), (25, 45)]  # 5个带
    else:
        freq_bands = [(8, 13), (13, 25)]  # 只使用Alpha和Beta带
    
    n_bands = len(freq_bands)
    
    # 3. 选择代表性的epochs（为了加速）
    sample_epochs = [0]
    if n_epochs > 1:
        sample_epochs.append(n_epochs // 2)
    if n_epochs > 2:
        sample_epochs.append(n_epochs - 1)
    
    # 4. 初始化特征存储
    all_features = torch.zeros((B, n_bands * 3, 4), device=device)
    
    # 5. 处理每个频率带
    for band_idx, (f_low, f_high) in enumerate(freq_bands):
        band_features_idx = band_idx * 3
        
        # 存储当前频率带的所有epoch特征
        epoch_features_list = []
        
        # 对每个采样epoch计算特征
        for epoch_idx in sample_epochs:
            epoch_data = segmented[:, epoch_idx, :, :]  # (B, C, N_epoch)
            
            # 滤波
            filtered_data = bandpass_filter(epoch_data, f_low, f_high, sfreq)
            
            # 希尔伯特变换
            analytic_signal = hilbert_transform(filtered_data)
            
            # 计算连接矩阵
            ispc_mat, wpli_mat, aec_mat = calculate_connectivity_matrices(
                filtered_data, analytic_signal
            )
            
            # 提取统计特征
            ispc_features = extract_statistical_features(ispc_mat)
            wpli_features = extract_statistical_features(wpli_mat)
            aec_features = extract_statistical_features(aec_mat)
            
            # 组合特征 (B, 3, 4)
            epoch_features = torch.stack([ispc_features, wpli_features, aec_features], dim=1)
            epoch_features_list.append(epoch_features)
        
        # 平均所有采样epoch的特征
        if epoch_features_list:
            avg_epoch_features = torch.mean(torch.stack(epoch_features_list, dim=0), dim=0)
            all_features[:, band_features_idx:band_features_idx+3, :] = avg_epoch_features
    
    # 6. 构建最终的特征图
    feature_maps = build_feature_maps(all_features, C, device)
    
    return feature_maps


def extract_simple_eeg_features(eeg_data, sfreq=500, epoch_length=1):
    """
    简化版特征提取：只计算关键特征，速度更快
    
    参数:
        eeg_data: torch.Tensor，形状为 (B, C, N)
        sfreq: 采样频率
        epoch_length: 分段长度（秒）
    
    返回:
        feature_maps: torch.Tensor，形状为 (B, 4, C, C)
    """
    B, C, N = eeg_data.shape
    device = eeg_data.device
    
    # 1. 只处理一个epoch
    n_samples = int(epoch_length * sfreq)
    if n_samples > N:
        n_samples = N
    
    # 取中间部分的数据（通常比较稳定）
    start_idx = max(0, (N - n_samples) // 2)
    data = eeg_data[:, :, start_idx:start_idx+n_samples]
    
    # 2. 只计算两个主要频带
    freq_bands = [(8, 13), (13, 25)]  # Alpha和Beta
    
    feature_maps = torch.zeros((B, 4, C, C), device=device)
    
    for b in range(B):
        # 为每个样本单独处理
        sample_data = data[b:b+1]
        
        # 存储所有连接值
        all_conn_values = []
        
        for f_low, f_high in freq_bands:
            # 滤波
            filtered = bandpass_filter(sample_data, f_low, f_high, sfreq)
            analytic = hilbert_transform(filtered)
            
            # 计算相位和振幅
            phase = torch.angle(analytic)
            amplitude = torch.abs(analytic)
            
            # 采样几个关键通道对
            # 使用相邻通道和跨半球连接
            channel_pairs = [
                (0, 1),                    # 相邻通道
                (C//4, C//4 + 1),         # 中间区域
                (0, C//2),                # 跨半球
                (C//2, C-1),              # 另一侧跨半球
                (C//4, 3*C//4)            # 对称位置
            ]
            
            # 确保通道索引有效
            valid_pairs = [(i, j) for i, j in channel_pairs if i < C and j < C]
            
            for i, j in valid_pairs:
                # ISPC
                phase_i = phase[:, i:i+1]
                phase_j = phase[:, j:j+1]
                phase_diff = phase_i - phase_j
                ispc_val = torch.abs(torch.mean(torch.exp(1j * phase_diff), dim=-1)).item()
                
                # 振幅相关性
                amp_i = amplitude[:, i:i+1]
                amp_j = amplitude[:, j:j+1]
                amp_i_centered = amp_i - torch.mean(amp_i, dim=-1, keepdim=True)
                amp_j_centered = amp_j - torch.mean(amp_j, dim=-1, keepdim=True)
                
                cov = torch.sum(amp_i_centered * amp_j_centered, dim=-1).item()
                var_i = torch.sum(amp_i_centered**2, dim=-1).item()
                var_j = torch.sum(amp_j_centered**2, dim=-1).item()
                
                if var_i > 0 and var_j > 0:
                    corr_val = cov / np.sqrt(var_i * var_j)
                    all_conn_values.extend([ispc_val, abs(corr_val)])
        
        # 计算统计特征
        if all_conn_values:
            conn_tensor = torch.tensor(all_conn_values, device=device)
            
            # 均值
            mean_val = torch.mean(conn_tensor)
            
            # 标准差
            std_val = torch.std(conn_tensor, unbiased=False)
            
            # 偏度
            if std_val > 1e-8:
                skew_val = torch.mean(((conn_tensor - mean_val) / std_val) ** 3)
            else:
                skew_val = torch.tensor(0.0, device=device)
            
            # 熵（简化计算）
            hist = torch.histc(conn_tensor, bins=min(10, len(conn_tensor)))
            prob = hist / len(conn_tensor) + 1e-10
            entropy_val = -torch.sum(prob * torch.log(prob))
            
            # 构建特征图
            stats_list = [mean_val, std_val, skew_val, entropy_val]
            
            for f_idx, stat_val in enumerate(stats_list):
                mat = torch.ones((C, C), device=device) * stat_val
                mat.fill_diagonal_(1.0)
                feature_maps[b, f_idx] = mat
    
    return feature_maps


# 使用示例
if __name__ == "__main__":
    # 测试数据
    B = 4
    C = 19
    N = 15000  # 30秒数据
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"使用设备: {device}")
    
    # 生成模拟EEG数据
    eeg_data = torch.randn(B, C, N, device=device) * 50
    
    print("\n1. 运行完整版本（优化参数）:")
    print("-" * 50)
    feature_maps_full = extract_eeg_connectivity_features(
        eeg_data,
        epoch_length=2,      # 减少epoch长度
        use_all_bands=False  # 只使用Alpha和Beta带
    )
    print(f"输入数据形状: {eeg_data.shape}")
    print(f"输出特征图形状: {feature_maps_full.shape}")
    print(f"特征范围: [{feature_maps_full.min():.3f}, {feature_maps_full.max():.3f}]")
    
    print("\n2. 运行简化版本（最快）:")
    print("-" * 50)
    feature_maps_simple = extract_simple_eeg_features(
        eeg_data,
        epoch_length=1  # 只处理1秒数据
    )
    print(f"输出特征图形状: {feature_maps_simple.shape}")
    print(f"特征范围: [{feature_maps_simple.min():.3f}, {feature_maps_simple.max():.3f}]")
    
    # 打印统计信息
    print("\n3. 特征图统计:")
    print("-" * 50)
    for b in range(min(2, B)):  # 只显示前2个batch
        print(f"\nBatch {b}:")
        for f_idx, f_name in enumerate(['均值', '标准差', '偏度', '熵']):
            mat_full = feature_maps_full[b, f_idx]
            mat_simple = feature_maps_simple[b, f_idx]
            
            # 计算非对角线元素的统计
            idx = torch.triu_indices(C, C, offset=1)
            full_vals = mat_full[idx[0], idx[1]]
            simple_vals = mat_simple[idx[0], idx[1]]
            
            print(f"  {f_name:6s} | 完整版: {torch.mean(full_vals):.4f} ± {torch.std(full_vals):.4f} "
                  f"| 简化版: {torch.mean(simple_vals):.4f} ± {torch.std(simple_vals):.4f}")




# import torch
# import torch.nn.functional as F
# import torch.fft as fft
# import numpy as np
# from scipy import stats
# import antropy as ant

# def extract_eeg_connectivity_features(eeg_data, sfreq=500):
#     """
#     从EEG数据中提取连接性特征图（纯PyTorch实现）
    
#     参数:
#         eeg_data: torch.Tensor，形状为 (B, C, N)
#                   B: batch_size, C: 通道数, N: 采样点数
#         sfreq: 采样频率，默认500Hz
    
#     返回:
#         feature_maps: torch.Tensor，形状为 (B, 4, C, C)
#                       4个特征图：均值、方差、偏度、熵
#     """
    
#     def segment_eeg(data):
#         """将数据分段为6秒不重叠的epochs"""
#         B, C, N = data.shape
#         epoch_length = 6  # 6秒
#         n_samples = int(epoch_length * sfreq)
#         n_epochs = N // n_samples
        
#         if n_epochs == 0:
#             raise ValueError(f"数据长度不足6秒，需要至少{epoch_length*sfreq}个采样点")
        
#         # 截取整数倍的epochs
#         data = data[:, :, :n_epochs * n_samples]
#         # 重塑为 (B, C, n_epochs, n_samples)
#         segmented = data.reshape(B, C, n_epochs, n_samples)
#         # 转置为 (B, n_epochs, C, n_samples)
#         segmented = segmented.transpose(1, 2).contiguous()
        
#         return segmented, n_epochs
    
#     def bandpass_filter(data, f_low, f_high):
#         """频域带通滤波（PyTorch实现）"""
#         B, C, N = data.shape
#         device = data.device
        
#         # 计算FFT
#         data_fft = fft.fft(data, dim=-1)
#         freqs = fft.fftfreq(N, 1/sfreq).abs().to(device)
        
#         # 创建滤波器掩码
#         mask = (freqs >= f_low) & (freqs <= f_high)
#         mask = mask.float().view(1, 1, -1)
        
#         # 应用滤波器
#         filtered_fft = data_fft * mask
        
#         # 逆FFT
#         filtered = fft.ifft(filtered_fft, dim=-1).real
        
#         return filtered
    
#     def hilbert_transform(data):
#         """希尔伯特变换计算解析信号（PyTorch实现）"""
#         N = data.shape[-1]
        
#         # 计算FFT
#         data_fft = fft.fft(data, dim=-1)
        
#         # 创建希尔伯特滤波器
#         h = torch.zeros(N, device=data.device)
#         if N % 2 == 0:
#             h[0] = h[N//2] = 1
#             h[1:N//2] = 2
#         else:
#             h[0] = 1
#             h[1:(N+1)//2] = 2
        
#         h = h.view(1, 1, -1)
        
#         # 应用滤波器
#         analytic_fft = data_fft * h
        
#         # 逆FFT得到解析信号
#         analytic = fft.ifft(analytic_fft, dim=-1)
        
#         return analytic
    
#     def calculate_connectivity_matrices(epoch_data, freq_band):
#         """计算三种连接矩阵（PyTorch实现）"""
#         f_low, f_high = freq_band
        
#         # 滤波
#         filtered = bandpass_filter(epoch_data, f_low, f_high)
        
#         # Hilbert变换
#         analytic = hilbert_transform(filtered)
        
#         # 获取相位和振幅
#         phase = torch.angle(analytic)
#         amplitude = torch.abs(analytic)
        
#         B, C, N = filtered.shape
#         device = filtered.device
        
#         # 初始化连接矩阵
#         ispc_matrices = torch.zeros((B, C, C), device=device)
#         wpli_matrices = torch.zeros((B, C, C), device=device)
#         aec_matrices = torch.zeros((B, C, C), device=device)
        
#         # 对角线设为1
#         idx = torch.arange(C, device=device)
#         ispc_matrices[:, idx, idx] = 1.0
#         wpli_matrices[:, idx, idx] = 1.0
#         aec_matrices[:, idx, idx] = 1.0
        
#         # 批量计算ISPC
#         # 使用广播计算所有电极对之间的相位差
#         for i in range(C):
#             phase_i = phase[:, i:i+1, :]  # (B, 1, N)
#             for j in range(i+1, C):
#                 phase_j = phase[:, j:j+1, :]  # (B, 1, N)
#                 phase_diff = phase_i - phase_j
                
#                 # ISPC
#                 ispc_val = torch.abs(torch.mean(torch.exp(1j * phase_diff), dim=-1))
#                 ispc_matrices[:, i, j] = ispc_val.squeeze()
#                 ispc_matrices[:, j, i] = ispc_val.squeeze()
                
#                 # wPLI
#                 analytic_i = analytic[:, i:i+1, :]  # (B, 1, N)
#                 analytic_j = analytic[:, j:j+1, :]  # (B, 1, N)
#                 cross_spectrum = analytic_i * torch.conj(analytic_j)
#                 imag_part = torch.imag(cross_spectrum)
                
#                 numerator = torch.mean(torch.abs(imag_part) * torch.sign(imag_part), dim=-1)
#                 denominator = torch.mean(torch.abs(imag_part), dim=-1)
#                 wpli_val = numerator / denominator
#                 wpli_val = torch.nan_to_num(wpli_val, nan=0.0)
                
#                 wpli_matrices[:, i, j] = wpli_val.squeeze()
#                 wpli_matrices[:, j, i] = wpli_val.squeeze()
                
#                 # AEC
#                 amp_i = amplitude[:, i:i+1, :]  # (B, 1, N)
#                 amp_j = amplitude[:, j:j+1, :]  # (B, 1, N)
                
#                 # 计算相关系数（批次处理）
#                 amp_i_centered = amp_i - torch.mean(amp_i, dim=-1, keepdim=True)
#                 amp_j_centered = amp_j - torch.mean(amp_j, dim=-1, keepdim=True)
                
#                 numerator = torch.sum(amp_i_centered * amp_j_centered, dim=-1)
#                 denominator = torch.sqrt(
#                     torch.sum(amp_i_centered**2, dim=-1) * 
#                     torch.sum(amp_j_centered**2, dim=-1)
#                 )
                
#                 aec_val = numerator / denominator
#                 aec_val = torch.nan_to_num(aec_val, nan=0.0)
                
#                 aec_matrices[:, i, j] = aec_val.squeeze()
#                 aec_matrices[:, j, i] = aec_val.squeeze()
        
#         return ispc_matrices, wpli_matrices, aec_matrices
    
#     def extract_statistical_features(matrices):
#         """从连接矩阵中提取4个统计特征"""
#         B, C, _ = matrices.shape
#         device = matrices.device
        
#         # 获取上三角索引（不包括对角线）
#         triu_indices = torch.triu_indices(C, C, offset=1, device=device)
#         features = torch.zeros((B, 4), device=device)
        
#         for b in range(B):
#             # 提取上三角值
#             values = matrices[b][triu_indices[0], triu_indices[1]]
            
#             if len(values) == 0:
#                 continue
            
#             # 均值
#             features[b, 0] = torch.mean(values)
            
#             # 方差
#             features[b, 1] = torch.var(values, unbiased=False)
            
#             # 偏度（使用scipy.stats，需转换为numpy）
#             values_np = values.cpu().numpy()
#             features[b, 2] = torch.tensor(stats.skew(values_np) if len(values_np) > 0 else 0.0, 
#                                           device=device)
            
#             # 近似熵（使用antropy，需转换为numpy）
#             if len(values_np) >= 2:
#                 entropy_val = ant.app_entropy(values_np, order=2)
#             else:
#                 entropy_val = 0.0
#             features[b, 3] = torch.tensor(entropy_val, device=device)
        
#         return features
    
#     # ========== 主流程开始 ==========
    
#     device = eeg_data.device
#     B, C, N = eeg_data.shape
    
#     # 1. 分段处理
#     segmented, n_epochs = segment_eeg(eeg_data)
    
#     # 2. 定义频率带
#     freq_bands = [
#         (0.5, 4),    # Delta
#         (4, 8),      # Theta
#         (8, 13),     # Alpha
#         (13, 25),    # Beta
#         (25, 45)     # Gamma
#     ]
    
#     # 3. 初始化特征存储
#     all_features = torch.zeros((B, len(freq_bands) * 3, 4), device=device)  # (B, 15, 4)
    
#     # 4. 处理每个频率带
#     for band_idx, freq_band in enumerate(freq_bands):
#         band_features_idx = band_idx * 3
        
#         # 对每个epoch计算特征并平均
#         epoch_features_list = []
        
#         for epoch_idx in range(n_epochs):
#             epoch_data = segmented[:, epoch_idx, :, :]  # (B, C, N_epoch)
            
#             # 计算连接矩阵
#             ispc_mat, wpli_mat, aec_mat = calculate_connectivity_matrices(epoch_data, freq_band)
            
#             # 提取特征
#             ispc_features = extract_statistical_features(ispc_mat)  # (B, 4)
#             wpli_features = extract_statistical_features(wpli_mat)  # (B, 4)
#             aec_features = extract_statistical_features(aec_mat)    # (B, 4)
            
#             epoch_features = torch.stack([ispc_features, wpli_features, aec_features], dim=1)  # (B, 3, 4)
#             epoch_features_list.append(epoch_features)
        
#         # 平均所有epochs的特征
#         if epoch_features_list:
#             avg_epoch_features = torch.mean(torch.stack(epoch_features_list, dim=0), dim=0)  # (B, 3, 4)
#             all_features[:, band_features_idx:band_features_idx+3, :] = avg_epoch_features
    
#     # 5. 构建特征图 (B, 4, C, C)
#     feature_maps = torch.zeros((B, 4, C, C), device=device)
    
#     # 获取上三角索引
#     triu_indices = torch.triu_indices(C, C, offset=1, device=device)
    
#     for b in range(B):
#         # 对每个样本，平均所有频率带和连接度量的特征
#         avg_features = torch.mean(all_features[b], dim=0)  # (4,)
        
#         for f_idx in range(4):
#             # 创建连接矩阵
#             mat = torch.zeros((C, C), device=device)
            
#             # 填充上三角部分
#             mat[triu_indices[0], triu_indices[1]] = avg_features[f_idx]
            
#             # 对称复制到下三角
#             mat = mat + mat.T
            
#             # 对角线设为1（自连接）
#             mat.fill_diagonal_(1.0)
            
#             feature_maps[b, f_idx] = mat
    
#     return feature_maps

# # 使用示例
# if __name__ == "__main__":
#     # 模拟EEG数据 (B, C, N)
#     B = 2      # batch size
#     C = 19     # 通道数
#     N = 30000  # 采样点数 (60秒, 500Hz)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # 生成随机EEG数据
#     eeg_data = torch.randn(B, C, N, device=device) * 50  # 模拟真实EEG幅值
    
#     # 提取特征图
#     feature_maps = extract_eeg_connectivity_features(eeg_data)
    
#     print(f"输入数据形状: {eeg_data.shape}")
#     print(f"设备: {eeg_data.device}")
#     print(f"输出特征图形状: {feature_maps.shape}")
#     print(f"特征图说明: (batch_size={B}, 4个特征, {C}个通道, {C}个通道)")
#     print(f"特征图值范围: [{feature_maps.min():.3f}, {feature_maps.max():.3f}]")
    
#     # 检查特征图
#     for b in range(B):
#         print(f"\nBatch {b} 特征图统计:")
#         for f_idx, f_name in enumerate(['均值', '方差', '偏度', '熵']):
#             mat = feature_maps[b, f_idx]
#             diag_mean = torch.mean(torch.diag(mat))
#             triu_mean = torch.mean(mat[torch.triu_indices(C, C, offset=1)])
#             print(f"  {f_name}: 对角线均值={diag_mean:.3f}, "
#                   f"上三角均值={triu_mean:.3f}")