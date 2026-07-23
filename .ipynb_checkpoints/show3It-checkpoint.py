import os
import re
import matplotlib.pyplot as plt
import numpy as np

import argparse

import pdb

def parse_log(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    auc_pattern = re.compile(r'AUC: ([\d.]+)')
    acc_pattern = re.compile(r'Accuracy: ([\d.]+)')
    prec_pattern = re.compile(r'Precision: ([\d.]+)')
    recall_pattern = re.compile(r'Recall: ([\d.]+)')
    fscore_pattern = re.compile(r'F_score: ([\d.]+)')
    test_loss_pattern = re.compile(r'Loss: ([\d.]+)')
    train_loss_pattern = re.compile(r'Train loss: ([\d.]+)')

    # 按 Repeat 分割
    train_blocks = ''.join(lines).split('########## Repeat:')[1:]  # 去掉开头的空字符串
    
    if len(train_blocks) != 3:
        print(f"警告: {path} 中只找到 {len(train_blocks)} 次训练，期望3次")
        return None
    
    best_epochs_per_run = []
    epoches = []
    
    for block in train_blocks:
        aucs = [float(m) for m in auc_pattern.findall(block)]
        accs = [float(m) for m in acc_pattern.findall(block)]
        precs = [float(m) for m in prec_pattern.findall(block)]
        recalls = [float(m) for m in recall_pattern.findall(block)]
        fscores = [float(m) for m in fscore_pattern.findall(block)]
        test_losses = [float(m) for m in test_loss_pattern.findall(block)]
        train_losses = [float(m) for m in train_loss_pattern.findall(block)]

        # 确保每个指标数量一致
        if not len(aucs) == len(accs) == len(precs) == len(recalls) == len(fscores) == len(test_losses) == len(train_losses):
            pdb.set_trace()
            print(f"警告: {path} 中指标数量不一致")
            return None
            
        epochs = [{
            'auc': aucs[i],
            'accuracy': accs[i],
            'precision': precs[i],
            'recall': recalls[i],
            'f_score': fscores[i],
            'test_loss': test_losses[i],
            'train_loss': train_losses[i]
        } for i in range(len(aucs))]
        
        # 找到当前训练中具有最大相关指标的epoch
        best_epoch = max(epochs, key=lambda x: x[metrix])
        best_epochs_per_run.append(best_epoch)
        epoches.append(epochs)

    return best_epochs_per_run, epoches

# ------------------ 主流程 ------------------
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True, help="")
parser.add_argument("--type", type=int, required=True, help="")
args = parser.parse_args()

path = args.path

model_name = os.path.basename(path).replace("train_", "").replace(".log", "")
OUTPUT_IMG = '{}_Three_Repeat_Metrics.png'.format(model_name)
OUT_DIR = "./analysis"
if args.type == 2:
    metrix = "auc"
else:
    metrix = "f_score"

model_data = {}
metrics = ['auc', 'accuracy', 'precision', 'recall', 'f_score', 'test_loss', 'train_loss']
labels = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F-score', 'Test-loss', 'Train-loss']

best_epochs_per_run, epochses = parse_log(path)
if not best_epochs_per_run:
    print(f"无法解析")
    exit()
for i, metrics in enumerate(best_epochs_per_run, 1):
    print(f"第 {i} 次训练评估指标:")
    print(f"  AUC:        {metrics['auc']:.4f}")
    print(f"  准确率:     {metrics['accuracy']:.4f}")
    print(f"  精确率:     {metrics['precision']:.4f}")
    print(f"  召回率:     {metrics['recall']:.4f}")
    print(f"  F1分数:     {metrics['f_score']:.4f}")
    print(f"  测试集损失:     {metrics['test_loss']:.4f}")
    print(f"  训练集损失:     {metrics['train_loss']:.4f}")
    print("-" * 40)

model_data["It1"] = epochses[0]
model_data["It2"] = epochses[1]
model_data["It3"] = epochses[2]

# ------------------ 绘图 ------------------
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
fig.suptitle('Multi-Model Metrics Comparison (Best Metrix Repeat)', fontsize=16)

# 绘制所有7个指标
for i, (metric, label) in enumerate(zip(metrics, labels)):
    row = i // 3
    col = i % 3
    ax = axes[row, col]
    
    for model_name, epochs in model_data.items():
        x = list(range(1, len(epochs) + 1))
        y = [e[metric] for e in epochs]
        ax.plot(x, y, label=model_name)
        if i >= len(metrics) - 2:  # 最后一个指标
            ax.set_ylim(0, 2000)
    ax.set_title(label)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(label)
    ax.grid(True)
    ax.legend()

# 隐藏最后一个空子图（如果有8个指标但只有7个需要显示）
axes[2, 2].set_visible(False)
axes[2, 1].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUT_DIR, OUTPUT_IMG), dpi=300)
print(f"图像已保存：{OUTPUT_IMG}")
# plt.show()