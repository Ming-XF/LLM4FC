import os
import re
import matplotlib.pyplot as plt
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--type", type=int, required=True, help="")
args = parser.parse_args()

LOG_DIR = '.'          # 日志文件所在目录

if args.type == 2:
    OUTPUT_IMG = 'multi_model_metrics_dementia.png'
    OUTPUT_HTML = 'model_metrics_comparison_dementia.html'
    DATASET = "Dementia"
    metrix = "auc"
else:
    OUTPUT_IMG = 'multi_model_metrics_dementia400.png'
    OUTPUT_HTML = 'model_metrics_comparison_dementia400.html'
    DATASET = "Dementia400"
    metrix = "f_score"


def generate_html_table(model_data):
    """生成对比不同方法的HTML表格"""
    html = """
    <html>
    <head>
        <title>Model Metrics Comparison</title>
        <style>
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
                font-family: Arial, sans-serif;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: center;
            }
            th {
                background-color: #f2f2f2;
                position: sticky;
                top: 0;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            tr:hover {
                background-color: #f1f1f1;
            }
            .metric-header {
                background-color: #e6f7ff;
            }
        </style>
    </head>
    <body>
        <h1>Model Metrics Comparison (Average of Best Epochs from 3 Runs)</h1>
        <table>
            <thead>
                <tr>
                    <th rowspan="2">Model</th>
                    <th colspan="6" class="metric-header">Metrics (Average of Best Epochs)</th>
                </tr>
                <tr>
                    <th>AUC</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F-score</th>
                    <th>Loss</th>
                </tr>
            </thead>
            <tbody>
    """
    
    sorted_models = []
    for model_name, best_epochs in model_data.items():
        # 计算平均值
        avg_metrics = {
            'auc': np.mean([e['auc'] for e in best_epochs]),
            'accuracy': np.mean([e['accuracy'] for e in best_epochs]),
            'precision': np.mean([e['precision'] for e in best_epochs]),
            'recall': np.mean([e['recall'] for e in best_epochs]),
            'f_score': np.mean([e['f_score'] for e in best_epochs]),
            'loss': np.mean([e['loss'] for e in best_epochs])
        }
        sorted_models.append((model_name, avg_metrics))

    # 按选择的指标排序
    sorted_models.sort(key=lambda x: x[1][metrix], reverse=True)
    
    for rank, (model_name, avg_metrics) in enumerate(sorted_models, 1):    
        html += f"""
                <tr>
                    <td>{model_name}</td>
                    <td>{avg_metrics['auc']:.4f}</td>
                    <td>{avg_metrics['accuracy']:.4f}</td>
                    <td>{avg_metrics['precision']:.4f}</td>
                    <td>{avg_metrics['recall']:.4f}</td>
                    <td>{avg_metrics['f_score']:.4f}</td>
                    <td>{avg_metrics['loss']:.4f}</td>
                </tr>
        """
    
    html += """
            </tbody>
        </table>
    </body>
    </html>
    """
    
    return html

def parse_log(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    auc_pattern = re.compile(r'AUC: ([\d.]+)')
    acc_pattern = re.compile(r'Accuracy: ([\d.]+)')
    prec_pattern = re.compile(r'Precision: ([\d.]+)')
    recall_pattern = re.compile(r'Recall: ([\d.]+)')
    fscore_pattern = re.compile(r'F_score: ([\d.]+)')
    loss_pattern = re.compile(r'Loss: ([\d.]+)')

    # 按 Repeat 分割
    train_blocks = ''.join(lines).split('########## Repeat:')[1:]  # 去掉开头的空字符串
    
    if len(train_blocks) != 3:
        print(f"警告: {path} 中只找到 {len(train_blocks)} 次训练，期望3次")
        return None
    
    best_epochs_per_run = []
    
    for block in train_blocks:
        aucs = [float(m) for m in auc_pattern.findall(block)]
        accs = [float(m) for m in acc_pattern.findall(block)]
        precs = [float(m) for m in prec_pattern.findall(block)]
        recalls = [float(m) for m in recall_pattern.findall(block)]
        fscores = [float(m) for m in fscore_pattern.findall(block)]
        losses = [float(m) for m in loss_pattern.findall(block)]

        # 确保每个指标数量一致
        if not len(aucs) == len(accs) == len(precs) == len(recalls) == len(fscores) == len(losses):
            print(f"警告: {path} 中指标数量不一致")
            return None
            
        epochs = [{
            'auc': aucs[i],
            'accuracy': accs[i],
            'precision': precs[i],
            'recall': recalls[i],
            'f_score': fscores[i],
            'loss': losses[i]
        } for i in range(len(aucs))]
        
        # 找到当前训练中具有最大相关指标的epoch
        best_epoch = max(epochs, key=lambda x: x[metrix])
        best_epochs_per_run.append(best_epoch)

    return best_epochs_per_run

# ------------------ 主流程 ------------------

model_data = {}
metrics = ['auc', 'accuracy', 'precision', 'recall', 'f_score', 'loss']
labels = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F-score', 'Loss']

for fname in os.listdir(LOG_DIR):
    if not fname.endswith(DATASET+'.log'):
        continue
    model_name = fname.replace('train_', '').replace('.log', '')
    path = os.path.join(LOG_DIR, fname)
    best_epochs_per_run = parse_log(path)
    if best_epochs_per_run:
        model_data[model_name] = best_epochs_per_run
    else:
        print(f"无法解析 {fname}，已跳过")

if not model_data:
    print("没有找到任何可解析的日志文件")
    exit()

# 生成HTML表格
html_content = generate_html_table(model_data)
with open(OUTPUT_HTML, 'w') as f:
    f.write(html_content)
print(f"HTML表格已保存：{OUTPUT_HTML}")
    
# ------------------ 绘图 ------------------
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle('Multi-Model Metrics Comparison (Average of Best Epochs from 3 Runs)', fontsize=16)

for ax, metric, label in zip(axes.flatten(), metrics, labels):
    for model_name, best_epochs in model_data.items():
        # 为每个模型计算平均值和标准差
        values = [e[metric] for e in best_epochs]
        mean_value = np.mean(values)
        std_value = np.std(values)
        
        # 绘制平均值点和误差线
        ax.errorbar(model_name, mean_value, yerr=std_value, 
                   fmt='o', capsize=5, label=model_name)
    
    ax.set_title(label)
    ax.set_ylabel(label)
    ax.grid(True)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
print(f"图像已保存：{OUTPUT_IMG}")
plt.show()