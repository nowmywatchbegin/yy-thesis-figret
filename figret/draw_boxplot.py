import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"警告：找不到文件 {file_path}")
        return []
    with open(file_path, 'r') as f:
        content = f.read().replace(',', ' ').split()
        return [float(x) for x in content if x.strip()]

# 指向三个成绩单
ecmp_file = "Result/Facebook_pod_a/Figret/result_ecmp.txt"
baseline_file = "Result/Facebook_pod_a/Figret/result_alpha0.txt"
figret_file = "Result/Facebook_pod_a/Figret/result_alpha1.txt"

ecmp_mlu = load_data(ecmp_file)
baseline_mlu = load_data(baseline_file)
figret_mlu = load_data(figret_file)

data = []
for mlu in ecmp_mlu:
    data.append({'Algorithm': 'ECMP\n(Traditional Heuristic)', 'Normalized Congestion Ratio': mlu})
for mlu in baseline_mlu:
    data.append({'Algorithm': 'Ablation Baseline\n(alpha=0.0)', 'Normalized Congestion Ratio': mlu})
for mlu in figret_mlu:
    data.append({'Algorithm': 'FIGRET (Ours)\n(alpha=1.0)', 'Normalized Congestion Ratio': mlu})

df = pd.DataFrame(data)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.figure(figsize=(9, 6))

# 加入绿色代表传统算法
ax = sns.boxplot(x='Algorithm', y='Normalized Congestion Ratio', data=df,
                 palette=['#99FF99', '#FF9999', '#99CCFF'], width=0.5, fliersize=4, linewidth=1.5)

plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
plt.title('Performance Comparison: Traditional vs DL-based Algorithms', fontsize=14, fontweight='bold', pad=15)
plt.ylabel('Normalized Congestion Ratio (Max MLU / Opt MLU)', fontsize=12, fontweight='bold')
plt.xlabel('Traffic Engineering Algorithm', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('final_comparison_boxplot.png', dpi=300)
print("✅ 高清学术图表已保存为：final_comparison_boxplot.png")
plt.show()