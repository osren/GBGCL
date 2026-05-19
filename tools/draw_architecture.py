"""
GBGCL Model Architecture Diagram
仿照参考图风格绘制
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(18, 14))
ax.set_xlim(0, 18)
ax.set_ylim(0, 14)
ax.axis('off')
ax.set_title('GBGCL: Granular Ball Graph Contrastive Learning', fontsize=18, fontweight='bold', pad=20)

# 颜色定义
colors = {
    'input': '#E8F4FD',      # 浅蓝
    'encoder': '#D4E6F1',    # 蓝色
    'embed': '#FEF9E7',      # 浅黄
    'granule': '#E8DAEF',    # 紫色
    'diffusion': '#D5F5E3',  # 绿色
    'loss': '#FADBD8',       # 红色
    'arrow': '#2C3E50',      # 深灰
    'text': '#1A1A1A',       # 黑色
}

def draw_box(ax, x, y, w, h, text, color, fontsize=10, bold=False):
    """绘制圆角矩形框"""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor='#34495E', linewidth=2)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, color=colors['text'])

def draw_arrow(ax, x1, y1, x2, y2, label='', color='#2C3E50'):
    """绘制箭头连线"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8))
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.2, label, ha='center', va='bottom',
                fontsize=8, color='#555555')

# ==================== 主流程绘制 ====================

# ---- 输入层 ----
input_y = 11.5
draw_box(ax, 0.5, input_y, 2.5, 1.5, 'Node Features\nx ∈ ℝⁿˣᵈ', colors['input'], fontsize=10, bold=True)
draw_box(ax, 3.3, input_y, 2.2, 1.5, 'Edge Index\n[E, 2]', colors['input'], fontsize=10, bold=True)

# ---- GCN Encoder ----
encoder_y = 9.2
draw_box(ax, 0.5, encoder_y, 5.5, 1.8, 'GCN Encoder\n(Online / Target)', colors['encoder'], fontsize=11, bold=True)

# 连接线和标签
ax.annotate('', xy=(3.3, input_y + 0.5), xytext=(3, input_y + 0.5),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
ax.annotate('', xy=(3.3, input_y + 1.0), xytext=(3, input_y + 1.0),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

# ---- Hidden Embeddings ----
embed_y = 6.8
draw_box(ax, 0.5, embed_y, 5.5, 1.6, 'Hidden Embeddings h = or_embeds + pr_embeds\n[N, d]', colors['embed'], fontsize=10, bold=True)

# 分支箭头
ax.annotate('', xy=(0.8, embed_y + 1.6), xytext=(0.8, encoder_y),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
ax.annotate('', xy=(5.5, embed_y + 1.6), xytext=(5.5, encoder_y),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

# ==================== 右侧 Granule Diffusion Module ====================

# granule_y = 6.8
draw_box(ax, 7.0, embed_y, 4.5, 1.6, 'Granule Ball Clustering\n(quality-based graph partitioning)', colors['granule'], fontsize=9, bold=True)

# Ball Graph Construction
ball_y = 4.8
draw_box(ax, 7.0, ball_y, 4.5, 1.5, 'Ball Graph Construction\nW~ = f(topo, center_sim)', colors['granule'], fontsize=9)

# K-step Diffusion
diff_y = 2.8
draw_box(ax, 7.0, diff_y, 2.0, 1.5, 'K-step\nDiffusion\nH^{t+1} = (1-β)Hᵗ + βD⁻¹WHᵗ', colors['diffusion'], fontsize=9, bold=True)

# Write Back
write_y = 2.8
draw_box(ax, 9.8, write_y, 1.7, 1.5, 'Write Back\nz_new = αh + (1-α)Hᴷ', colors['diffusion'], fontsize=8, bold=True)

# 连接线 - Granule分支
ax.annotate('', xy=(9.0, embed_y + 0.8), xytext=(6.0, embed_y + 0.8),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
ax.text(7.5, embed_y + 1.0, 'every N epochs', fontsize=7, color='#777777', style='italic')

ax.annotate('', xy=(9.25, ball_y + 1.5), xytext=(9.25, embed_y),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

ax.annotate('', xy=(9.25, diff_y + 1.5), xytext=(9.25, ball_y),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

ax.annotate('', xy=(10.65, diff_y + 0.75), xytext=(9.0, diff_y + 0.75),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

# ==================== 左侧 Loss 分支 ====================

# Predictor
pred_y = 4.8
draw_box(ax, 0.5, pred_y, 2.2, 1.5, 'Predictor\nMLP', colors['encoder'], fontsize=10, bold=True)
ax.annotate('', xy=(1.1, pred_y + 1.5), xytext=(1.1, embed_y),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
ax.text(0.5, 5.8, 'h_pred', fontsize=8, color='#777777')

# Target Encoder (无梯度)
draw_box(ax, 0.5, pred_y - 2.2, 2.2, 1.5, 'Target Encoder\n(EMA, no grad)', '#BDC3C7', fontsize=9, bold=True)
ax.annotate('', xy=(1.1, pred_y - 0.1), xytext=(1.1, pred_y - 0.7),
            arrowprops=dict(arrowstyle='->', color='#95A5A6', lw=1.5, linestyle='dashed'))
ax.text(0.5, pred_y - 1.5, 'h_target', fontsize=8, color='#777777')

# Loss 框
loss_y = 1.0
draw_box(ax, 3.5, loss_y, 3.5, 2.5, 'Losses', colors['loss'], fontsize=11, bold=True)

# BYOL Loss
draw_box(ax, 3.7, loss_y + 1.6, 1.5, 0.7, 'BYOL Loss\n(node-level)', '#F5B7B1', fontsize=8)
ax.annotate('', xy=(4.45, loss_y + 2.5), xytext=(1.6, pred_y + 0.3),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.2))
ax.annotate('', xy=(4.45, loss_y + 2.5), xytext=(1.6, pred_y - 0.7),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.2))

# Ball Scatter Loss
draw_box(ax, 5.5, loss_y + 1.6, 1.4, 0.7, 'Ball Scatter\n(RSM)', '#F5B7B1', fontsize=8)
ax.annotate('', xy=(6.2, loss_y + 2.5), xytext=(7.5, ball_y + 0.3),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.2))

# Ball InfoNCE
draw_box(ax, 3.7, loss_y + 0.1, 1.5, 0.7, 'Ball InfoNCE\n(alignment)', '#F5B7B1', fontsize=8)
ax.annotate('', xy=(4.45, loss_y + 0.8), xytext=(7.5, ball_y - 0.3),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.2))

# ==================== EMA Update ====================
ema_y = 0.3
draw_box(ax, 7.3, ema_y, 3.5, 1.0, 'EMA Update Target Encoder\np ← momentum · p + (1-momentum) · new_p', '#D5D8DC', fontsize=9, bold=True)
ax.annotate('', xy=(9.05, loss_y), xytext=(9.05, ema_y + 1.0),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

# ==================== Legend / 图例 ====================
legend_y = 0.3
legend_items = [
    (colors['input'], 'Input'),
    (colors['encoder'], 'Encoder'),
    (colors['embed'], 'Embeddings'),
    (colors['granule'], 'Granule Module'),
    (colors['diffusion'], 'Diffusion'),
    (colors['loss'], 'Losses'),
]
for i, (c, label) in enumerate(legend_items):
    rect = FancyBboxPatch((0.5 + i*2.4, legend_y), 0.4, 0.3, boxstyle="round,pad=0.02",
                          facecolor=c, edgecolor='#34495E', linewidth=1)
    ax.add_patch(rect)
    ax.text(1.0 + i*2.4, legend_y - 0.1, label, fontsize=7, ha='left', va='top')

# ==================== 模块说明注释 ====================
# Granule 模块详细说明
ax.text(12.5, 8.5, 'Granule Ball Clustering:\n'
         '• Quality metric: detach/homo/edges/deg\n'
         '• BFS-based ball expansion\n'
         '• auto_quality() auto-selects metric',
       fontsize=8, va='top', ha='left',
       bbox=dict(boxstyle='round', facecolor='#F7F9F9', edgecolor='#BDC3C7', alpha=0.9))

ax.text(12.5, 5.5, 'Ball Graph:\n'
         '• Topology: cross-ball edge counts\n'
         '• Center sim: cosine similarity\n'
         '• KNN sparse + self-loops',
       fontsize=8, va='top', ha='left',
       bbox=dict(boxstyle='round', facecolor='#F7F9F9', edgecolor='#BDC3C7', alpha=0.9))

ax.text(12.5, 2.5, 'K-step Diffusion:\n'
         '• Random walk on ball graph\n'
         '• β: mixing coefficient\n'
         '• K: diffusion steps (3/5/10/20)',
       fontsize=8, va='top', ha='left',
       bbox=dict(boxstyle='round', facecolor='#F7F9F9', edgecolor='#BDC3C7', alpha=0.9))

# ==================== 标题标注 ====================
ax.text(3.0, 13.2, 'BYOL-style\nSelf-Supervised', fontsize=9, ha='center', va='bottom',
        color='#2C3E50', style='italic',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='#BDC3C7', alpha=0.8))

ax.text(8.75, 13.2, 'Granule Diffusion Module', fontsize=9, ha='center', va='bottom',
        color='#8E44AD', style='italic',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='#BDC3C7', alpha=0.8))

ax.text(5.25, 3.5, 'Multi-Level\nContrastive', fontsize=9, ha='center', va='bottom',
        color='#C0392B', style='italic',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='#BDC3C7', alpha=0.8))

plt.tight_layout()
plt.savefig('/Users/didi/Desktop/GBGCL/figures/gbgcl_architecture.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('/Users/didi/Desktop/GBGCL/figures/gbgcl_architecture.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print('Architecture diagram saved to figures/gbgcl_architecture.png')
plt.show()