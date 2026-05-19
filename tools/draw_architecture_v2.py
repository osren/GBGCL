"""
GBGCL Model Architecture Diagram - Refined Version
精确仿照参考图风格绘制，包含数据流、模块内部结构、公式
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Arc, Circle
from matplotlib.lines import Line2D
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(20, 16))
ax = fig.add_subplot(111)
ax.set_xlim(0, 20)
ax.set_ylim(0, 16)
ax.axis('off')

# ============ 颜色主题 ============
COLORS = {
    'bg': '#FFFFFF',
    'input': '#E3F2FD',      # 浅蓝 - 输入
    'encoder': '#BBDEFB',    # 中蓝 - 编码器
    'hidden': '#FFF9C4',     # 浅黄 - 隐层
    'granule': '#E1BEE7',    # 浅紫 - 粒球
    'diffusion': '#C8E6C9',  # 浅绿 - 扩散
    'loss': '#FFCDD2',       # 浅红 - 损失
    'arrow': '#37474F',      # 深灰 - 箭头
    'text': '#212121',       # 深灰 - 文字
    'math': '#1565C0',       # 蓝色 - 公式
    'border': '#90A4AE',     # 边框色
    'target': '#CFD8DC',     # 目标网络
}

def draw_box(x, y, w, h, text, color, fontsize=9, bold=False, style='round'):
    """绘制圆角矩形框"""
    if style == 'round':
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                             facecolor=color, edgecolor=COLORS['border'], linewidth=1.5)
    else:
        box = Rectangle((x, y), w, h, facecolor=color, edgecolor=COLORS['border'], linewidth=1.5)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, color=COLORS['text'])

def draw_arrow(x1, y1, x2, y2, color=None, lw=2, label='', label_pos=0.5, fontsize=8):
    """绘制箭头连线"""
    if color is None:
        color = COLORS['arrow']
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                               connectionstyle='arc3,rad=0'))
    if label:
        mid_x = x1 + (x2 - x1) * label_pos
        mid_y = y1 + (y2 - y1) * label_pos + 0.15
        ax.text(mid_x, mid_y, label, ha='center', va='bottom', fontsize=fontsize, color='#555555')

def draw_dashed_arrow(x1, y1, x2, y2, color='#78909C'):
    """绘制虚线箭头"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5,
                               linestyle='dashed'))

# ============ 标题 ============
ax.text(10, 15.5, 'GBGCL: Granular Ball Graph Contrastive Learning', fontsize=16, fontweight='bold',
        ha='center', va='center', color='#1A237E')
ax.text(10, 15.1, 'Framework Architecture', fontsize=11, ha='center', va='center', color='#424242')

# ============ 阶段1: 输入层 ============
input_y = 13.2
draw_box(0.8, input_y, 3.2, 1.2, 'Node Features\nx ∈ ℝⁿˣᵈ', COLORS['input'], fontsize=10, bold=True)
draw_box(0.8, input_y - 1.6, 3.2, 1.2, 'Edge Index\nedge_index [2, E]', COLORS['input'], fontsize=10, bold=True)

# 数据加载器
draw_box(4.5, input_y - 0.8, 2.0, 1.6, 'Data\nLoader', '#B3E5FC', fontsize=9, bold=True)
draw_arrow(4.0, input_y, 4.5, input_y - 0.0)
draw_arrow(4.0, input_y - 1.6, 4.5, input_y - 0.8)

# ============ 阶段2: GCN Encoder ============
encoder_y = 10.2

# Online Encoder
draw_box(0.8, encoder_y, 5.5, 1.8, '', COLORS['encoder'], fontsize=9)
ax.text(3.55, encoder_y + 1.5, 'GCN Encoder (Online)', fontsize=10, fontweight='bold',
        ha='center', va='center', color='#0D47A1')

# 内部结构
draw_box(1.0, encoder_y + 0.2, 2.4, 1.2, 'GCNConv × L layers\ninput→hidden→hidden', '#90CAF9', fontsize=8)
draw_box(3.6, encoder_y + 0.2, 2.5, 1.2, 'Projection Head\nPReLU → Dropout → Linear', '#90CAF9', fontsize=8)

# Target Encoder (separate box, slightly different shade)
draw_box(0.8, encoder_y - 2.8, 5.5, 1.8, '', COLORS['target'], fontsize=9)
ax.text(3.55, encoder_y - 1.0, 'GCN Encoder (Target)', fontsize=10, fontweight='bold',
        ha='center', va='center', color='#546E7A')
draw_box(1.0, encoder_y - 2.6, 5.1, 1.2, 'GCNConv × L layers (EMA Updated)', '#B0BEC5', fontsize=8)

# 连接
draw_arrow(6.3, input_y - 0.8, 0.8, encoder_y + 0.9, label='x, edge_index', fontsize=8, label_pos=0.3)
draw_arrow(6.3, input_y - 0.8, 0.8, encoder_y - 1.0, color='#78909C', lw=1.5)

# ============ 阶段3: Hidden Embeddings ============
hidden_y = 7.0

# Online hidden
draw_box(0.8, hidden_y, 5.5, 1.5, 'Hidden Embeddings\nh = or_embeds + pr_embeds [N, d]', COLORS['hidden'], fontsize=10, bold=True)

# Target hidden
draw_box(0.8, hidden_y - 2.0, 5.5, 1.5, 'Target Embeddings\nh_target [N, d]', '#ECEFF1', fontsize=9)

# 连接线 - 从encoder到hidden
draw_arrow(3.55, encoder_y, 3.55, hidden_y + 1.5, label='or_embeds, pr_embeds', fontsize=8)
draw_arrow(3.55, encoder_y - 2.8, 3.55, hidden_y - 0.5, color='#78909C', lw=1.5)

# ============ 阶段4: Predictor (Online branch) ============
pred_y = 5.5
draw_box(0.8, pred_y, 2.2, 1.2, 'Predictor\nMLP', COLORS['encoder'], fontsize=9, bold=True)
draw_arrow(3.55, hidden_y + 0.75, 0.8, pred_y + 1.2, label='h', fontsize=8, label_pos=0.3)

# ============ 阶段5: Granule Diffusion Module ============
gb_y = 7.0
draw_box(7.5, gb_y, 5.0, 1.5, 'Granule Ball Clustering\nquality-based graph partitioning', COLORS['granule'], fontsize=9, bold=True)

# Arrow from hidden to GB
draw_arrow(6.3, hidden_y + 0.75, 7.5, gb_y + 0.75, label='h (every N epochs)', fontsize=8, label_pos=0.2)

# Granule Internal Details
gb_detail_y = 5.3
draw_box(7.5, gb_detail_y, 5.0, 1.3, 'Quality Metric: auto_quality()\nBFS-based ball expansion', '#E6CEF5', fontsize=8)

# Ball Graph Construction
ball_y = 3.8
draw_box(7.5, ball_y, 5.0, 1.2, 'Ball Graph Construction\nW~ = f(topo_edges, center_cosine_sim)', COLORS['granule'], fontsize=9)
draw_arrow(10.0, gb_y, 10.0, ball_y + 1.2, label='', fontsize=8)

# K-step Diffusion
diff_y = 2.4
draw_box(7.5, diff_y, 2.5, 1.1, 'K-step Diffusion\nH^{t+1} = (1-β)Hᵗ\n              + βD⁻¹WHᵗ', COLORS['diffusion'], fontsize=8, bold=True)

# Write Back
write_y = 2.4
draw_box(10.5, diff_y, 2.0, 1.1, 'Write Back\nz_new = α·h\n    + (1-α)·Hᴷ', COLORS['diffusion'], fontsize=8, bold=True)

draw_arrow(10.0, ball_y, 8.75, diff_y + 1.1, label='', fontsize=7)
draw_arrow(10.0, diff_y + 0.55, 10.5, diff_y + 0.55, label='', fontsize=7)

# Enhanced Feature Output
enhance_y = 1.0
draw_box(7.5, enhance_y, 5.0, 1.0, 'Enhanced Features z_new\nUsed for next epoch\'s prediction', '#A5D6A7', fontsize=9)

draw_arrow(10.0, diff_y, 10.0, enhance_y + 1.0, label='', fontsize=7)

# Connect enhanced back to predictor
draw_dashed_arrow(10.0, enhance_y + 0.5, 3.0, pred_y + 0.6, color='#78909C')

# ============ 阶段6: Loss Computation ============
loss_y = 1.0

# BYOL Loss
draw_box(0.5, loss_y, 2.5, 2.8, 'BYOL Loss\n(node-level)', COLORS['loss'], fontsize=10, bold=True)
draw_arrow(1.75, pred_y, 1.75, loss_y + 2.8, label='h_pred', fontsize=8, label_pos=0.3)
draw_arrow(1.75, hidden_y - 2.0 + 0.75, 1.75, loss_y + 2.0, color='#78909C', lw=1.5, label='h_target', fontsize=7, label_pos=0.4)

# Formula
ax.text(1.75, loss_y + 1.2, 'L_BYOL = -cos(h_pred, h_target)', fontsize=7, ha='center', color=COLORS['math'])

# Ball Scatter Loss
draw_box(3.5, loss_y + 1.5, 2.5, 1.3, 'Ball Scatter\n(RSM)', COLORS['loss'], fontsize=9)
draw_arrow(8.75, ball_y + 0.6, 3.5, loss_y + 2.15, label='H_ball', fontsize=7, label_pos=0.5)
ax.text(6.0, loss_y + 2.5, 'L_scatter = sep + uni', fontsize=7, ha='center', color=COLORS['math'])

# Ball InfoNCE
draw_box(3.5, loss_y, 2.5, 1.3, 'Ball InfoNCE\n(alignment)', COLORS['loss'], fontsize=9)
draw_arrow(8.75, ball_y + 0.6, 4.75, loss_y + 0.35, color='#78909C', lw=1, label='', fontsize=7)

# ============ 阶段7: EMA Update ============
ema_y = -0.8
draw_box(0.5, ema_y, 5.5, 0.8, 'EMA Update: θ_target ← momentum·θ_target + (1-momentum)·θ_online', '#CFD8DC', fontsize=9, bold=True)
draw_arrow(3.55, loss_y, 3.55, ema_y + 0.8, label='', fontsize=7)

# ============ Legend ============
legend_y = 14.5
legend_items = [
    ('#E3F2FD', 'Input'),
    ('#BBDEFB', 'Encoder'),
    ('#FFF9C4', 'Embeddings'),
    ('#E1BEE7', 'Granule Module'),
    ('#C8E6C9', 'Diffusion'),
    ('#FFCDD2', 'Losses'),
]
for i, (c, label) in enumerate(legend_items):
    rect = FancyBboxPatch((0.8 + i*2.2, legend_y), 0.35, 0.25, boxstyle="round,pad=0.02",
                          facecolor=c, edgecolor=COLORS['border'], linewidth=1)
    ax.add_patch(rect)
    ax.text(1.3 + i*2.2, legend_y - 0.15, label, fontsize=8, ha='left', va='top')

# ============ 详细说明框 ============
# Granule Ball Clustering Details
detail1_x, detail1_y = 13.5, 10.5
ax.text(detail1_x, detail1_y, 'Granule Ball Clustering:', fontsize=9, fontweight='bold', va='top')
ax.text(detail1_x, detail1_y - 0.4, '• Quality: homo/detach/edges/deg', fontsize=8, va='top')
ax.text(detail1_x, detail1_y - 0.8, '• BFS expansion from high-degree', fontsize=8, va='top')
ax.text(detail1_x, detail1_y - 1.2, '• Recursive split if quality improves', fontsize=8, va='top')
ax.text(detail1_x, detail1_y - 1.6, '• auto_quality() auto-selects metric', fontsize=8, va='top')

# Ball Graph Details
detail2_x, detail2_y = 13.5, 7.5
ax.text(detail2_x, detail2_y, 'Ball Graph W~:', fontsize=9, fontweight='bold', va='top')
ax.text(detail2_x, detail2_y - 0.4, '• Topology: cross-ball edge counts', fontsize=8, va='top')
ax.text(detail2_x, detail2_y - 0.8, '• Center sim: ||hi·hj|| / (||hi||·||hj||)', fontsize=8, va='top')
ax.text(detail2_x, detail2_y - 1.2, '• KNN sparse (top-k similar)', fontsize=8, va='top')
ax.text(detail2_x, detail2_y - 1.6, '• Self-loops added', fontsize=8, va='top')

# Diffusion Details
detail3_x, detail3_y = 13.5, 4.5
ax.text(detail3_x, detail3_y, 'K-step Diffusion:', fontsize=9, fontweight='bold', va='top')
ax.text(detail3_x, detail3_y - 0.4, '• Random walk on ball graph', fontsize=8, va='top')
ax.text(detail3_x, detail3_y - 0.8, '• β: mixing coefficient (default 0.2)', fontsize=8, va='top')
ax.text(detail3_x, detail3_y - 1.2, '• K: steps (3/5/10/20)', fontsize=8, va='top')
ax.text(detail3_x, detail3_y - 1.6, '• α: fusion weight (default 0.6)', fontsize=8, va='top')

# Loss Details
detail4_x, detail4_y = 13.5, 1.5
ax.text(detail4_x, detail4_y, 'Loss Functions:', fontsize=9, fontweight='bold', va='top')
ax.text(detail4_x, detail4_y - 0.4, '• BYOL: node-level contrastive', fontsize=8, va='top')
ax.text(detail4_x, detail4_y - 0.8, '• Scatter (RSM): ball diversity', fontsize=8, va='top')
ax.text(detail4_x, detail4_y - 1.2, '• InfoNCE: ball alignment via', fontsize=8, va='top')
ax.text(detail4_x, detail4_y - 1.6, '  Hungarian-matched pairs', fontsize=8, va='top')

# ============ 流程阶段标注 ============
stage_x = 0.3
ax.text(stage_x, 13.2, '①', fontsize=14, fontweight='bold', color='#1565C0')
ax.text(stage_x, encoder_y + 0.9, '②', fontsize=14, fontweight='bold', color='#1565C0')
ax.text(stage_x, hidden_y + 0.75, '③', fontsize=14, fontweight='bold', color='#1565C0')
ax.text(stage_x, pred_y + 0.6, '④', fontsize=14, fontweight='bold', color='#1565C0')
ax.text(stage_x, loss_y + 1.4, '⑤', fontsize=14, fontweight='bold', color='#1565C0')

# ============ Section Labels ============
ax.text(-0.8, 11.0, 'Online\nBranch', fontsize=9, ha='center', va='center', color='#0D47A1',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='#BBDEFB', alpha=0.9))
ax.text(-0.8, 8.0, 'Target\nBranch', fontsize=9, ha='center', va='center', color='#546E7A',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='#CFD8DC', alpha=0.9))
ax.text(10.0, 8.5, 'Granule Diffusion Module', fontsize=10, ha='center', va='center', color='#6A1B9A',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='#E1BEE7', alpha=0.9))

plt.tight_layout()
plt.savefig('/Users/didi/Desktop/GBGCL/figures/gbgcl_architecture_v2.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('/Users/didi/Desktop/GBGCL/figures/gbgcl_architecture_v2.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print('Architecture diagram v2 saved!')
plt.show()