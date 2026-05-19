"""
GBGCL Model Architecture Diagram - Professional Version
仿照参考图的专业风格绘制
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(22, 14))
ax = fig.add_subplot(111)
ax.set_xlim(0, 22)
ax.set_ylim(0, 14)
ax.axis('off')
ax.set_facecolor('#FAFAFA')

# ============ 颜色定义 ============
C = {
    'input': '#E3F2FD',
    'encoder': '#BBDEFB',
    'encoder_dark': '#64B5F6',
    'hidden': '#FFF9C4',
    'hidden_dark': '#FFF176',
    'granule': '#E1BEE7',
    'granule_dark': '#CE93D8',
    'diff': '#C8E6C9',
    'diff_dark': '#81C784',
    'loss': '#FFCDD2',
    'loss_dark': '#EF9A9A',
    'target': '#ECEFF1',
    'text': '#212121',
    'border': '#546E7A',
    'math': '#1565C0',
    'gray': '#78909C',
}

def box(x, y, w, h, text, fc, fontsize=9, bold=False, fs=8):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                      facecolor=fc, edgecolor=C['border'], linewidth=1.5, zorder=3)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold' if bold else 'normal', color=C['text'], zorder=4)

def arr(x1, y1, x2, y2, c=None, lw=2, tag='', tc='#555555', ts=7.5):
    if c is None: c = C['border']
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw,
                               connectionstyle='arc3,rad=0'), zorder=2)
    if tag:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my+0.12, tag, ha='center', va='bottom', fontsize=ts, color=tc, zorder=5)

def line(x1, y1, x2, y2, c='#B0BEC5', lw=1.5, ls='--'):
    ax.plot([x1, x2], [y1, y2], color=c, lw=lw, linestyle=ls, zorder=2)

# ============ 标题 ============
ax.text(11, 13.5, 'GBGCL: Granular Ball Graph Contrastive Learning', fontsize=18, fontweight='bold',
        ha='center', va='center', color='#1A237E', zorder=5)
ax.text(11, 13.0, 'Framework Architecture', fontsize=12, ha='center', va='center', color='#455A64', zorder=5)

# ========== 1. INPUT LAYER ==========
iy = 11.0
box(0.5, iy, 2.5, 1.0, 'Node Features\nx ∈ ℝⁿˣᵈ', C['input'], fontsize=9, bold=True)
box(0.5, iy-1.4, 2.5, 1.0, 'Edge Index\n[E, 2]', C['input'], fontsize=9, bold=True)
box(3.3, iy-0.7, 1.8, 1.4, 'Data\nLoader', '#B3E5FC', fontsize=9, bold=True)
arr(3.0, iy, 3.3, iy-0.7)
arr(3.0, iy-1.4, 3.3, iy-0.7)
ax.text(1.5, iy+1.2, '① Input', fontsize=11, fontweight='bold', color=C['encoder_dark'])

# ========== 2. ENCODERS ==========
ey = 8.5
# Online Encoder box
box(0.5, ey, 5.5, 1.8, '', C['encoder'], fontsize=9)
ax.text(3.25, ey+1.5, 'GCN Encoder (Online)', fontsize=11, fontweight='bold', color='#0D47A1', zorder=5)
box(0.7, ey+0.3, 2.4, 1.1, 'GCNConv × L\ninput→hidden→hidden', '#90CAF9', fontsize=8)
box(3.3, ey+0.3, 2.5, 1.1, 'Projection Head\nLinear→PReLU→Dropout→Linear', '#90CAF9', fontsize=8)

# Target Encoder box
box(0.5, ey-2.8, 5.5, 1.8, '', C['target'], fontsize=9)
ax.text(3.25, ey-1.0, 'GCN Encoder (Target, EMA)', fontsize=11, fontweight='bold', color='#546E7A', zorder=5)
box(0.7, ey-2.5, 5.1, 1.1, 'GCNConv × L (momentum updated)', '#B0BEC5', fontsize=8)

arr(3.3, iy-0.7, 0.5, ey+1.8, tag='x, edge_index', tc='#37474F')
line(6.8, iy-0.7, 6.8, ey-1.0, c='#90A4AE', lw=1.5, ls='--')

ax.text(1.5, ey+2.4, '② Encoder', fontsize=11, fontweight='bold', color=C['encoder_dark'])

# ========== 3. EMBEDDINGS ==========
hy = 5.8
box(0.5, hy, 5.5, 1.3, 'Hidden Embeddings\nh = or_embeds + pr_embeds [N, d]', C['hidden'], fontsize=10, bold=True)
box(0.5, hy-1.7, 5.5, 1.3, 'Target Embeddings\nh_target [N, d]', C['target'], fontsize=9)

arr(3.25, ey, 3.25, hy+1.3, tag='or_embeds + pr_embeds')
line(3.25, ey-2.8+1.8, 3.25, hy-1.7+0.65, c='#90A4AE', lw=1.5)

ax.text(1.5, hy+1.8, '③ Hidden', fontsize=11, fontweight='bold', color='#F9A825')

# ========== 4. GRANULE DIFFUSION MODULE ==========
ax.text(12.5, ey+0.5, 'Granule Diffusion Module', fontsize=12, fontweight='bold', ha='center',
        color='#6A1B9A', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#CE93D8', alpha=0.9), zorder=5)

# Arrow to granule
arr(6.0, hy+0.65, 7.5, ey+0.2, tag='h (periodic)', tc='#7B1FA2')

# Granule clustering
gy = 7.5
box(7.5, gy, 5.0, 1.3, 'Granule Ball Clustering\nquality-based graph partitioning', C['granule'], fontsize=9, bold=True)
box(7.5, gy-1.5, 5.0, 1.1, 'Quality: auto_quality() selects\nBFS expansion, recursive split', '#D7CCC8', fontsize=8)

arr(10.0, gy, 10.0, gy-1.5, tag='', tc='#8D6E63')

# Ball graph
bgy = 5.2
box(7.5, bgy, 5.0, 1.1, 'Ball Graph Construction\nW~ = topo_edges + center_cosine', C['granule'], fontsize=9)
box(7.5, bgy-1.3, 5.0, 0.9, 'KNN sparse · self-loops · normalize', '#E6EEF5', fontsize=7)

arr(10.0, gy-1.5, 10.0, bgy+1.1, tag='', tc='#8D6E63')

# K-step diffusion
ky = 3.2
box(7.5, ky, 2.8, 1.1, 'K-step Diffusion\nH^{t+1} = (1-β)Hᵗ + βD⁻¹WHᵗ', C['diff'], fontsize=8, bold=True)
box(10.8, ky, 2.0, 1.1, 'Write Back\nz_new = α·h + (1-α)·Hᴷ', C['diff'], fontsize=8, bold=True)

arr(10.0, bgy-1.3+0.9, 8.9, ky+1.1, tag='', tc='#689F38')
arr(10.3, ky+0.55, 10.8, ky+0.55, tag='', tc='#689F38')

# Enhanced features
efy = 1.5
box(7.5, efy, 5.0, 1.0, 'Enhanced Features z_new\n→ next epoch prediction', '#A5D6A7', fontsize=9, bold=True)

arr(10.0, ky, 10.0, efy+1.0, tag='', tc='#689F38')

# Dashed return arrow
line(12.5, efy+0.5, 3.5, 5.5, c='#90A4AE', lw=1.5, ls='--')
ax.text(7.0, 4.0, 'z_new', fontsize=8, color='#546E7A', ha='center')

# ========== 5. PREDICTOR ==========
py = 3.8
box(0.5, py, 2.0, 1.0, 'Predictor\nMLP', C['encoder'], fontsize=9, bold=True)
arr(3.25, hy+0.65, 0.5, py+1.0, tag='h')

# ========== 6. LOSSES ==========
ly = 0.8

# BYOL
box(0.5, ly, 2.4, 2.4, 'BYOL Loss\n(node-level)', C['loss'], fontsize=10, bold=True)
arr(1.7, py, 1.7, ly+2.4, tag='h_pred', tc='#C62828')
line(1.7, hy-1.7+0.65, 1.7, ly+1.0, c='#90A4AE', lw=1.5)
ax.text(1.7, ly+0.8, 'L=-cos(h_pred,h_target)', fontsize=7, ha='center', color=C['math'])

# Ball Scatter
box(3.3, ly+1.3, 2.2, 1.1, 'Ball Scatter\n(RSM)', C['loss'], fontsize=9)
arr(10.0, bgy+0.55, 5.5, ly+1.85, tag='H_ball', tc='#C62828')
ax.text(5.5, ly+2.5, 'L_scatter = sep+uni', fontsize=7, ha='center', color=C['math'])

# Ball InfoNCE
box(3.3, ly, 2.2, 1.1, 'Ball InfoNCE\n(alignment)', C['loss'], fontsize=9)
arr(10.0, bgy+0.55, 5.5, ly+0.15, tag='H_ball', tc='#90A4AE', lw=1)

ax.text(1.5, ly+3.0, '④ Losses', fontsize=11, fontweight='bold', color='#C62828')

# ========== 7. EMA UPDATE ==========
box(0.5, -0.6, 5.5, 0.7, 'EMA Update: θ_target ← momentum·θ_target + (1-momentum)·θ_online', C['target'], fontsize=9, bold=True)
arr(3.25, ly, 3.25, -0.6+0.7, tag='')

# ========== LEGEND ============
lx, ly2 = 0.5, 13.2
items = [('Input', C['input']), ('Encoder', C['encoder']), ('Hidden', C['hidden']),
         ('Granule', C['granule']), ('Diffusion', C['diff']), ('Loss', C['loss'])]
for i, (l, c) in enumerate(items):
    rect = FancyBboxPatch((lx + i*2.3, ly2), 0.35, 0.22, boxstyle="round,pad=0.01",
                          facecolor=c, edgecolor=C['border'], linewidth=1, zorder=5)
    ax.add_patch(rect)
    ax.text(lx + i*2.3 + 0.5, ly2-0.1, l, fontsize=8, va='top', zorder=5)

# ========== DETAIL BOXES ==========
# Granule detail
ax.text(14.0, 11.0, 'Granule Ball Clustering:', fontsize=9, fontweight='bold', va='top', color='#4A148C')
ax.text(14.0, 10.6, '• Quality: homo/detach/edges/deg', fontsize=8, va='top')
ax.text(14.0, 10.25, '• BFS expansion from high-degree nodes', fontsize=8, va='top')
ax.text(14.0, 9.9, '• Recursive split if quality improves', fontsize=8, va='top')
ax.text(14.0, 9.55, '• auto_quality() auto-selects', fontsize=8, va='top')

# Ball graph detail
ax.text(14.0, 7.0, 'Ball Graph W~:', fontsize=9, fontweight='bold', va='top', color='#4A148C')
ax.text(14.0, 6.6, '• Topology: cross-ball edge counts', fontsize=8, va='top')
ax.text(14.0, 6.25, '• Center sim: cosine similarity', fontsize=8, va='top')
ax.text(14.0, 5.9, '• KNN sparse (keep top-k)', fontsize=8, va='top')
ax.text(14.0, 5.55, '• Self-loops added for stability', fontsize=8, va='top')

# Diffusion detail
ax.text(14.0, 3.5, 'K-step Diffusion:', fontsize=9, fontweight='bold', va='top', color='#2E7D32')
ax.text(14.0, 3.1, '• Random walk on ball graph', fontsize=8, va='top')
ax.text(14.0, 2.75, '• β: mixing coefficient (0.2)', fontsize=8, va='top')
ax.text(14.0, 2.4, '• K: diffusion steps (3/5/10/20)', fontsize=8, va='top')
ax.text(14.0, 2.05, '• α: fusion weight (0.6)', fontsize=8, va='top')

# Loss detail
ax.text(14.0, 0.5, 'Loss Functions:', fontsize=9, fontweight='bold', va='top', color='#B71C1C')
ax.text(14.0, 0.1, '• BYOL: node-level contrastive', fontsize=8, va='top')
ax.text(14.0, -0.25, '• Scatter (RSM): ball diversity', fontsize=8, va='top')
ax.text(14.0, -0.6, '• InfoNCE: ball alignment via', fontsize=8, va='top')
ax.text(14.0, -0.95, '  Hungarian-matched pairs', fontsize=8, va='top')

# ========== STAGE LABELS ==========
ax.text(-0.8, iy+0.5, '①', fontsize=14, fontweight='bold', color=C['encoder_dark'])
ax.text(-0.8, ey+0.9, '②', fontsize=14, fontweight='bold', color=C['encoder_dark'])
ax.text(-0.8, hy+0.65, '③', fontsize=14, fontweight='bold', color='#F9A825')
ax.text(-0.8, py+0.5, '④', fontsize=14, fontweight='bold', color=C['encoder_dark'])
ax.text(-0.8, ly+1.2, '⑤', fontsize=14, fontweight='bold', color='#C62828')

plt.tight_layout()
plt.savefig('/Users/didi/Desktop/GBGCL/figures/gbgcl_architecture_v3.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('/Users/didi/Desktop/GBGCL/figures/gbgcl_architecture_v3.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print('v3 saved!')
plt.show()