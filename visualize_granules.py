# visualize_granules.py
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from data import load_dataset
from models import Conv
from gb_utils import build_granules_and_rewrite

# ========== 参数 ==========
dataset_name = "Computers"
dataset_dir = "./datasets"
hidden_dim = 256
use_cuda = False

device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')

# ========== 加载数据 ==========
dataset = load_dataset(dataset_name, dataset_dir)
data = dataset[0].to(device)
print(data)

# ========== 获取节点表示（简单GNN编码） ==========
model = Conv(data.x.size(1), hidden_dim, hidden_dim, torch.nn.PReLU(), num_layers=1).to(device)
model.eval()
with torch.no_grad():
    node_embed, _ = model(data.x, data.edge_index)
node_embed = node_embed.detach().cpu()

# ========== 构建粒球 ==========
z_new, gb_sizes = build_granules_and_rewrite(
    node_embed=node_embed,
    edge_index=data.edge_index,
    quity='homo',
    sim='dot',
    alpha_write=0.5
)
print(f"[INFO] 粒球数量={len(gb_sizes)}, 平均大小={np.mean(gb_sizes):.2f}, min={np.min(gb_sizes)}, max={np.max(gb_sizes)}")

# ========== 粒球标签可视化 ==========
# 在 granular.py 中 Granular.forward 返回 GB_node_list
from granular import Granular
from torch_geometric.utils import to_scipy_sparse_matrix

adj_csr = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.x.size(0))
gb = Granular(quity='homo', sim='dot')
gb.z_detached = node_embed
GB_node_list, _, _ = gb.forward(adj_csr)

# 构造节点对应的粒球标签
node_to_ball = np.zeros(data.x.size(0), dtype=int)
for i, members in enumerate(GB_node_list):
    node_to_ball[members] = i

# ========= 降维 =========
tsne = TSNE(n_components=2, random_state=42, init='pca', perplexity=30)
emb_2d = tsne.fit_transform(node_embed)

# ========= 绘制 =========
plt.figure(figsize=(8, 6))
num_balls = len(GB_node_list)
colors = plt.cm.tab20(np.linspace(0, 1, num_balls))
for i in range(num_balls):
    members = np.array(GB_node_list[i])
    plt.scatter(emb_2d[members, 0], emb_2d[members, 1],
                s=8, color=colors[i % len(colors)], label=f"Ball {i}" if i < 10 else None, alpha=0.6)
plt.title(f"Granule Distribution on {dataset_name} dataset\n(count={num_balls}, avg size={np.mean(gb_sizes):.1f})")
plt.xlabel("t-SNE dim 1")
plt.ylabel("t-SNE dim 2")
plt.legend(markerscale=3, loc='best', fontsize=6)
plt.tight_layout()
plt.show()
