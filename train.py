"""
Train Script for SGRL + Granule Diffusion (CUDA-ready, CSV logging)
- 粒球扩散（球图 K 步扩散 + 回写）
- 球级散射（RSM 升维）
- 球级 InfoNCE（两视图球对齐）
- 多 trial 训练
- 结果保存到 results/<DATASET>_summary.csv
"""

import os, time, csv, warnings, argparse
import numpy as np
import torch
import torch_geometric
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder, normalize
from rich.progress import Progress, BarColumn, TimeRemainingColumn, TimeElapsedColumn, TextColumn

from data import load_dataset
from models import Conv, Online, Target
from gb_utils import (
    granule_diffuse_and_write,
    build_granules,
    ball_scatter_loss,
    jaccard_between_balls,
    hungarian_matching,
    ball_infonce,
    compute_ball_centers
)

# =========================================================
# 工具函数
# =========================================================
def adj_norm(adj_t):
    deg = torch.sparse.sum(adj_t, dim=1).to_dense()
    deg_inv_sqrt = deg.pow(-0.5)
    adj_t = adj_t * deg_inv_sqrt.view(1, -1)
    adj_t = adj_t * deg_inv_sqrt.view(-1, 1)
    return adj_t


def fit_logistic_regression(X, y, data_random_seed=1, repeat=3):
    """多次划分下的 Logistic 回归分类精度平均值。"""
    one_hot_encoder = OneHotEncoder(categories='auto')
    y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(bool)
    X = normalize(X, norm='l2')
    rng = np.random.RandomState(data_random_seed)
    accuracies = []

    for _ in range(repeat):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=rng)
        logreg = LogisticRegression(solver='liblinear', max_iter=2000)
        c = 2.0 ** np.arange(-10, 11)
        cv = ShuffleSplit(n_splits=5, test_size=0.5)
        clf = GridSearchCV(
            estimator=OneVsRestClassifier(logreg),
            param_grid=dict(estimator__C=c),
            n_jobs=1, cv=cv, verbose=0
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(bool)
        accuracies.append(metrics.accuracy_score(y_test, y_pred))

    return accuracies


# =========================================================
# 在线网络训练（含粒球模块）
# =========================================================
def train_online(online, optimizer, data, device, epoch, args):
    """单 epoch 的 online 模块训练过程（含粒球扩散与球级loss）。"""
    online.train()
    h, h_pred, h_target = online(data.x, data.edge_index)

    if args.use_gb and (epoch % args.gb_rebuild_every == 0):
        with torch.no_grad():
            # 1) 构球 + 球图扩散 + 回写
            z_new, gb_sizes, H_ball, GB_node_list = granule_diffuse_and_write(
                node_embed=h, edge_index=data.edge_index,
                quity=args.gb_quity, sim=args.gb_sim,
                alpha_write=args.gb_alpha,
                beta=args.gb_beta, K=args.gb_K,
                w_mode=args.gb_w_mode, knn=args.gb_knn
            )
            # 粒球统计
            num_balls = len(gb_sizes)
            avg_size = sum(gb_sizes) / max(1, num_balls)
            min_size = min(gb_sizes) if gb_sizes else 0
            max_size = max(gb_sizes) if gb_sizes else 0
            os.makedirs('granular_count', exist_ok=True)
            with open(os.path.join('granular_count', f"{args.dataset_name}.txt"), 'a', encoding='utf-8') as f_log:
                f_log.write(
                    f"epoch={epoch:04d}, count={num_balls}, avg_size={avg_size:.1f}, "
                    f"min={min_size}, max={max_size}\n"
                )

        # 新表示进入 predictor
        h_pred = online.predictor(z_new)

        # 2) 球级散射（RSM 升维）
        if args.ball_loss_weight > 0 and (H_ball is not None) and (H_ball.size(0) > 1):
            neighbor_mask = None  # 如需只对邻近球生效，可传入球图邻接掩码
            loss_ball_scatter = ball_scatter_loss(
                H_ball,
                angle_thresh_deg=args.ball_angle_thresh,
                neighbor_mask=neighbor_mask,
                tau_u=args.ball_uniform_tau
            )
        else:
            loss_ball_scatter = torch.tensor(0.0, device=device)

        # 3) 两视图球对齐 + 球级 InfoNCE
        if args.ball_infonce_weight > 0:
            with torch.no_grad():
                GB2_node_list, _, _ = build_granules(h_target, data.edge_index, quity=args.gb_quity, sim=args.gb_sim)
                J = jaccard_between_balls(GB_node_list, GB2_node_list).to(device)
                pairs = hungarian_matching(J)
                H_target_ball = compute_ball_centers(h_target, GB2_node_list)
            loss_ball_infonce = ball_infonce(H_ball, H_target_ball, pairs, temp=args.ball_infonce_temp)
        else:
            loss_ball_infonce = torch.tensor(0.0, device=device)
    else:
        loss_ball_scatter = torch.tensor(0.0, device=device)
        loss_ball_infonce = torch.tensor(0.0, device=device)

    # === Loss 合成 ===
    loss_node = online.get_loss(h_pred, h_target.detach())
    loss = loss_node \
         + args.ball_loss_weight * loss_ball_scatter \
         + args.ball_infonce_weight * loss_ball_infonce

    loss.backward()
    optimizer.step()
    online.update_target_encoder()

    sim_mean = torch.nn.functional.cosine_similarity(h_pred, h_target, dim=-1).mean().item()
    return loss.item(), sim_mean


def train_target(target, optimizer, data):
    """Target 网络训练一步。"""
    target.train()
    h_target = target(data.x, data.edge_index)
    loss = target.get_loss(h_target)
    loss.backward()
    optimizer.step()
    return loss.item()


# =========================================================
# 主流程（含 CSV 保存）
# =========================================================
def run(args):
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 目录
    os.makedirs('results', exist_ok=True)
    os.makedirs(os.path.dirname(args.log_dir), exist_ok=True) if args.log_dir and os.path.dirname(args.log_dir) else None

    # 记录 args（可选）
    if args.log_dir:
        with open(args.log_dir, 'a', encoding='utf-8') as f:
            f.write(str(args) + '\n\n')

    torch_geometric.seed.seed_everything(args.seed)

    # === 多 trial 训练 ===
    csv_path = f"results/{args.dataset_name}_summary.csv"
    write_header = not os.path.exists(csv_path)

    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow([
                'trial', 'dataset', 'best_online_loss', 'best_target_loss',
                'clf_mean', 'clf_var', 'num_epochs', 'hidden_dim',
                'use_gb', 'gb_quity', 'gb_sim', 'gb_alpha',
                'gb_beta', 'gb_K', 'gb_w_mode', 'gb_knn',
                'gb_rebuild_every',
                'ball_loss_weight', 'ball_angle_thresh', 'ball_uniform_tau',
                'ball_infonce_weight', 'ball_infonce_temp',
                'seed'
            ])

        for trial in range(1, args.trials + 1):
            # 数据
            dataset = load_dataset(args.dataset_name, args.data_dir)
            data = dataset[0].to(device)

            # 带自环的归一化邻接（稀疏）
            nb_nodes = data.x.size(0)
            self_loop = torch.arange(nb_nodes, device='cpu').unsqueeze(0)
            self_loop = torch.cat([self_loop, self_loop], dim=0)
            edge_index_cpu = data.edge_index.detach().cpu()
            slsp_idx = torch.cat([edge_index_cpu, self_loop], dim=1)
            slsp_adj = torch.sparse.FloatTensor(
                slsp_idx.long(),
                torch.ones(slsp_idx.size(1)),
                torch.Size([nb_nodes, nb_nodes])
            )
            slsp_adj = adj_norm(slsp_adj).to(device)

            # 模型
            activation = torch.nn.PReLU()
            online_conv = Conv(data.x.size(1), args.hidden_dim, args.hidden_dim, activation, args.num_layers)
            target_conv = Conv(data.x.size(1), args.hidden_dim, args.hidden_dim, activation, args.num_layers)

            online_model = Online(online_conv, target_conv, args.hidden_dim, slsp_adj, args.num_hop, args.momentum).to(device)
            target_model = Target(target_conv).to(device)

            online_optimizer = torch.optim.Adam(online_model.parameters(), lr=args.e1_lr)
            target_optimizer = torch.optim.Adam(target_model.parameters(), lr=args.e2_lr)

            best_online_loss, best_target_loss = float('inf'), float('inf')

            # 进度条
            with Progress(
                TextColumn(f"[bold blue]{args.dataset_name} | Trial {trial}/{args.trials}"),
                BarColumn(bar_width=40),
                TimeElapsedColumn(),
                TimeRemainingColumn()
            ) as progress:
                task = progress.add_task("[green]Training", total=args.num_epochs)

                for epoch in range(args.num_epochs):
                    online_optimizer.zero_grad(set_to_none=True)
                    target_optimizer.zero_grad(set_to_none=True)

                    online_loss, sim_mean = train_online(online_model, online_optimizer, data, device, epoch, args)
                    target_loss = train_target(target_model, target_optimizer, data)

                    best_online_loss = min(best_online_loss, online_loss)
                    best_target_loss = min(best_target_loss, target_loss)

                    if (epoch + 1) % args.log_every == 0 and args.log_dir:
                        with open(args.log_dir, 'a', encoding='utf-8') as f:
                            f.write(f"trial={trial:02d}, epoch={epoch+1:04d}, loss={online_loss:.4f}, sim={sim_mean:.4f}\n")

                    progress.update(task, advance=1)

            # 评估（线性探测）
            online_model.eval()
            with torch.no_grad():
                or_embeds, pr_embeds = online_model.embed(data.x, data.edge_index, slsp_adj, args.num_hop)
                embeds = (or_embeds + pr_embeds).cpu().numpy()
            scores = fit_logistic_regression(embeds, data.y.cpu().numpy())
            clf_mean, clf_var = float(np.mean(scores)), float(np.var(scores))

            # 写 CSV（每个 trial 一行）
            writer.writerow([
                trial, args.dataset_name, best_online_loss, best_target_loss,
                clf_mean, clf_var, args.num_epochs, args.hidden_dim,
                int(args.use_gb), args.gb_quity, args.gb_sim, args.gb_alpha,
                args.gb_beta, args.gb_K, args.gb_w_mode, args.gb_knn,
                args.gb_rebuild_every,
                args.ball_loss_weight, args.ball_angle_thresh, args.ball_uniform_tau,
                args.ball_infonce_weight, args.ball_infonce_temp,
                args.seed
            ])
            csvfile.flush()  # 立刻落盘，方便中途分析

            print(f"[TRIAL {trial}] ACC = {clf_mean:.4f} ± {clf_var:.6f}")

    print(f"\n[SAVED] {csv_path}")
    print("[DONE] All trials finished.")
    return


# =========================================================
# 参数
# =========================================================
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser("SGRL + Granule Diffusion (CUDA-ready, CSV logging)")

    # 基础
    parser.add_argument('--dataset_name', type=str, default='Computers')
    parser.add_argument('--data_dir', type=str, default='../../datasets')
    parser.add_argument('--log_dir', type=str, default='./logs/log_Computers_cuda.txt')
    parser.add_argument('--e1_lr', type=float, default=1e-4)
    parser.add_argument('--e2_lr', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--num_epochs', type=int, default=700)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_hop', type=int, default=1)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--gb_rebuild_every', type=int, default=50)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--seed', type=int, default=66666)
    parser.add_argument('--device', type=str, default='cuda')

    # 粒球开关与基础
    parser.add_argument('--use_gb', action='store_true')
    parser.add_argument('--gb_quity', type=str, default='detach', choices=['homo', 'detach', 'edges', 'deg'])
    parser.add_argument('--gb_sim', type=str, default='dot', choices=['dot', 'cos', 'per'])
    parser.add_argument('--gb_alpha', type=float, default=0.6)

    # 粒球扩散参数
    parser.add_argument('--gb_beta', type=float, default=0.2)
    parser.add_argument('--gb_K', type=int, default=10)
    parser.add_argument('--gb_w_mode', type=str, default='topo+center', choices=['topo', 'center', 'topo+center'])
    parser.add_argument('--gb_knn', type=int, default=10)

    # 球级 Loss
    parser.add_argument('--ball_loss_weight', type=float, default=0.05)
    parser.add_argument('--ball_angle_thresh', type=float, default=15.0)
    parser.add_argument('--ball_uniform_tau', type=float, default=0.1)
    parser.add_argument('--ball_infonce_weight', type=float, default=0.02)
    parser.add_argument('--ball_infonce_temp', type=float, default=0.2)

    args = parser.parse_args()
    run(args)
