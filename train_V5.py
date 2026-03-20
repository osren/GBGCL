import numpy as np
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Amazon, Coauthor, WebKB, Actor, WikipediaNetwork
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder, normalize
from data import load_dataset
from models import Conv, Online, Target
import torch
import random
import torch_geometric
import copy
import time
# from tqdm import tqdm  # replaced by rich progress
from torch_geometric.utils import train_test_split_edges
import warnings
import argparse
import sys
import os
import csv
from gb_utils import build_granules_and_rewrite

# === NEW: rich progress bar ===
from rich.progress import Progress, BarColumn, TimeRemainingColumn, TimeElapsedColumn, TextColumn


def adj_norm(adj_t):
    deg = torch.sparse.sum(adj_t, dim=1).to_dense()
    deg_inv_sqrt = deg.pow(-0.5)
    adj_t = adj_t * deg_inv_sqrt.view(1, -1)
    adj_t = adj_t * deg_inv_sqrt.view(-1, 1)
    return adj_t


def fit_logistic_regression(X, y, data_random_seed=1, repeat=3):
    one_hot_encoder = OneHotEncoder(categories='auto')

    y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(bool)

    X = normalize(X, norm='l2')
    rng = np.random.RandomState(data_random_seed)
    accuracies = []
    for _ in range(repeat):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=rng)

        logreg = LogisticRegression(solver='liblinear')
        c = 2.0 ** np.arange(-10, 11)
        cv = ShuffleSplit(n_splits=5, test_size=0.5)
        clf = GridSearchCV(estimator=OneVsRestClassifier(logreg), param_grid=dict(estimator__C=c),
                           n_jobs=5, cv=cv, verbose=0)
        clf.fit(X_train, y_train)

        y_pred = clf.predict_proba(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(bool)

        test_acc = metrics.accuracy_score(y_test, y_pred)
        accuracies.append(test_acc)
    return accuracies


def train_online(online, optimizer, data, use_gb=False, gb_quity='homo', gb_sim='dot', gb_alpha=0.5,
                 epoch=1, dataset_name='default', gb_rebuild_every=10):
    online.train()
    # 原始 forward
    h, h_pred, h_target = online(data.x, data.edge_index)
    E = gb_rebuild_every
    # 只在每 E 个 epoch 重建粒球
    did_rebuild = False
    if use_gb and (epoch % E == 0):
        # 在 loss 前插入“粒球 → 回写”
        with torch.no_grad():
            # 用 online 的“融合后表征” h 来做粒球（也可以只用 or_embeds，看你偏好）
            z_new, gb_sizes = build_granules_and_rewrite(
                node_embed=h, edge_index=data.edge_index,
                quity=gb_quity, sim=gb_sim, alpha_write=gb_alpha
            )
            # === 粒球统计信息写入文件（不打印） ===
            num_balls = len(gb_sizes)
            avg_size = sum(gb_sizes) / max(1, num_balls)
            min_size = min(gb_sizes) if gb_sizes else 0
            max_size = max(gb_sizes) if gb_sizes else 0
            # Write to 'granular_count' (original) and 'granluar_count' (alt spelling) for safety
            for d in ['granular_count']:
                os.makedirs(d, exist_ok=True)
                log_path = os.path.join(d, f"{dataset_name}.txt")
                with open(log_path, 'a', encoding='utf-8') as f_log:
                    f_log.write(
                        f"epoch={epoch:04d}, count={num_balls}, avg_size={avg_size:.1f}, min={min_size}, max={max_size}\n"
                    )
        # 粒球回写后的表征再过 predictor
        h_pred = online.predictor(z_new)
        did_rebuild = True
    else:
        # 不重建时，直接用 h_pred（或缓存上次 z_new）
        pass

    loss = online.get_loss(h_pred, h_target.detach())
    loss.backward()
    optimizer.step()
    online.update_target_encoder()
    # 监控相似度均值（不打印，返回给上层做条件日志）
    sim_mean = torch.nn.functional.cosine_similarity(h_pred, h_target, dim=-1).mean().item()
    return loss.item(), sim_mean, did_rebuild


def train_target(target, optimizier, data):
    target.train()
    h_target = target(data.x, data.edge_index)
    loss = target.get_loss(h_target)
    loss.backward()
    optimizier.step()
    return loss.item()


def run(args):
    log_dir = args.log_dir
    path = args.data_dir
    # 确保权重保存目录存在
    os.makedirs('pkl/pkl_online', exist_ok=True)
    os.makedirs('pkl/pkl_target', exist_ok=True)

    # 记录运行参数
    os.makedirs(os.path.dirname(log_dir), exist_ok=True) if os.path.dirname(log_dir) else None
    with open(log_dir, 'a', encoding='utf-8') as f:
        f.write(str(args))
        f.write('\n\n\n')

    trials = args.trials
    torch_geometric.seed.seed_everything(args.seed)

    # 跨 trial 汇总
    all_trial_summaries = []

    for trial in range(trials):
        e1_lr = args.e1_lr
        e2_lr = args.e2_lr
        weight_decay = args.weight_decay
        hidden_dim = args.hidden_dim
        activation = torch.nn.PReLU()
        num_layers = args.num_layers
        num_epochs = args.num_epochs
        momentum = args.momentum
        dataset_name = args.dataset_name

        dataset = load_dataset(dataset_name, path)
        data = dataset[0]
        # device = torch.device('cuda')
        device = torch.device('cpu')
        data = data.to(device)

        num_hop = args.num_hop
        nb_nodes = data.x.size()[0]

        self_loop_for_adj = torch.Tensor([i for i in range(nb_nodes)]).unsqueeze(0)
        self_loop_for_adj = torch.concat([self_loop_for_adj, self_loop_for_adj], dim=0)
        slsp_adj = torch.concat([data.edge_index.to('cpu'), self_loop_for_adj], dim=1)
        slsp_adj = torch.sparse.FloatTensor(slsp_adj.long(), torch.ones(slsp_adj.size()[1]),
                                            torch.Size([nb_nodes, nb_nodes]))

        slsp_adj = adj_norm(slsp_adj).to(device)

        online_conv = Conv(data.x.size()[1], hidden_dim, hidden_dim, activation, num_layers).to(device)
        target_conv = Conv(data.x.size()[1], hidden_dim, hidden_dim, activation, num_layers).to(device)

        online_model = Online(online_conv, target_conv, hidden_dim, slsp_adj, num_hop, momentum).to(device)
        target_model = Target(target_conv).to(device)

        online_optimizer = torch.optim.Adam(online_model.parameters(), lr=e1_lr)
        target_optimizer = torch.optim.Adam(target_model.parameters(), lr=e2_lr)

        best_online_loss = 1e9
        best_target_loss = 1e9

        tag = dataset_name + '_' + str(time.time())

        # === Rich Progress Bar per trial ===
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•", TimeElapsedColumn(),
            "•", TimeRemainingColumn(),
        ) as progress:

            # Note: trials_task 是可选的，这里用于显示当前 trial 进度（对多 trial 也友好）
            trials_task = progress.add_task(f"[cyan]Trials {trial+1}/{trials}", total=1)
            epoch_task = progress.add_task("[green]Training...", total=num_epochs)

            for epoch in range(num_epochs):
                online_optimizer.zero_grad()
                target_optimizer.zero_grad()

                online_loss, sim_mean, did_rebuild = train_online(
                    online_model, online_optimizer, data,
                    use_gb=args.use_gb, gb_quity=args.gb_quity, gb_sim=args.gb_sim, gb_alpha=args.gb_alpha,
                    epoch=epoch, dataset_name=dataset_name, gb_rebuild_every=args.gb_rebuild_every
                )

                # 判断显著提升
                significant_improve = False
                if online_loss < best_online_loss:
                    abs_improve = best_online_loss - online_loss  # 绝对改善量
                    best_online_loss = online_loss
                    torch.save(online_model.state_dict(), 'pkl/pkl_online/best_online' + tag + '.pkl')
                    if abs_improve >= args.imp_thresh:
                        significant_improve = True

                # target 训练
                target_loss = train_target(target_model, target_optimizer, data)
                if target_loss < best_target_loss:
                    best_target_loss = target_loss
                    torch.save(target_model.state_dict(), 'pkl/pkl_target/best_target' + tag + '.pkl')
                    target_model.load_state_dict(torch.load('pkl/pkl_target/best_target' + tag + '.pkl'))

                # 条件日志：减少冗余
                if ((epoch + 1) % args.log_every == 0) or did_rebuild or significant_improve:
                    tags = []
                    if did_rebuild:
                        tags.append("[yellow]Granule Rebuild")
                    if significant_improve:
                        tags.append("[magenta]Significant Loss ↓")
                    tag_str = " | ".join(tags) if tags else "log"
                    progress.log(
                        f"[white]Trial {trial+1}/{trials} • Epoch {epoch+1}/{num_epochs} "
                        f"• loss={online_loss:.4f} • sim={sim_mean:.4f} • {tag_str}"
                    )

                # 进度条最小更新 & 描述更新
                if (epoch + 1) % args.log_every == 0 or epoch == 0 or epoch == num_epochs - 1:
                    progress.update(
                        epoch_task,
                        advance=args.log_every if (epoch + 1) % args.log_every == 0 else 1,
                        description=f"[green]Epoch {epoch+1}/{num_epochs} | best_loss={best_online_loss:.4f}"
                    )
                else:
                    progress.update(epoch_task, advance=1)

            # ============ Trial-level summary ============
            online_model.load_state_dict(torch.load('pkl/pkl_online/best_online' + tag + '.pkl'))
            online_model.eval()
            or_embeds, pr_embeds = online_model.embed(data.x, data.edge_index, slsp_adj, num_hop)
            embeds = or_embeds + pr_embeds
            scores = fit_logistic_regression(embeds.detach().cpu().numpy(), data.y.cpu().numpy())
            m = np.mean(scores)
            n = np.var(scores)

            trial_summary = {
                "trial": trial + 1,
                "dataset": dataset_name,
                "best_online_loss": float(best_online_loss),
                "best_target_loss": float(best_target_loss),
                "clf_mean": float(m),
                "clf_var": float(n),
            }
            all_trial_summaries.append(trial_summary)
            progress.log(
                f"[cyan]Trial {trial+1} summary → best_loss={best_online_loss:.4f}, "
                f"clf_mean={m:.4f}, clf_var={n:.6f}"
            )

            with open(log_dir, 'a', encoding='utf-8') as f:
                f.write('sgrl_mean: ' + str(m)[0:7] + ' sgrl_std: ' + str(n))
                f.write('\n')

            # 完成当前 trial 任务
            progress.update(trials_task, advance=1)

            # === Append trial summary to results/<dataset>_summary.csv ===
            results_dir = 'results'
            os.makedirs(results_dir, exist_ok=True)
            results_path = os.path.join(results_dir, f"{dataset_name}_summary.csv")
            write_header = not os.path.exists(results_path)
            with open(results_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                if write_header:
                    writer.writerow([
                        'trial', 'dataset', 'best_online_loss', 'best_target_loss',
                        'clf_mean', 'clf_var', 'num_epochs', 'hidden_dim', 'num_hop',
                        'use_gb', 'gb_quity', 'gb_sim', 'gb_alpha',
                        'log_every', 'imp_thresh', 'gb_rebuild_every'
                    ])
                writer.writerow([
                    trial_summary['trial'], trial_summary['dataset'],
                    trial_summary['best_online_loss'], trial_summary['best_target_loss'],
                    trial_summary['clf_mean'], trial_summary['clf_var'],
                    args.num_epochs, args.hidden_dim, args.num_hop,
                    int(args.use_gb), args.gb_quity, args.gb_sim, args.gb_alpha,
                    args.log_every, args.imp_thresh, args.gb_rebuild_every
                ])


    # ===== After all trials: aggregate & show overall stats =====
    if len(all_trial_summaries) > 0:
        overall_mean = np.mean([t["clf_mean"] for t in all_trial_summaries])
        overall_var = np.var([t["clf_mean"] for t in all_trial_summaries])
        best_trial = max(all_trial_summaries, key=lambda t: t["clf_mean"])
        worst_trial = min(all_trial_summaries, key=lambda t: t["clf_mean"])

        print("==== Overall Summary ====")
        print(f"Trials: {len(all_trial_summaries)} | Dataset: {all_trial_summaries[0]['dataset']}")
        print(f"Classifier mean (across trials): {overall_mean:.4f} ± {overall_var:.6f}")
        print(f"Best trial: #{best_trial['trial']} | mean={best_trial['clf_mean']:.4f} | loss={best_trial['best_online_loss']:.4f}")
        print(f"Worst trial: #{worst_trial['trial']} | mean={worst_trial['clf_mean']:.4f} | loss={worst_trial['best_online_loss']:.4f}")


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser('SGRL')
    parser.add_argument('--dataset_name', type=str, default='Photo', help='dataset_name')
    parser.add_argument('--data_dir', type=str, default='../../datasets', help='data_dir')
    parser.add_argument('--log_dir', type=str, default='./log/log_Photo', help='log_dir')
    parser.add_argument('--e1_lr', type=float, default=0.001, help='online_learning_rate')
    parser.add_argument('--e2_lr', type=float, default=0.001, help='target_learning_rate')
    parser.add_argument('--momentum', type=float, default=0.99, help='EMA')
    parser.add_argument('--weight_decay', type=float, default=0., help='weight_decay')
    parser.add_argument('--num_epochs', type=int, default=700, help='num_epochs')
    parser.add_argument('--seed', type=int, default=66666, help='seed')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='hidden_dim')
    parser.add_argument('--num_layers', type=int, default=1, help='num_layers')
    parser.add_argument('--num_hop', type=int, default=1, help='num_hop')
    parser.add_argument('--trials', type=int, default=20, help='trials')
    parser.add_argument('--log_every', type=int, default=100, help='每隔多少个 epoch 输出一次详细日志')
    parser.add_argument('--imp_thresh', type=float, default=0.5, help='相对损失改善阈值(例如 0.05=5%) 用于高亮')
    parser.add_argument('--gb_rebuild_every', type=int, default=100, help='粒球重建的周期 E（每 E 个 epoch 重建一次）')
    # 新增粒球控制参数
    parser.add_argument('--use_gb', action='store_true', help='是否启用粒球回写(最小可运行版)')
    parser.add_argument('--gb_quity', type=str, default='homo', choices=['homo', 'detach', 'edges', 'deg'],
                        help='粒球质量判据(见 granular.py )')
    parser.add_argument('--gb_sim', type=str, default='dot', choices=['dot', 'cos', 'per'],
                        help='粒球分配的相似度')
    parser.add_argument('--gb_alpha', type=float, default=0.5, help='回写强度: 0=全用球均值, 1=不用球')

    args = parser.parse_args()
    run(args)
