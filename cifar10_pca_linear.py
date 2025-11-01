#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIFAR-10 → PCA 降维 → 线性分类器（Logistic 回归，SGDClassifier 实现）
输出各维度的测试集准确率，并绘制 Accuracy vs Dimension 曲线。

用法示例：
  python cifar10_pca_linear.py --data-dir ./data --download \
    --dims 2 5 10 20 50 100 200 300 512 1024 --out-dir ./outputs

注意：
- PCA 仅在训练集上拟合；测试集只做变换与评估（符合作业规约）。
- 若服务器无外网，请先把 CIFAR-10 放到 --data-dir 指定目录，并去掉 --download。
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 服务器无显示环境时也能保存图
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

import torch
from torchvision import datasets, transforms
from datetime import datetime

def load_cifar10_numpy(data_dir: str, download: bool):
    """加载 CIFAR-10，返回 (X_train, y_train, X_test, y_test)，X 为 [N, 3072] 的 float32。"""
    tfm = transforms.ToTensor()  # [0,1] float32
    train_set = datasets.CIFAR10(root=data_dir, train=True, transform=tfm, download=download)
    test_set  = datasets.CIFAR10(root=data_dir, train=False, transform=tfm, download=download)

    def as_numpy(ds):
        xs = []
        ys = []
        for img, lbl in ds:
            # img: [3, 32, 32] torch.float32 in [0,1]
            xs.append(img.view(-1).numpy())  # -> [3072]
            ys.append(lbl)
        return np.stack(xs, axis=0).astype(np.float32), np.array(ys, dtype=np.int64)

    X_train, y_train = as_numpy(train_set)
    X_test,  y_test  = as_numpy(test_set)
    return X_train, y_train, X_test, y_test

def parse_args():
    parser = argparse.ArgumentParser(
        description="CIFAR-10 dimensionality reduction vs classifier comparisons"
    )
    parser.add_argument("--data-dir", type=str, default="./data", help="CIFAR-10 data directory")
    parser.add_argument("--download", action="store_true", help="Download CIFAR-10 if missing")
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=[2, 5, 10, 20, 50, 100, 200, 300, 512, 1024, 2048, 4096],
        help="PCA component counts to evaluate",
    )
    parser.add_argument(
        "--lda-dims",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5, 6, 7, 8, 9],
        help="LDA component counts to evaluate",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Optional: limit number of training samples (debug/speed)",
    )
    parser.add_argument(
        "--max-test",
        type=int,
        default=None,
        help="Optional: limit number of test samples (debug/speed)",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--out-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument(
        "--pca-whiten",
        action="store_true",
        help="Enable PCA whitening (may improve linear separability)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="SGDClassifier max epochs")
    parser.add_argument("--alpha", type=float, default=1e-4, help="SGDClassifier L2 strength")
    parser.add_argument(
        "--reducers",
        type=str,
        nargs="+",
        choices=["pca", "lda"],
        default=["pca", "lda"],
        help="Dimensionality reduction methods to evaluate",
    )
    parser.add_argument(
        "--classifiers",
        type=str,
        nargs="+",
        choices=["linear", "knn"],
        default=["linear", "knn"],
        help="Classifiers to evaluate",
    )
    parser.add_argument("--knn-k", type=int, default=5, help="k-NN: neighborhood size k")
    parser.add_argument(
        "--knn-weights",
        type=str,
        default="uniform",
        choices=["uniform", "distance"],
        help="k-NN weighting strategy",
    )
    parser.add_argument(
        "--knn-metric",
        type=str,
        default="minkowski",
        help="k-NN distance metric (e.g. euclidean, manhattan)",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.random_state)

    print(f"[{datetime.now()}] Loading CIFAR-10 ...")
    X_train, y_train, X_test, y_test = load_cifar10_numpy(args.data_dir, args.download)

    if args.max_train is not None:
        X_train, y_train = X_train[:args.max_train], y_train[:args.max_train]
    if args.max_test is not None:
        X_test, y_test = X_test[:args.max_test], y_test[:args.max_test]

    n_train, n_feat = X_train.shape
    n_test = X_test.shape[0]
    dim_cap = min(n_feat, n_train)  # PCA dimension upper bound limited by features and samples
    pca_dims = sorted({d for d in args.dims if 1 <= d <= dim_cap})
    if not pca_dims:
        raise ValueError(f"No valid PCA dims; got {args.dims}, but cap={dim_cap} with n_feat={n_feat}, n_train={n_train}")
    n_classes = len(np.unique(y_train))
    lda_cap = min(dim_cap, n_classes - 1)
    lda_dims = sorted({d for d in args.lda_dims if 1 <= d <= lda_cap})

    print(f"Train: {X_train.shape}, Test: {X_test.shape}, PCA dim cap: {dim_cap}")
    print(f"PCA dims: {pca_dims}")
    if lda_dims:
        print(f"LDA dims (cap {lda_cap}): {lda_dims}")
    else:
        print(f"LDA dims (cap {lda_cap}): [] -- requested {args.lda_dims}")
    # 线性分类器：logistic 回归（SGD），适合大样本/高维
    def make_linear_clf():
        return SGDClassifier(
            loss="log_loss", penalty="l2",
            alpha=args.alpha, max_iter=args.epochs, tol=1e-3,
            early_stopping=True, n_iter_no_change=5,
            learning_rate="optimal", random_state=args.random_state,
            n_jobs=-1
        )

    results = []
    for k in pca_dims:
        print(f"[{datetime.now()}] === Dim = {k} ===")
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("pca", PCA(n_components=k, svd_solver="randomized", whiten=args.pca_whiten, random_state=args.random_state)),
            ("clf", make_linear_clf())
        ])

        # 仅在训练集上拟合
        pipe.fit(X_train, y_train)

        # 测试评估
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Dim={k:4d}  Test Acc = {acc*100:.2f}%")
        results.append((k, acc))

    # 保存 CSV
    csv_path = os.path.join(args.out_dir, "accuracy_vs_dim.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("dimension,accuracy\n")
        for k, acc in results:
            f.write(f"{k},{acc:.6f}\n")
    print(f"Saved CSV -> {csv_path}")

    # 绘图
    plt.figure(figsize=(7, 5))
    xs = [k for k, _ in results]
    ys = [acc for _, acc in results]
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Dimension (PCA components)")
    plt.ylabel("Test Accuracy")
    plt.title("CIFAR-10: Accuracy vs PCA Dimension (Linear Classifier)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, "accuracy_vs_dim.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Saved Figure -> {fig_path}")

    # 控制台摘要
    best_k, best_acc = max(results, key=lambda t: t[1])
    print(f"\nBest Dim = {best_k}  |  Best Test Acc = {best_acc*100:.2f}%")

    # 追加：运行其余组合，并汇总所有方法结果与图表
    # 保留当前 PCA+Linear 的结果
    all_results = {("pca", "linear"): results.copy()}

    # k-NN 分类器
    def make_knn_clf():
        return KNeighborsClassifier(n_neighbors=args.knn_k, weights=args.knn_weights, metric=args.knn_metric, n_jobs=-1)

    # 构建流水线辅助
    def build_pipeline(reducer: str, dim_k: int, classifier: str) -> Pipeline:
        steps = [("scaler", StandardScaler(with_mean=True, with_std=True))]
        if reducer == "pca":
            steps.append(("pca", PCA(n_components=dim_k, svd_solver="randomized", whiten=args.pca_whiten, random_state=args.random_state)))
        elif reducer == "lda":
            steps.append(("lda", LDA(n_components=dim_k)))
        else:
            raise ValueError(f"Unknown reducer: {reducer}")

        if classifier == "linear":
            steps.append(("clf", make_linear_clf()))
        elif classifier == "knn":
            steps.append(("clf", make_knn_clf()))
        else:
            raise ValueError(f"Unknown classifier: {classifier}")
        return Pipeline(steps)

    # 计算每个 reducer 的有效维度
    pca_dims_valid = pca_dims
    lda_cap = min(dim_cap, n_classes - 1)
    lda_dims_valid = lda_dims

    # 需要执行的组合列表（若包含已有的 PCA+linear 则跳过重复训练）
    for reducer in args.reducers:
        if reducer == "pca":
            dims_valid = pca_dims
            cap_info = f"cap={dim_cap}"
        elif reducer == "lda":
            dims_valid = lda_dims
            cap_info = f"cap={lda_cap} (<= classes-1)"
        else:
            print(f"[WARN] Unknown reducer={reducer}; skip.")
            continue
        if not dims_valid:
            print(f"[WARN] No valid dims for reducer={reducer}; skip.")
            continue
        for classifier in args.classifiers:
            key = (reducer, classifier)
            if key == ("pca", "linear"):
                continue  # 已经完成
            all_results[key] = []
            print(f"\n[{datetime.now()}] >>> Reducer={reducer.upper()} | Classifier={classifier.upper()} | dims={dims_valid} ({cap_info})")
            for k in dims_valid:
                print(f"[{datetime.now()}] --- Dim = {k} ---")
                pipe = build_pipeline(reducer, k, classifier)
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                print(f"{reducer}+{classifier} | Dim={k:4d}  Test Acc = {acc*100:.2f}%")
                all_results[key].append((k, acc))

    # 导出每个组合 CSV 与单独图
    def combo_name(reducer, classifier):
        return f"{reducer}_{classifier}"

    colors = {
        ("pca", "linear"): "tab:blue",
        ("pca", "knn"): "tab:orange",
        ("lda", "linear"): "tab:green",
        ("lda", "knn"): "tab:red",
    }

    for key, res in all_results.items():
        if not res:
            continue
        reducer, classifier = key
        tag = combo_name(reducer, classifier)
        csv_path = os.path.join(args.out_dir, f"accuracy_vs_dim_{tag}.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("dimension,accuracy\n")
            for k, acc in res:
                f.write(f"{k},{acc:.6f}\n")
        print(f"Saved CSV -> {csv_path}")

        plt.figure(figsize=(7, 5))
        xs = [k for k, _ in res]
        ys = [acc for _, acc in res]
        plt.plot(xs, ys, marker="o", color=colors.get(key, None))
        plt.xlabel("Dimension")
        plt.ylabel("Test Accuracy")
        title_reducer = reducer.upper()
        title_clf = "Linear" if classifier == "linear" else "k-NN"
        plt.title(f"CIFAR-10: Accuracy vs Dim ({title_reducer} + {title_clf})")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig_path = os.path.join(args.out_dir, f"accuracy_vs_dim_{tag}.png")
        plt.savefig(fig_path, dpi=150)
        print(f"Saved Figure -> {fig_path}")

    # 聚合图：同一降维方法内部的对比
    for reducer in ["pca", "lda"]:
        group_keys = [key for key, res in all_results.items() if key[0] == reducer and res]
        if not group_keys:
            continue
        plt.figure(figsize=(8, 6))
        for key in group_keys:
            res = all_results[key]
            xs = [k for k, _ in res]
            ys = [acc for _, acc in res]
            label = f"{key[0].upper()} + {'Linear' if key[1]=='linear' else 'k-NN'}"
            plt.plot(xs, ys, marker="o", label=label, color=colors.get(key, None))
        plt.xlabel("Dimension")
        plt.ylabel("Test Accuracy")
        title = "PCA" if reducer == "pca" else "LDA"
        plt.title(f"CIFAR-10: {title} Accuracy vs Dimension (Classifiers)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        fig_path = os.path.join(args.out_dir, f"accuracy_vs_dim_{reducer}_comparison.png")
        plt.savefig(fig_path, dpi=150)
        print(f"Saved Figure -> {fig_path}")

    # 每个组合最佳值
    for key, res in all_results.items():
        if not res:
            continue
        best_k, best_acc = max(res, key=lambda t: t[1])
        reducer, classifier = key
        print(f"Best [{reducer.upper()} + {'Linear' if classifier=='linear' else 'k-NN'}] -> Dim={best_k} | Acc={best_acc*100:.2f}%")

if __name__ == "__main__":
    main()
