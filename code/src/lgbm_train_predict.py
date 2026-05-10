"""
LightGBM 排序学习版本：训练 + 预测一体化脚本
用法：uv run python code/src/lgbm_train_predict.py
输出：output/result.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_CSV  = os.path.join(ROOT, 'data', 'train.csv')
OUTPUT_CSV = os.path.join(ROOT, 'output', 'result.csv')
MODEL_DIR  = os.path.join(ROOT, 'model', 'lgbm')
os.makedirs(MODEL_DIR, exist_ok=True)

SEQUENCE_DAYS       = 20
FUTURE_DAYS         = 5
TRAIN_RATIO         = 0.9
TOP_N               = 5
WEIGHT_CASH_RESERVE = 0.0001

LGBM_PARAMS = {
    'objective':         'lambdarank',
    'metric':            'ndcg',
    'ndcg_eval_at':      [5],
    'learning_rate':     0.05,
    'num_leaves':        63,
    'min_child_samples': 20,
    'feature_fraction':  0.8,
    'bagging_fraction':  0.8,
    'bagging_freq':      5,
    'lambda_l1':         0.1,
    'lambda_l2':         1.0,
    'verbose':           -1,
    'n_jobs':            -1,
}
NUM_BOOST_ROUND = 300
EARLY_STOPPING  = 30


def add_technical_features(df):
    close  = df['收盘'].values.astype(np.float64)
    open_  = df['开盘'].values.astype(np.float64)
    high   = df['最高'].values.astype(np.float64)
    low    = df['最低'].values.astype(np.float64)
    vol    = df['成交量'].values.astype(np.float64)
    feats  = {}

    for w in [5, 10, 20, 60]:
        ret = np.where(np.roll(close, w) != 0, close / (np.roll(close, w) + 1e-12) - 1, 0)
        ret[:w] = 0
        feats[f'mom_{w}'] = ret
        ma = pd.Series(close).rolling(w, min_periods=1).mean().values
        feats[f'ma_ratio_{w}'] = close / (ma + 1e-12) - 1

    log_ret = np.diff(np.log(close + 1e-12), prepend=np.log(close[0] + 1e-12))
    for w in [5, 10, 20]:
        feats[f'vol_{w}'] = pd.Series(log_ret).rolling(w, min_periods=1).std().values

    delta = pd.Series(close).diff()
    gain  = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    feats['rsi_14'] = (gain / (gain + loss + 1e-12)).values

    ema12  = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema26  = pd.Series(close).ewm(span=26, adjust=False).mean().values
    macd   = ema12 - ema26
    signal = pd.Series(macd).ewm(span=9, adjust=False).mean().values
    feats['macd']        = macd   / (close + 1e-12)
    feats['macd_signal'] = signal / (close + 1e-12)
    feats['macd_hist']   = (macd - signal) / (close + 1e-12)

    mid = pd.Series(close).rolling(20, min_periods=1).mean().values
    std = pd.Series(close).rolling(20, min_periods=1).std().values
    feats['boll_pos']   = (close - mid) / (std + 1e-12)
    feats['boll_width'] = std / (mid + 1e-12)

    feats['open_close_ratio'] = (close - open_) / (open_ + 1e-12)
    feats['high_low_ratio']   = (high - low)    / (close + 1e-12)
    feats['upper_shadow']     = (high - np.maximum(close, open_)) / (close + 1e-12)
    feats['lower_shadow']     = (np.minimum(close, open_) - low)  / (close + 1e-12)

    vol_ma5  = pd.Series(vol).rolling(5,  min_periods=1).mean().values
    vol_ma20 = pd.Series(vol).rolling(20, min_periods=1).mean().values
    feats['vol_ratio_5']  = vol / (vol_ma5  + 1e-12)
    feats['vol_ratio_20'] = vol / (vol_ma20 + 1e-12)

    if '换手率' in df.columns:
        turn = df['换手率'].values.astype(np.float64)
        feats['turnover']      = turn
        feats['turnover_ma5']  = pd.Series(turn).rolling(5,  min_periods=1).mean().values
        feats['turnover_ma20'] = pd.Series(turn).rolling(20, min_periods=1).mean().values

    return pd.concat([df, pd.DataFrame(feats, index=df.index)], axis=1)


def build_dataset(df, seq_days, future_days):
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['日期', '股票代码']).reset_index(drop=True)
    all_dates = sorted(df['日期'].unique())

    skip_cols = {'股票代码', '日期', '开盘', '收盘', '最高', '最低',
                 '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅'}
    feat_cols = [c for c in df.columns if c not in skip_cols]

    records = []
    for i, pred_date in enumerate(all_dates):
        if i < seq_days:
            continue
        future_idx = i + future_days
        if future_idx >= len(all_dates):
            break
        first_date = all_dates[i + 1]
        last_date  = all_dates[future_idx]

        day_feats  = df[df['日期'] == pred_date][['股票代码'] + feat_cols].copy()
        open_first = df[df['日期'] == first_date][['股票代码', '开盘']].rename(columns={'开盘': 'open_first'})
        open_last  = df[df['日期'] == last_date ][['股票代码', '开盘']].rename(columns={'开盘': 'open_last'})

        merged = day_feats.merge(open_first, on='股票代码', how='inner')
        merged = merged.merge(open_last,  on='股票代码', how='inner')
        merged['future_ret'] = (merged['open_last'] - merged['open_first']) / (merged['open_first'] + 1e-12)
        merged['pred_date']  = pred_date
        records.append(merged)

    if not records:
        raise ValueError("没有构建出任何训练样本。")

    full = pd.concat(records, ignore_index=True).dropna(subset=feat_cols + ['future_ret'])

    full['relevance'] = full.groupby('pred_date')['future_ret'].transform(
        lambda x: pd.qcut(x, q=min(10, len(x.unique())), labels=False, duplicates='drop').fillna(0).astype(int)
    )

    groups = full.groupby('pred_date').size().values
    X      = full[feat_cols].astype(np.float32)
    y      = full['relevance'].values.astype(np.int32)
    meta   = full[['pred_date', '股票代码', 'future_ret']].copy()
    return X, y, groups, meta, feat_cols


def main():
    print("=" * 50)
    print("LightGBM 股票排序模型")
    print("=" * 50)

    print("\n[1/4] 加载数据并计算技术指标...")
    raw = pd.read_csv(TRAIN_CSV, dtype={'股票代码': str})
    raw['日期'] = pd.to_datetime(raw['日期'])
    raw = raw.sort_values(['股票代码', '日期']).reset_index(drop=True)
    print(f"  {len(raw)} 行，{raw['股票代码'].nunique()} 只股票，"
          f"{raw['日期'].min().date()} ~ {raw['日期'].max().date()}")

    processed = []
    for code, grp in raw.groupby('股票代码'):
        processed.append(add_technical_features(grp.sort_values('日期').reset_index(drop=True)))
    df = pd.concat(processed, ignore_index=True)

    print("\n[2/4] 构建训练样本...")
    X, y, groups, meta, feat_cols = build_dataset(df, SEQUENCE_DAYS, FUTURE_DAYS)
    print(f"  特征数：{len(feat_cols)}，总样本：{len(X)}，预测日数：{len(groups)}")

    n_train_days = int(len(groups) * TRAIN_RATIO)
    train_size   = groups[:n_train_days].sum()
    X_train, X_val = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_val = y[:train_size],       y[train_size:]
    g_train, g_val = groups[:n_train_days], groups[n_train_days:]

    scaler     = RobustScaler()
    X_train_s  = scaler.fit_transform(X_train)
    X_val_s    = scaler.transform(X_val)

    print(f"\n[3/4] 训练 LightGBM（训练日={n_train_days}，验证日={len(g_val)}）...")
    dtrain = lgb.Dataset(X_train_s, label=y_train, group=g_train, free_raw_data=False)
    dval   = lgb.Dataset(X_val_s,   label=y_val,   group=g_val,   reference=dtrain, free_raw_data=False)

    model = lgb.train(
        LGBM_PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False), lgb.log_evaluation(50)],
    )
    print(f"  最佳迭代：{model.best_iteration}，验证 NDCG@5：{model.best_score['val']['ndcg@5']:.4f}")
    model.save_model(os.path.join(MODEL_DIR, 'lgbm_model.txt'))

    imp = pd.Series(model.feature_importance(importance_type='gain'), index=feat_cols).sort_values(ascending=False)
    print("\n  特征重要性 Top10：")
    for fname, val in imp.head(10).items():
        print(f"    {fname:<25} {val:.1f}")

    print("\n[4/4] 预测最新日期（含风险优化）...")
    latest_date = df['日期'].max()
    latest_day  = df[df['日期'] == latest_date].copy()
    X_latest    = scaler.transform(latest_day[feat_cols].astype(np.float32))
    latest_day  = latest_day.copy()
    latest_day['score'] = model.predict(X_latest)

    # ── Step1：候选股票（top5）─────────────────────────────
    candidates = latest_day.nlargest(TOP_N, 'score')[['股票代码', 'score']].reset_index(drop=True)
    top1_score = candidates['score'].iloc[0]

    # 全部候选股票参与融合优化，由优化器自动决定权重
    # （不手动过滤，低分股会因模型期望低而自动被压低权重）
    selected = candidates.reset_index(drop=True)

    codes = selected['股票代码'].tolist()
    scores_arr = selected['score'].values.astype(np.float64)
    n_sel = len(selected)

    print(f"\n  候选5只全部参与优化：")
    for _, row in candidates.iterrows():
        print(f"    {row['股票代码']}  score={row['score']:.4f}")

    # ── Step2：计算历史风险指标 ────────────────────────────
    # 用训练集最近60天的日收益率计算协方差矩阵和波动率
    LOOKBACK = 60
    recent_dates = sorted(df['日期'].unique())[-LOOKBACK:]
    hist = df[(df['日期'].isin(recent_dates)) & (df['股票代码'].isin(codes))].copy()
    ret_pivot = hist.pivot(index='日期', columns='股票代码', values='涨跌幅').fillna(0)
    ret_pivot = ret_pivot.reindex(columns=codes, fill_value=0)

    volatility = ret_pivot.std().values + 1e-8          # 日波动率
    cov_matrix = ret_pivot.cov().values                  # 协方差矩阵（未年化，已够用）
    sharpe_proxy = ret_pivot.mean().values / volatility  # 简化夏普（无风险利率=0）

    print(f"\n  风险指标（近{LOOKBACK}天）：")
    for i, c in enumerate(codes):
        print(f"    {c}  日波动率={volatility[i]:.4f}  夏普代理={sharpe_proxy[i]:.3f}")

    # ── Step3：融合权重优化 ──────────────────────────────────

    # ── 融合优化：最大夏普 + 风险平价约束 ──────────────────
    #
    # 目标函数：
    #   maximize  (w @ mu) / sqrt(w @ cov @ w)      ← 夏普：收益/风险
    #           - ALPHA * Σ(RC_i - RC_target)²      ← 风险平价：各股风险贡献均等惩罚
    #
    # 两个信号的角色：
    #   model_mu（模型分数）→ 决定押多少（分高多买）
    #   volatility/cov     → 决定怎么分散（波动大少买，相关性高少买）
    #
    # ALPHA 越小：越接近纯最大夏普（模型分数主导）
    # ALPHA 越大：越接近纯风险平价（波动率主导）
    # ALPHA=0.3：模型分数为主，轻度风险平价约束
    model_mu = scores_arr / (scores_arr.sum() + 1e-12)  # 归一化模型分数
    ALPHA    = 0.3   # 轻度风险平价惩罚，让模型分数主导

    def combined_objective(w, mu, cov, vol, n, alpha):
        # 收益项：模型分数加权期望
        ret    = w @ mu
        # 风险项：组合波动率（协方差矩阵，含相关性）
        risk   = np.sqrt(w @ cov @ w + 1e-10)
        sharpe = ret / risk

        # 风险平价惩罚：各股票风险贡献偏离均等的程度
        # 使用真实边际风险贡献 MRC_i = w_i * (Σw)_i / sqrt(w'Σw)
        sigma_w = cov @ w
        mrc     = w * sigma_w / (risk + 1e-10)   # 真实边际风险贡献
        target  = mrc.sum() / n                   # 均等目标
        penalty_raw = np.sum((mrc - target) ** 2)

        # ── 量纲对齐 ──────────────────────────────────────────
        # 夏普项和惩罚项数量级可能差几个量级，导致 alpha 失效
        # 用各自的绝对值归一化，使两项初始贡献在同一量级
        sharpe_scale  = abs(sharpe)  + 1e-10
        penalty_scale = abs(penalty_raw) + 1e-10
        penalty_normalized = penalty_raw / penalty_scale * sharpe_scale

        return -(sharpe - alpha * penalty_normalized)

    n = len(model_mu)
    if n == 1:
        best_w = np.array([1.0])
    else:
        w0   = np.ones(n) / n
        cons = [{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}]
        # 每只股票权重下限 1%（不完全排除），上限 90%（不完全集中）
        bnds = [(0.01, 0.90)] * n
        res  = minimize(
            combined_objective, w0,
            args=(model_mu, cov_matrix, volatility, n, ALPHA),
            method='SLSQP', bounds=bnds, constraints=cons,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )
        best_w = res.x if res.success else w0

    # 打印各股票风险贡献情况
    risk_contribs = best_w * volatility
    print(f"\n  融合优化结果（最大夏普 + 风险平价，alpha={ALPHA}）：")
    print(f"  {'股票':<10} {'权重':>8} {'波动率':>8} {'风险贡献':>10} {'模型分数':>10}")
    for i, c in enumerate(codes):
        print(f"  {c:<10} {best_w[i]:>8.3f} {volatility[i]:>8.4f} {risk_contribs[i]:>10.4f} {scores_arr[i]:>10.4f}")
    port_ret  = float(best_w @ model_mu)
    port_risk = float(np.sqrt(best_w @ cov_matrix @ best_w))
    print(f"  组合期望={port_ret:.4f}  组合波动={port_risk:.4f}  夏普={port_ret/port_risk:.4f}")

    # ── Step4：输出结果 ────────────────────────────────────
    weights = best_w * (1.0 - WEIGHT_CASH_RESERVE)

    result = pd.DataFrame({'stock_id': codes, 'weight': weights.tolist()})
    result.to_csv(OUTPUT_CSV, index=False)

    print(f"  预测日期：{latest_date.date()}")
    for code, w in zip(codes, weights):
        print(f"    {code}  weight={w:.4f}")
    print(f"  权重总和：{weights.sum():.6f}（现金占比：{1-weights.sum():.4f}）")
    print(f"\n结果已写入：{OUTPUT_CSV}")


if __name__ == '__main__':
    main()
