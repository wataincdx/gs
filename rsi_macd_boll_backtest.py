#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RSI(14) + MACD(12,26,9) + 布林带(20,2) 三指标组合策略回测。

题面规则
--------
买入：RSI < 30 且 MACD 金叉 且 价格突破布林带下轨
卖出：RSI > 70 或 MACD 死叉 或 价格突破布林带上轨
周期：2018-01-01 ~ 2023-12-31，沪深300成分股

口径说明（重要）
----------------
* 「突破下轨」按题面超卖反弹逻辑实现为 `close <= bb_lower`（跌破/触及下轨）；
  `--bb-entry cross_up` 可切换为「自下向上穿越下轨」（字面意义的突破）。
  「突破上轨」实现为 `close >= bb_upper`。
* RSI 用 Wilder EWM 平滑，MACD 用 EMA(12)-EMA(26) 与 EMA(9) 信号线，
  布林带用 20 日 SMA ± 2×样本标准差（ddof=1）——与常见行情软件口径一致。
* 实测发现：严格同日「RSI<30 且 MACD 金叉 且 close<=下轨」三条件在日线上时序互斥，
  6 年 73 万+ 股票日中「MACD 金叉 ∩ 价格在下轨外」出现 0 次（金叉需价格反弹，
  下轨外仍在下跌）。故默认 `--signal-window 5` 共振口径：金叉在最近 5 个交易日
  内出现、当日 RSI 与布林带条件同时满足；`--signal-window 1` 复现严格口径（0 笔成交）。
* 指标只用 T 日及之前收盘价；信号 T 日收盘生成，**T+1 开盘价成交**。
* 成分股按历史生效日判定（baostock 逐月采样），避免用未来成分回溯。
* 组合等权、最多 `--max-positions` 只，A股一手 100 股整数倍；
  佣金双边万三，卖出印花税 2023-08-28 前千一、之后万五。
* 未模拟：涨跌停无法成交、停牌顺延、滑点、分红税、融券。结果偏乐观。

运行
----
python hs300_data.py                         # 先构建真实数据集
python rsi_macd_boll_backtest.py             # 回测 + 图表 + HTML 报告
python rsi_macd_boll_backtest.py --bb-entry cross_up --max-positions 10

输出（./backtest_output/）
-------------------------
equity.csv / trades.csv / signals.csv / performance.json
charts/*.png  与 report.html
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Config:
    start: str = "2018-01-01"
    end: str = "2023-12-31"
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    boll_window: int = 20
    boll_k: float = 2.0
    bb_entry: str = "below"       # "below" | "cross_up"
    bb_exit: str = "above"        # "above"  | "cross_down"
    signal_window: int = 5        # MACD 金叉允许在最近 N 个交易日内出现；1 = 严格同日
    initial_capital: float = 1_000_000.0
    max_positions: int = 20
    commission: float = 0.0003
    stamp_before_20230828: float = 0.001
    stamp_after_20230828: float = 0.0005
    risk_free: float = 0.02


# --------------------------------------------------------------------------- #
# 指标（口径与 workbuddy-finance-skill/scripts/price-action/basic_indicators.py 一致）
# --------------------------------------------------------------------------- #
def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def compute_macd(close: pd.Series, cfg: Config) -> pd.DataFrame:
    dif = close.ewm(span=cfg.macd_fast, adjust=False).mean() - close.ewm(
        span=cfg.macd_slow, adjust=False
    ).mean()
    dea = dif.ewm(span=cfg.macd_signal, adjust=False).mean()
    prev_dif, prev_dea = dif.shift(1), dea.shift(1)
    return pd.DataFrame({
        "dif": dif,
        "dea": dea,
        "hist": dif - dea,
        "golden": (dif > dea) & (prev_dif <= prev_dea),
        "death": (dif < dea) & (prev_dif >= prev_dea),
    })


def compute_boll(close: pd.Series, cfg: Config) -> pd.DataFrame:
    mid = close.rolling(cfg.boll_window).mean()
    std = close.rolling(cfg.boll_window).std(ddof=1)
    return pd.DataFrame({
        "bb_mid": mid,
        "bb_upper": mid + cfg.boll_k * std,
        "bb_lower": mid - cfg.boll_k * std,
    })


def build_indicators(prices: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = prices.sort_values(["code", "date"]).copy()
    parts = []
    for code, sub in df.groupby("code", sort=False):
        close = sub["close"].reset_index(drop=True)
        rsi = compute_rsi(close, cfg.rsi_period)
        macd = compute_macd(close, cfg)
        boll = compute_boll(close, cfg)
        block = pd.concat([close.rename("close"), rsi.rename("rsi"), macd, boll], axis=1)
        if cfg.bb_entry == "cross_up":
            block["bb_low_break"] = (close > block["bb_lower"]) & (
                close.shift(1) <= block["bb_lower"].shift(1)
            )
        else:
            block["bb_low_break"] = close <= block["bb_lower"]
        if cfg.bb_exit == "cross_down":
            block["bb_high_break"] = (close < block["bb_upper"]) & (
                close.shift(1) >= block["bb_upper"].shift(1)
            )
        else:
            block["bb_high_break"] = close >= block["bb_upper"]

        # 严格同日口径（题面字面）：RSI<30 且当日 MACD 金叉 且价格在下轨下方
        block["strict_buy"] = (
            (rsi < cfg.rsi_oversold) & macd["golden"] & block["bb_low_break"]
        ).fillna(False)
        # 可交易口径：MACD 金叉允许在最近 N 日出现（N=1 即退化为严格同日）
        golden_recent = (
            macd["golden"].rolling(cfg.signal_window, min_periods=1).max().astype(bool)
            if cfg.signal_window > 1
            else macd["golden"]
        )
        block["golden_recent"] = golden_recent
        block["buy_signal"] = (
            (rsi < cfg.rsi_oversold) & golden_recent & block["bb_low_break"]
        ).fillna(False)
        block["sell_signal"] = (
            (rsi > cfg.rsi_overbought) | macd["death"] | block["bb_high_break"]
        ).fillna(False)
        block["date"] = sub["date"].to_numpy()
        block["code"] = code
        parts.append(block)
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out


# --------------------------------------------------------------------------- #
# 回测引擎
# --------------------------------------------------------------------------- #
def stamp_duty(dt: pd.Timestamp, cfg: Config) -> float:
    return (
        cfg.stamp_after_20230828
        if dt >= pd.Timestamp("2023-08-28")
        else cfg.stamp_before_20230828
    )


def prepare_signal_maps(
    indicators: pd.DataFrame,
    membership: pd.DataFrame,
    cfg: Config,
) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    """按历史成分过滤信号，生成逐日买卖清单。"""
    act = indicators[
        (indicators["date"] >= pd.Timestamp(cfg.start))
        & (indicators["date"] <= pd.Timestamp(cfg.end))
        & (indicators["buy_signal"] | indicators["sell_signal"])
    ].copy()

    eff = pd.DataFrame({"effective_date": pd.to_datetime(
        sorted(membership["effective_date"].unique())
    )})
    mem = membership.copy()
    mem["effective_date"] = pd.to_datetime(mem["effective_date"])

    if len(act):
        act = pd.merge_asof(
            act.sort_values("date"), eff.sort_values("effective_date"),
            left_on="date", right_on="effective_date", direction="backward",
        )
        act = act.merge(mem, on=["effective_date", "code"], how="inner")
    buy_map = {d: list(g["code"]) for d, g in act[act["buy_signal"]].groupby("date")}
    sell_map = {d: set(g["code"]) for d, g in act[act["sell_signal"]].groupby("date")}
    return buy_map, sell_map, act, eff


def run_backtest(
    prices: pd.DataFrame,
    index: pd.DataFrame,
    indicators: pd.DataFrame,
    membership: pd.DataFrame,
    cfg: Config,
) -> dict:
    buy_map, sell_map, signals, eff = prepare_signal_maps(indicators, membership, cfg)

    close_px = prices.pivot(index="date", columns="code", values="close")
    open_px = prices.pivot(index="date", columns="code", values="open")
    trade_dates = [d for d in index["date"] if pd.Timestamp(cfg.start) <= d <= pd.Timestamp(cfg.end)]
    bench = index.set_index("date")["close"].reindex(trade_dates)

    cash = cfg.initial_capital
    positions: dict[str, int] = {}
    entry: dict[str, dict] = {}
    equity_prev = cfg.initial_capital
    equity_rows, trade_rows, closed = [], [], []

    def px_at(frame: pd.DataFrame, dt: pd.Timestamp, code: str) -> float:
        if code not in frame.columns or dt not in frame.index:
            return np.nan
        val = frame.at[dt, code]
        return float(val) if pd.notna(val) else np.nan

    for i, dt in enumerate(trade_dates):
        prev = trade_dates[i - 1] if i > 0 else None

        if prev is not None:
            # 卖出：昨日出现任一恶化条件
            for code in sorted(sell_map.get(prev, set())):
                if code not in positions:
                    continue
                price = px_at(open_px, dt, code)
                if not np.isfinite(price):
                    continue
                shares = positions.pop(code)
                amount = shares * price
                fee = amount * (cfg.commission + stamp_duty(dt, cfg))
                cash += amount - fee
                info = entry.pop(code)
                pnl = amount - fee - info["cost"]
                closed.append({
                    "code": code, "entry_date": info["date"], "exit_date": dt,
                    "hold_days": (dt - info["date"]).days,
                    "pnl": pnl, "return": pnl / info["cost"] if info["cost"] else np.nan,
                    "reason": "sell_signal",
                })
                trade_rows.append({
                    "date": dt, "signal_date": prev, "code": code, "side": "SELL",
                    "price": price, "shares": shares, "amount": amount, "fee": fee,
                })

            # 买入：昨日三因子同时满足，按 RSI 由低到高优先（超卖更深者优先）
            candidates = [
                c for c in buy_map.get(prev, [])
                if c not in positions and c not in sell_map.get(prev, set())
            ]
            if candidates:
                day_sig = signals[signals["date"] == prev].set_index("code")
                candidates.sort(key=lambda c: day_sig.at[c, "rsi"] if c in day_sig.index else 50)
            for code in candidates:
                if len(positions) >= cfg.max_positions:
                    break
                price = px_at(open_px, dt, code)
                if not np.isfinite(price):
                    continue
                target = equity_prev / cfg.max_positions
                unit = price * (1 + cfg.commission)
                max_by_cash = int(cash / (unit * 100)) * 100
                max_by_target = int(target / (price * 100)) * 100
                shares = min(max_by_cash, max_by_target)
                if shares < 100:
                    continue
                cost = shares * price
                fee = cost * cfg.commission
                cash -= cost + fee
                positions[code] = shares
                entry[code] = {"date": dt, "cost": cost + fee, "price": price}
                trade_rows.append({
                    "date": dt, "signal_date": prev, "code": code, "side": "BUY",
                    "price": price, "shares": shares, "amount": cost, "fee": fee,
                })

        market_value = 0.0
        for code, shares in positions.items():
            cpx = px_at(close_px, dt, code)
            if np.isfinite(cpx):
                market_value += shares * cpx
            else:  # 停牌：按最近可用收盘价估值
                hist = close_px[code].loc[:dt].dropna()
                if len(hist):
                    market_value += shares * float(hist.iloc[-1])
        equity = cash + market_value
        equity_rows.append({
            "date": dt, "equity": equity, "cash": cash,
            "n_positions": len(positions),
            "benchmark": float(bench.loc[dt]) if pd.notna(bench.loc[dt]) else np.nan,
        })
        equity_prev = equity

    eq = pd.DataFrame(equity_rows)
    trades = pd.DataFrame(trade_rows, columns=[
        "date", "signal_date", "code", "side", "price", "shares", "amount", "fee",
    ])
    closed_df = pd.DataFrame(closed)
    return {"equity": eq, "trades": trades, "closed": closed_df,
            "signals": signals, "eff": eff, "positions": positions,
            "indicators": indicators}


# --------------------------------------------------------------------------- #
# 绩效
# --------------------------------------------------------------------------- #
def performance(eq: pd.DataFrame, cfg: Config) -> dict:
    eq = eq.copy()
    eq["ret"] = eq["equity"].pct_change()
    eq["drawdown"] = eq["equity"] / eq["equity"].cummax() - 1
    eq["bench_norm"] = eq["benchmark"] / eq["benchmark"].iloc[0] * cfg.initial_capital
    eq["bench_drawdown"] = eq["bench_norm"] / eq["bench_norm"].cummax() - 1
    eq["strategy_norm"] = eq["equity"] / cfg.initial_capital
    eq["bench_ret"] = eq["benchmark"].pct_change()

    n = len(eq)
    years = n / TRADING_DAYS
    total_ret = eq["equity"].iloc[-1] / cfg.initial_capital - 1
    ann_ret = (1 + total_ret) ** (1 / years) - 1
    ann_vol = eq["ret"].std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = (ann_ret - cfg.risk_free) / ann_vol if ann_vol > 0 else np.nan
    mdd = eq["drawdown"].min()
    downside = eq.loc[eq["ret"] < 0, "ret"].std(ddof=1) * np.sqrt(TRADING_DAYS)
    sortino = (ann_ret - cfg.risk_free) / downside if downside and downside > 0 else np.nan

    bench_total = eq["benchmark"].iloc[-1] / eq["benchmark"].iloc[0] - 1
    bench_ann = (1 + bench_total) ** (1 / years) - 1
    bench_vol = eq["bench_ret"].std(ddof=1) * np.sqrt(TRADING_DAYS)
    bench_sharpe = (bench_ann - cfg.risk_free) / bench_vol if bench_vol > 0 else np.nan

    yearly = []
    for y, g in eq.assign(year=eq["date"].dt.year).groupby("year"):
        start_eq = g["equity"].iloc[0] / (1 + g["ret"].iloc[0]) if pd.notna(g["ret"].iloc[0]) else cfg.initial_capital
        yearly.append({
            "year": int(y),
            "strategy": g["equity"].iloc[-1] / start_eq - 1,
            "benchmark": g["benchmark"].iloc[-1] / g["benchmark"].iloc[0] - 1,
        })

    stats = {
        "period": f"{eq['date'].iloc[0].date()} ~ {eq['date'].iloc[-1].date()}",
        "n_trading_days": n,
        "initial_capital": cfg.initial_capital,
        "final_equity": float(eq["equity"].iloc[-1]),
        "total_return": float(total_ret),
        "annual_return": float(ann_ret),
        "annual_volatility": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(mdd),
        "calmar": float(ann_ret / abs(mdd)) if mdd < 0 else np.nan,
        "benchmark_total_return": float(bench_total),
        "benchmark_annual_return": float(bench_ann),
        "benchmark_max_drawdown": float(eq["bench_drawdown"].min()),
        "benchmark_sharpe": float(bench_sharpe),
        "excess_annual_return": float(ann_ret - bench_ann),
        "yearly": yearly,
    }
    return {"stats": stats, "equity": eq}


# --------------------------------------------------------------------------- #
# 可视化
# --------------------------------------------------------------------------- #
def make_charts(eq: pd.DataFrame, trades: pd.DataFrame, closed: pd.DataFrame,
                indicators: pd.DataFrame, prices: pd.DataFrame,
                stats: dict, out_dir: Path, cfg: Config) -> dict[str, Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    fonts = {f.name for f in font_manager.fontManager.ttflist}
    cjk = [f for f in ("Microsoft YaHei", "SimHei", "DengXian", "SimSun") if f in fonts]
    plt.rcParams["font.sans-serif"] = cjk + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    cjk_ok = bool(cjk)

    def L(zh: str, en: str) -> str:
        return zh if cjk_ok else en

    RED, GREEN, BLUE, GREY = "#d62728", "#2ca02c", "#1f77b4", "#888888"
    chart_dir = out_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # 1) 净值曲线 + 持仓数
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(eq["date"], eq["strategy_norm"], color=RED, lw=1.8, label=L("策略净值", "Strategy"))
    ax.plot(eq["date"], eq["bench_norm"] / cfg.initial_capital, color=GREY, lw=1.4,
            label=L("沪深300", "CSI 300"))
    ax.axhline(1.0, color=BLUE, lw=0.8, ls="--", alpha=0.6)
    ax.set_ylabel(L("归一化净值", "Normalized NAV"))
    ax.set_title(L("策略净值 vs 沪深300（2018-2023）", "Strategy NAV vs CSI 300 (2018-2023)"))
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    ax2.fill_between(eq["date"], eq["n_positions"], color=BLUE, alpha=0.45, step="post")
    ax2.set_ylabel(L("持仓数", "Positions"))
    ax2.set_xlabel(L("日期", "Date"))
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    p = chart_dir / "01_equity_curve.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths["equity"] = p

    # 2) 回撤
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.fill_between(eq["date"], eq["drawdown"] * 100, 0, color=GREEN, alpha=0.35)
    ax.plot(eq["date"], eq["drawdown"] * 100, color=GREEN, lw=1.0)
    ax.plot(eq["date"], eq["bench_drawdown"] * 100, color=GREY, lw=0.9,
            label=L("沪深300回撤", "CSI300 DD"))
    ax.set_ylabel(L("回撤 (%)", "Drawdown (%)"))
    ax.set_title(L(f"回撤曲线（策略最大回撤 {stats['max_drawdown']:.2%}）",
                   f"Drawdown (max {stats['max_drawdown']:.2%})"))
    ax.legend(loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p = chart_dir / "02_drawdown.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths["drawdown"] = p

    # 3) 月度收益热力图（A股口径：红涨绿跌）
    m = eq.set_index("date")["equity"].resample("ME").last()
    mret = m.pct_change()
    mret.iloc[0] = m.iloc[0] / cfg.initial_capital - 1
    tab = pd.DataFrame({"y": mret.index.year, "m": mret.index.month, "r": mret.values})
    pivot = tab.pivot(index="y", columns="m", values="r")
    fig, ax = plt.subplots(figsize=(11, 3.8))
    data = pivot.values * 100
    vmax = np.nanmax(np.abs(data)) or 1
    im = ax.imshow(data, cmap="RdYlGn_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(12), [f"{i}月" if cjk_ok else str(i) for i in range(1, 13)])
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for iy in range(data.shape[0]):
        for ix in range(data.shape[1]):
            if np.isfinite(data[iy, ix]):
                ax.text(ix, iy, f"{data[iy, ix]:.1f}", ha="center", va="center", fontsize=8)
    ax.set_title(L("策略月度收益 (%)  红涨绿跌", "Monthly Return (%)  Red=Up / Green=Down"))
    fig.colorbar(im, ax=ax, shrink=0.85, label="%")
    fig.tight_layout()
    p = chart_dir / "03_monthly_heatmap.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths["monthly"] = p

    # 4) 样本个股：价格+布林带+买卖点 / RSI / MACD
    if len(trades):
        recent_cut = trades["date"].max() - pd.Timedelta(days=400)
        recent = trades[trades["date"] >= recent_cut]
        pool = recent if len(recent) else trades
        sample_code = pool["code"].value_counts().index[0]
        sub = indicators[indicators["code"] == sample_code].sort_values("date")
        sub = sub[(sub["date"] >= pd.Timestamp(cfg.start)) & (sub["date"] <= pd.Timestamp(cfg.end))]
        sub = sub.tail(250)
        tr = trades[trades["code"] == sample_code]
        tr = tr[(tr["date"] >= sub["date"].min()) & (tr["date"] <= sub["date"].max())]
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                                gridspec_kw={"height_ratios": [3, 1, 1.2]})
        ax = axes[0]
        ax.plot(sub["date"], sub["close"], color="#222222", lw=1.1, label=L("收盘价", "Close"))
        ax.plot(sub["date"], sub["bb_upper"], color=GREY, lw=0.9, ls="--", label="BOLL +/-2σ")
        ax.plot(sub["date"], sub["bb_mid"], color=BLUE, lw=0.8, alpha=0.7)
        ax.plot(sub["date"], sub["bb_lower"], color=GREY, lw=0.9, ls="--")
        b = tr[tr["side"] == "BUY"]
        s = tr[tr["side"] == "SELL"]
        ax.scatter(b["date"], b["price"], marker="^", color=RED, s=70, zorder=5, label=L("买入", "BUY"))
        ax.scatter(s["date"], s["price"], marker="v", color=GREEN, s=70, zorder=5, label=L("卖出", "SELL"))
        ax.set_title(L(f"{sample_code}：布林带与买卖点（最近 250 个交易日）",
                       f"{sample_code}: Bollinger Bands & Trades (last 250 sessions)"))
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.25)
        ax = axes[1]
        ax.plot(sub["date"], sub["rsi"], color="#9467bd", lw=1.0)
        ax.axhline(cfg.rsi_overbought, color=RED, ls="--", lw=0.8)
        ax.axhline(cfg.rsi_oversold, color=GREEN, ls="--", lw=0.8)
        ax.set_ylabel("RSI(14)")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.25)
        ax = axes[2]
        ax.plot(sub["date"], sub["dif"], color=RED, lw=1.0, label="DIF")
        ax.plot(sub["date"], sub["dea"], color=BLUE, lw=1.0, label="DEA")
        ax.bar(sub["date"], sub["hist"], color=np.where(sub["hist"] >= 0, RED, GREEN),
               alpha=0.35, width=1.0)
        ax.set_ylabel("MACD")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        p = chart_dir / "04_signal_sample.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths["signals"] = p

    # 5) 年度收益对比
    yearly = pd.DataFrame(stats["yearly"])
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(len(yearly))
    ax.bar(x - 0.2, yearly["strategy"] * 100, width=0.4, color=RED, label=L("策略", "Strategy"))
    ax.bar(x + 0.2, yearly["benchmark"] * 100, width=0.4, color=GREY, label=L("沪深300", "CSI 300"))
    ax.set_xticks(x, yearly["year"])
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_ylabel(L("年度收益 (%)", "Annual Return (%)"))
    ax.set_title(L("分年度收益对比", "Yearly Return Comparison"))
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    p = chart_dir / "05_yearly_return.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths["yearly"] = p
    return paths


# --------------------------------------------------------------------------- #
# HTML 报告
# --------------------------------------------------------------------------- #
def write_html_report(out_dir: Path, stats: dict, cfg: Config, eq: pd.DataFrame,
                      trades: pd.DataFrame, closed: pd.DataFrame,
                      charts: dict[str, Path], meta: dict,
                      strict_n: int = 0, n_stock_days: int = 0,
                      cond: dict | None = None,
                      sens: pd.DataFrame | None = None) -> Path:
    cond = cond or {"rsi": 0, "golden": 0, "bb": 0, "golden_bb": 0}
    # 净值图按月采样，控制 JSON 体积
    sample = eq.iloc[::5].copy()
    dates = [d.strftime("%Y-%m-%d") for d in sample["date"]]
    nav = [round(float(v), 4) for v in sample["strategy_norm"]]
    bm = [round(float(v / eq["benchmark"].iloc[0]), 4) for v in sample["benchmark"]]

    n_trades = len(trades)
    n_closed = len(closed)
    win_rate = float((closed["return"] > 0).mean()) if n_closed else float("nan")
    avg_hold = float(closed["hold_days"].mean()) if n_closed else float("nan")
    fee_total = float(trades["fee"].sum()) if n_trades else 0.0

    def pct(v):
        return f"{v:.2%}" if np.isfinite(v) else "—"

    rows = "".join(
        f"<tr><td>{r['year']}</td><td class='num'>{pct(r['strategy'])}</td>"
        f"<td class='num'>{pct(r['benchmark'])}</td>"
        f"<td class='num'>{pct(r['strategy'] - r['benchmark'])}</td></tr>"
        for r in stats["yearly"]
    )
    monthly = eq.set_index("date")["equity"].resample("ME").last()
    monthly_rows = "".join(
        f"<tr><td>{d.date()}</td><td class='num'>{v:,.0f}</td>"
        f"<td class='num'>{v / cfg.initial_capital - 1:.2%}</td></tr>"
        for d, v in monthly.items()
    )
    kpis = [
        ("年化收益", pct(stats["annual_return"]), "strategy"),
        ("最大回撤", pct(stats["max_drawdown"]), "risk"),
        ("夏普比率", f"{stats['sharpe']:.2f}" if np.isfinite(stats["sharpe"]) else "—", "strategy"),
        ("累计收益", pct(stats["total_return"]), "strategy"),
        ("基准年化", pct(stats["benchmark_annual_return"]), "bench"),
        ("超额年化", pct(stats["excess_annual_return"]), "bench"),
    ]
    kpi_html = "".join(
        f"<div class='kpi'><div class='label'>{k}</div><div class='value {c}'>{v}</div></div>"
        for k, v, c in kpis
    )

    sens_html = ""
    if sens is not None and len(sens):
        def f2(v):
            return f"{v:.2f}" if np.isfinite(v) else "—"

        sens_rows = "".join(
            f"<tr><td>{int(r.signal_window)}</td><td>{r.strict_same_day}</td>"
            f"<td class='num'>{pct(r.annual_return)}</td>"
            f"<td class='num'>{pct(r.max_drawdown)}</td>"
            f"<td class='num'>{f2(r.sharpe)}</td>"
            f"<td class='num'>{int(r.n_trades)}</td>"
            f"<td class='num'>{pct(r.win_rate)}</td></tr>"
            for r in sens.itertuples()
        )
        sens_html = f"""<section>
  <h2>参数敏感性：金叉共振窗口</h2>
  <p style="font-size:13px;color:var(--sub);margin:0 0 10px">窗口=1 即题面严格同日口径；其余为「金叉在最近 N 个交易日内出现」的共振口径。同一份真实行情、同一套成本与执行假设，仅改变该参数。</p>
  <table><tr><th>窗口 N</th><th>严格同日</th><th style="text-align:right">年化收益</th><th style="text-align:right">最大回撤</th><th style="text-align:right">夏普</th><th style="text-align:right">成交笔数</th><th style="text-align:right">平仓胜率</th></tr>{sens_rows}</table>
</section>"""

    chart_imgs = "".join(
        f"<figure><img src='charts/{p.name}' alt='{key}' loading='lazy'>"
        f"<figcaption>{label}</figcaption></figure>"
        for key, p, label in [
            ("equity", charts.get("equity"), "净值曲线与持仓数"),
            ("drawdown", charts.get("drawdown"), "回撤曲线"),
            ("monthly", charts.get("monthly"), "月度收益热力图"),
            ("signals", charts.get("signals"), "样本个股信号图（布林带 / RSI / MACD）"),
            ("yearly", charts.get("yearly"), "分年度收益对比"),
        ] if p is not None
    )

    js_data = json.dumps({"dates": dates, "nav": nav, "bm": bm}, ensure_ascii=False)
    script = f"""
var DATA = {js_data};
var chart = echarts.init(document.getElementById('equityChart'));
chart.setOption({{
  tooltip: {{ trigger: 'axis' }},
  legend: {{ data: ['策略净值', '沪深300'] }},
  grid: {{ left: 55, right: 20, top: 40, bottom: 40 }},
  xAxis: {{ type: 'category', data: DATA.dates, axisLabel: {{ interval: 11 }} }},
  yAxis: {{ type: 'value', scale: true, name: 'NAV' }},
  series: [
    {{ name: '策略净值', type: 'line', data: DATA.nav, showSymbol: false, lineStyle: {{ width: 2, color: '#d62728' }}, itemStyle: {{ color: '#d62728' }} }},
    {{ name: '沪深300', type: 'line', data: DATA.bm, showSymbol: false, lineStyle: {{ width: 1.5, color: '#888888' }}, itemStyle: {{ color: '#888888' }} }}
  ]
}});
window.addEventListener('resize', function() {{ chart.resize(); }});
var chartDom = document.getElementById('equityChart');
var tableDom = document.getElementById('equityTable');
var btnC = document.getElementById('btnChart');
var btnT = document.getElementById('btnTable');
btnT.onclick = function() {{ chartDom.style.display='none'; tableDom.style.display='block'; btnT.classList.add('on'); btnC.classList.remove('on'); }};
btnC.onclick = function() {{ chartDom.style.display='block'; tableDom.style.display='none'; btnC.classList.add('on'); btnT.classList.remove('on'); chart.resize(); }};
"""
    (out_dir / "_inline.js").write_text(script, encoding="utf-8")

    source = "akshare / 东方财富（前复权日线）；baostock / 中证指数（历史成分）"
    if meta.get("membership_mode", "").startswith("akshare_current"):
        source += "；⚠️ 历史成分获取失败，降级为当前成分快照，存在幸存者偏差"
    built = meta.get("built_at", "—")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RSI+MACD+BOLL 三指标策略回测报告（2018-2023）</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
  :root {{ --ink:#1c2430; --sub:#5f6b7a; --line:#e3e8ee; --bg:#ffffff; --panel:#f7f9fb; --red:#c62828; --green:#2e7d32; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:#eef1f5; color:var(--ink); font-family:-apple-system,"Microsoft YaHei","Segoe UI",sans-serif; line-height:1.6; }}
  .wrap {{ max-width:1080px; margin:0 auto; padding:28px 20px 60px; }}
  header {{ background:var(--bg); border:1px solid var(--line); border-radius:10px; padding:22px 26px; margin-bottom:18px; }}
  h1 {{ font-size:22px; margin:0 0 6px; }}
  .meta {{ color:var(--sub); font-size:13px; }}
  .tldr {{ background:#fff8f6; border-left:4px solid var(--red); padding:12px 16px; margin-top:14px; font-size:14px; }}
  .kpis {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:18px 0; }}
  .kpi {{ background:var(--bg); border:1px solid var(--line); border-radius:8px; padding:14px 16px; }}
  .kpi .label {{ color:var(--sub); font-size:12px; }}
  .kpi .value {{ font-size:26px; font-weight:650; margin-top:4px; font-variant-numeric:tabular-nums; }}
  .kpi .value.strategy {{ color:var(--red); }}
  .kpi .value.risk {{ color:var(--green); }}
  .kpi .value.bench {{ color:var(--ink); }}
  section {{ background:var(--bg); border:1px solid var(--line); border-radius:10px; padding:20px 24px; margin-bottom:18px; }}
  h2 {{ font-size:16px; margin:0 0 14px; border-bottom:1px solid var(--line); padding-bottom:9px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th,td {{ padding:8px 10px; border-bottom:1px solid var(--line); text-align:left; }}
  th {{ background:var(--panel); font-weight:600; }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  #equityChart {{ width:100%; height:340px; }}
  .switch {{ margin-bottom:10px; }}
  .switch button {{ border:1px solid var(--line); background:var(--panel); color:var(--sub); padding:5px 14px; cursor:pointer; border-radius:5px; font-size:12px; }}
  .switch button.on {{ background:var(--red); color:#fff; border-color:var(--red); }}
  #equityTable {{ max-height:340px; overflow:auto; }}
  figure {{ margin:0 0 22px; }}
  figure img {{ width:100%; border:1px solid var(--line); border-radius:6px; background:#fff; }}
  figcaption {{ color:var(--sub); font-size:12px; margin-top:5px; text-align:center; }}
  ul {{ margin:6px 0; padding-left:20px; font-size:13px; }}
  li {{ margin:4px 0; }}
  .foot {{ color:var(--sub); font-size:12px; margin-top:16px; }}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>RSI(14) + MACD(12,26,9) + 布林带(20,2) 组合策略回测</h1>
  <div class="meta">回测区间 {stats['period']} · 沪深300历史成分股 · 初始资金 ¥{cfg.initial_capital:,.0f} · 数据源：{source} · 数据构建时间 {built}</div>
  <div class="tldr"><strong>结论：</strong>策略年化收益 {pct(stats['annual_return'])}，最大回撤 {pct(stats['max_drawdown'])}，夏普 {stats['sharpe']:.2f}；同期沪深300年化 {pct(stats['benchmark_annual_return'])}、最大回撤 {pct(stats['benchmark_max_drawdown'])}，超额年化 {pct(stats['excess_annual_return'])}。共 {n_trades} 笔成交，平仓胜率 {pct(win_rate)}。</div>
  <div class="tldr" style="background:#f4f8fc;border-left-color:#1f77b4"><strong>口径提示（重要）：</strong>严格按题面「同日 RSI&lt;30 且 MACD 金叉 且 close≤布林带下轨」，在 {n_stock_days:,} 个股票日中仅出现 {strict_n} 次（真实数据统计）——MACD 金叉需要价格反弹，而 close≤下轨要求仍在极端下跌，两者时序互斥，严格口径无可用交易样本。本报告采用 <code>--signal-window {cfg.signal_window}</code> 共振口径：金叉在最近 {cfg.signal_window} 个交易日内出现，当日 RSI 与布林带条件同时满足；<code>--signal-window 1</code> 可复现严格口径。</div>
</header>

<div class="kpis">{kpi_html}</div>

<section>
  <h2>交互式净值曲线（图 / 表可切换，精确数值见下表与 equity.csv）</h2>
  <div class="switch"><button id="btnChart" class="on">图</button><button id="btnTable">表</button></div>
  <div id="equityChart"></div>
  <div id="equityTable" style="display:none">
    <table><tr><th>月末</th><th style="text-align:right">组合权益（¥）</th><th style="text-align:right">累计收益</th></tr>{monthly_rows}</table>
  </div>
  <table>
    <tr><th>指标</th><th style="text-align:right">策略</th><th style="text-align:right">沪深300</th></tr>
    <tr><td>累计收益</td><td class="num">{pct(stats['total_return'])}</td><td class="num">{pct(stats['benchmark_total_return'])}</td></tr>
    <tr><td>年化收益</td><td class="num">{pct(stats['annual_return'])}</td><td class="num">{pct(stats['benchmark_annual_return'])}</td></tr>
    <tr><td>年化波动</td><td class="num">{pct(stats['annual_volatility'])}</td><td class="num">—</td></tr>
    <tr><td>最大回撤</td><td class="num">{pct(stats['max_drawdown'])}</td><td class="num">{pct(stats['benchmark_max_drawdown'])}</td></tr>
    <tr><td>夏普比率（rf={cfg.risk_free:.0%}）</td><td class="num">{stats['sharpe']:.2f}</td><td class="num">{stats['benchmark_sharpe']:.2f}</td></tr>
    <tr><td>Sortino / Calmar</td><td class="num">{stats['sortino']:.2f} / {stats['calmar']:.2f}</td><td class="num">—</td></tr>
    <tr><td>成交笔数 / 平仓笔数 / 平均持有天数</td><td class="num">{n_trades} / {n_closed} / {avg_hold:.0f}</td><td class="num">—</td></tr>
    <tr><td>累计交易成本</td><td class="num">¥{fee_total:,.0f}</td><td class="num">—</td></tr>
  </table>
</section>

<section>
  <h2>分年度收益</h2>
  <table><tr><th>年份</th><th style="text-align:right">策略</th><th style="text-align:right">沪深300</th><th style="text-align:right">超额</th></tr>{rows}</table>
</section>

{sens_html}

<section>
  <h2>可视化图表</h2>
  {chart_imgs}
</section>

<section>
  <h2>口径、局限与数据来源</h2>
  <ul>
    <li><strong>信号口径</strong>：买入 = RSI&lt;30 且 MACD 金叉（DIF 上穿 DEA，允许最近 {cfg.signal_window} 日共振）且 close≤布林带下轨；卖出 = RSI&gt;70 或 MACD 死叉或 close≥布林带上轨（卖出严格按「或」逻辑，触发频繁、持仓周期短）。「突破下轨」按超卖反弹理解为跌破/触及下轨，可用 <code>--bb-entry cross_up</code> 切换为向上穿越口径。</li>
    <li><strong>题面策略的时序冲突（实测）</strong>：严格同日三条件仅 {strict_n} 次 / {n_stock_days:,} 个股票日；单独看，RSI&lt;30 出现 {cond['rsi']:,} 次、MACD 金叉 {cond['golden']:,} 次、close≤下轨 {cond['bb']:,} 次，但「金叉 ∩ 下轨外」= {cond['golden_bb']:,} 次。该结论由真实行情统计得出，不是数据缺失；用 <code>--signal-window 1</code> 复现即得到 0 笔成交。</li>
    <li><strong>执行</strong>：T 日收盘出信号、T+1 开盘成交；等权最多 {cfg.max_positions} 只，100 股整数倍；佣金双边 {cfg.commission:.2%}，卖出印花税 2023-08-28 前 {cfg.stamp_before_20230828:.2%}、之后 {cfg.stamp_after_20230828:.2%}。</li>
    <li><strong>成分股</strong>：按 baostock 逐月采样的历史生效期成分判定，避免用当前成分回溯产生幸存者偏差；价格为东方财富前复权序列（含未来除权信息，但收益率与指标方向不受影响）。</li>
    <li><strong>未模拟</strong>：涨跌停无法成交、停牌顺延、滑点、分红红利税、融券做空；实盘结果通常低于回测。指标钝化、参数过拟合、样本外失效风险未纳入。</li>
    <li><strong>数据源</strong>：{source}；行情为公开数据，非投资建议。数据构建时间 {built}。</li>
  </ul>
  <div class="foot">本报告由 rsi_macd_boll_backtest.py 自动生成 · 结果可复现（固定数据缓存与参数）</div>
</section>

<section>
  <h2>免责声明</h2>
  <p style="font-size:13px;color:var(--sub);margin:0"><strong>免责声明</strong>：以上内容基于公开数据和量化分析，仅供参考，不构成投资建议。市场有风险，投资需谨慎。任何投资决策应结合个人风险承受能力、资金状况和投资目标独立判断，必要时咨询持牌专业机构。过往表现不预示未来收益。</p>
</section>
</div>
<script>{script}</script>
</body></html>
"""
    path = out_dir / "report.html"
    path.write_text(html, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# 参数敏感性
# --------------------------------------------------------------------------- #
def run_sensitivity(prices: pd.DataFrame, index: pd.DataFrame,
                    membership: pd.DataFrame, cfg: Config,
                    windows: tuple[int, ...] = (1, 3, 5, 10)) -> pd.DataFrame:
    """对比不同「金叉共振窗口」下的绩效，检验结果对参数是否稳健。"""
    rows = []
    for w in windows:
        c = replace(cfg, signal_window=w)
        ind = build_indicators(prices, c)
        res = run_backtest(prices, index, ind, membership, c)
        s = performance(res["equity"], c)["stats"]
        rows.append({
            "signal_window": w,
            "strict_same_day": "是" if w == 1 else "否",
            "annual_return": s["annual_return"],
            "max_drawdown": s["max_drawdown"],
            "sharpe": s["sharpe"],
            "total_return": s["total_return"],
            "n_trades": int(len(res["trades"])),
            "win_rate": float((res["closed"]["return"] > 0).mean())
            if len(res["closed"]) else np.nan,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #
def self_check(result: dict, cfg: Config) -> None:
    trades = result["trades"]
    if len(trades):
        assert (trades["signal_date"] < trades["date"]).all(), "存在当日信号当日成交（前向偏差）"
        assert trades["shares"].mod(100).eq(0).all(), "股数非一手整数倍"
    eq = result["equity"]
    assert eq["equity"].gt(0).all() and eq["equity"].notna().all(), "净值异常"
    assert eq["date"].is_monotonic_increasing, "日期未升序"
    print("[self-test] T+1 执行 / 整手 / 净值连续性 / 日期顺序：通过")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RSI+MACD+BOLL 三指标策略回测")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "backtest_output")
    ap.add_argument("--bb-entry", choices=["below", "cross_up"], default="below")
    ap.add_argument("--bb-exit", choices=["above", "cross_down"], default="above")
    ap.add_argument("--signal-window", type=int, default=5,
                    help="MACD 金叉共振窗口（交易日）；1=严格同日三条件")
    ap.add_argument("--max-positions", type=int, default=20)
    ap.add_argument("--initial-capital", type=float, default=1_000_000.0)
    ap.add_argument("--risk-free", type=float, default=0.02)
    ap.add_argument("--no-charts", action="store_true")
    ap.add_argument("--no-sensitivity", action="store_true",
                    help="跳过金叉共振窗口 1/3/5/10 的敏感性分析")
    args = ap.parse_args(argv)

    cfg = Config(
        bb_entry=args.bb_entry, bb_exit=args.bb_exit,
        signal_window=max(1, args.signal_window),
        max_positions=args.max_positions, initial_capital=args.initial_capital,
        risk_free=args.risk_free,
    )
    data_dir, out_dir = args.data_dir, args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in ("prices.csv", "index.csv", "membership.csv"):
        if not (data_dir / f).exists():
            raise SystemExit(f"缺少 {data_dir / f}，请先运行：python hs300_data.py")

    print("[1/5] 载入真实行情数据 ...")
    prices = pd.read_csv(data_dir / "prices.csv", parse_dates=["date"], dtype={"code": str})
    prices["code"] = prices["code"].str.strip().str.zfill(6)
    index = pd.read_csv(data_dir / "index.csv", parse_dates=["date"])
    membership = pd.read_csv(data_dir / "membership.csv", dtype={"code": str})
    membership["code"] = membership["code"].str.strip().str.zfill(6)
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8")) \
        if (data_dir / "meta.json").exists() else {}
    print(f"  股票 {prices['code'].nunique()} 只 / {len(prices):,} 行，成分口径 {meta.get('membership_mode','?')}")

    print("[2/5] 计算 RSI / MACD / 布林带 ...")
    indicators = build_indicators(prices, cfg)
    period_mask = indicators["date"].between(pd.Timestamp(cfg.start), pd.Timestamp(cfg.end))
    sub = indicators.loc[period_mask, ["rsi", "golden", "bb_low_break", "strict_buy"]]
    cond_counts = {
        "rsi": int((sub["rsi"] < cfg.rsi_oversold).sum()),
        "golden": int(sub["golden"].sum()),
        "bb": int(sub["bb_low_break"].sum()),
        "golden_bb": int((sub["golden"] & sub["bb_low_break"]).sum()),
    }
    strict_n = int(sub["strict_buy"].sum())
    n_stock_days = int(len(sub))
    print(f"  严格同日三条件信号 {strict_n} 次 / {n_stock_days:,} 个股票日"
          f"（金叉∩下轨外 {cond_counts['golden_bb']} 次）")

    print("[3/5] 运行组合回测 ...")
    result = run_backtest(prices, index, indicators, membership, cfg)
    perf = performance(result["equity"], cfg)
    stats, eq = perf["stats"], perf["equity"]
    self_check({**result, "equity": eq}, cfg)

    eq.to_csv(out_dir / "equity.csv", index=False, encoding="utf-8-sig")
    result["trades"].to_csv(out_dir / "trades.csv", index=False, encoding="utf-8-sig")
    result["closed"].to_csv(out_dir / "closed_trades.csv", index=False, encoding="utf-8-sig")
    sig_cols = ["date", "code", "rsi", "dif", "dea", "close", "bb_lower", "bb_upper",
                "bb_low_break", "bb_high_break", "golden", "golden_recent", "strict_buy",
                "death", "buy_signal", "sell_signal"]
    result["signals"][sig_cols].to_csv(out_dir / "signals.csv", index=False, encoding="utf-8-sig")

    sens = None
    if not args.no_sensitivity:
        print("[3.5/5] 参数敏感性：金叉共振窗口 1 / 3 / 5 / 10 ...")
        sens = run_sensitivity(prices, index, membership, cfg)
        sens.to_csv(out_dir / "sensitivity.csv", index=False, encoding="utf-8-sig")

    n_closed = len(result["closed"])
    stats.update({
        "config": {
            "bb_entry": cfg.bb_entry, "bb_exit": cfg.bb_exit,
            "max_positions": cfg.max_positions, "initial_capital": cfg.initial_capital,
            "commission": cfg.commission, "risk_free": cfg.risk_free,
            "rsi": [cfg.rsi_period, cfg.rsi_oversold, cfg.rsi_overbought],
            "macd": [cfg.macd_fast, cfg.macd_slow, cfg.macd_signal],
            "boll": [cfg.boll_window, cfg.boll_k],
        },
        "strict_same_day_buy_signals": strict_n,
        "n_stock_days": n_stock_days,
        "condition_counts": cond_counts,
        "signal_window": cfg.signal_window,
        "n_trades": int(len(result["trades"])),
        "n_closed": int(n_closed),
        "win_rate": float((result["closed"]["return"] > 0).mean()) if n_closed else None,
        "avg_hold_days": float(result["closed"]["hold_days"].mean()) if n_closed else None,
        "total_fee": float(result["trades"]["fee"].sum()),
        "sensitivity": json.loads(sens.to_json(orient="records")) if sens is not None else None,
        "data_source": "akshare/EastMoney (qfq OHLCV) + baostock/CSI historical membership",
        "backtest_run_at": datetime.now().isoformat(timespec="seconds"),
    })
    (out_dir / "performance.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("[4/5] 生成图表与 HTML 报告 ...")
    charts = {}
    report = None
    if not args.no_charts:
        charts = make_charts(eq, result["trades"], result["closed"], indicators,
                             prices, stats, out_dir, cfg)
        report = write_html_report(out_dir, stats, cfg, eq, result["trades"],
                                   result["closed"], charts, meta,
                                   strict_n, n_stock_days, cond_counts, sens)

    print("[5/5] 回测结果")
    print(f"  年化收益 {stats['annual_return']:.2%} | 最大回撤 {stats['max_drawdown']:.2%} | "
          f"夏普 {stats['sharpe']:.2f}")
    print(f"  累计收益 {stats['total_return']:.2%} | 基准年化 {stats['benchmark_annual_return']:.2%} | "
          f"超额 {stats['excess_annual_return']:.2%}")
    print(f"  成交 {stats['n_trades']} 笔 | 平仓 {n_closed} 笔 | "
          f"胜率 {stats['win_rate']:.2%}" if n_closed else f"  成交 {stats['n_trades']} 笔")
    for p in list(charts.values()) + ([report] if report else []):
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
