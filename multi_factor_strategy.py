#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""三因子选股策略：动量 + 估值 + 波动率（每日股票池 + 因子得分矩阵）。

规则（严格按题设）
------------------
1. 动量因子  : 20 日收益率横截面排名进入市场前 30%（pct_rank >= 0.70）
2. 估值因子  : PE(TTM) 低于所属行业均值 20%（pe_ttm / ind_pe < 0.80）
3. 波动率因子: 10 日波动率 < 该股历史 10 日波动率序列的 30 分位数
4. 买入      : 三因子同时满足
5. 卖出      : 任一因子恶化 —— 动量跌出前 50% / PE 达到或超过行业均值 /
               10 日波动率 > 历史 40 分位数（30%~50% 为持有缓冲带，不买也不卖）
6. 输出      : 每日符合条件股票池、每日因子得分矩阵、买卖订单、持仓、每日汇总

口径与防坑
----------
* 所有因子仅使用 T 日及之前数据；横截面排名/行业均值只用 T 日截面。
* 历史波动率分位数默认用「T-1 及之前」的滚动窗口（vol_q_include_current=False），
  避免把当日波动率混入自身的历史分布。
* 信号在 T 日收盘后生成，订单标注 signal_date 与 exec_date（默认下一交易日执行）。
* 停牌/无行情日不参与横截面与买卖判定；价格前向填充仅供计算 20 日收益率。
* PE <= 0（亏损）不参与行业均值计算，也不能通过估值因子。
* 行业均值至少需 min_industry_stocks 只有效样本，否则估值因子不可评估。

输入数据（--data-dir 目录下 3 个 CSV，长表）
------------------------------------------
prices.csv    : date, code, close [, volume]        # close 为前复权收盘价
pe_ttm.csv    : date, code, pe_ttm                  # 可非每日，内部 ffill 有限天数
industry.csv  : code, industry  [, date]            # 带 date 时按最近时点匹配

运行
----
python multi_factor_strategy.py --demo --out-dir ./strategy_output   # 合成数据自测
python multi_factor_strategy.py --data-dir ./data --out-dir ./out    # 真实数据
python multi_factor_strategy.py --data-dir ./data --start 2025-01-01 --max-positions 20

依赖：pandas>=2.0, numpy。仅 pandas/numpy，无需行情库；数据由外部 CSV 提供，
便于对接 akshare / tushare / iFinD 导出的面板数据。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

TRADING_DAYS_PER_YEAR = 250


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Config:
    mom_window: int = 20          # 动量回看交易日
    mom_entry_pct: float = 0.70   # 前 30% => 百分位排名 >= 0.70
    mom_exit_pct: float = 0.50    # 跌出前 50% => 百分位排名 < 0.50

    pe_discount: float = 0.20     # 低于行业均值 20%
    pe_exit_multiple: float = 1.00  # PE >= 1.0 倍行业均值即恶化
    pe_ffill_days: int = 5        # 估值数据最多前向填充 5 个交易日
    min_industry_stocks: int = 5  # 行业均值最少有效样本数
    industry_stat: str = "mean"   # "mean" | "median"

    vol_window: int = 10          # 10 日波动率
    vol_entry_q: float = 0.30     # < 30 分位数
    vol_exit_q: float = 0.40      # > 40 分位数
    hist_window: int = 250        # 历史波动率分位数回看窗口（交易日）
    hist_min_periods: int = 60    # 分位数最少样本数
    annualize_vol: bool = True    # 年化（对分位数比较无影响，仅影响展示量级）
    vol_q_include_current: bool = False  # 历史窗口是否包含当日

    max_positions: int = 0        # 0 = 不限；>0 时按 score_total 择优买入
    exec_lag: int = 1             # 信号日之后第 N 个交易日执行
    w_mom: float = 1.0 / 3.0
    w_val: float = 1.0 / 3.0
    w_vol: float = 1.0 / 3.0

    @property
    def pe_entry_ratio(self) -> float:
        return 1.0 - self.pe_discount


# --------------------------------------------------------------------------- #
# 数据层
# --------------------------------------------------------------------------- #
def _check_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} 缺少列: {sorted(missing)}")


def load_inputs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """从目录读取 3 张长表 CSV。"""
    price = pd.read_csv(data_dir / "prices.csv")
    pe = pd.read_csv(data_dir / "pe_ttm.csv")
    industry = pd.read_csv(data_dir / "industry.csv")
    _check_columns(price, {"date", "code", "close"}, "prices.csv")
    _check_columns(pe, {"date", "code", "pe_ttm"}, "pe_ttm.csv")
    _check_columns(industry, {"code", "industry"}, "industry.csv")
    for df in (price, pe):
        df["date"] = pd.to_datetime(df["date"])
    if "date" in industry.columns:
        industry["date"] = pd.to_datetime(industry["date"])
    return price, pe, industry


def prepare_panel(
    price: pd.DataFrame,
    pe: pd.DataFrame,
    industry: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """构造 (交易日 × 股票) 完整面板，对齐价格、PE、行业。"""
    price = price.copy()
    pe = pe.copy()
    industry = industry.copy()
    for df in (price, pe):
        df["date"] = pd.to_datetime(df["date"])
        df["code"] = df["code"].astype(str)
    if "date" in industry.columns:
        industry["date"] = pd.to_datetime(industry["date"])
    industry["code"] = industry["code"].astype(str)

    price = price.drop_duplicates(["date", "code"], keep="last")
    pe = pe.drop_duplicates(["date", "code"], keep="last")

    dates = pd.DatetimeIndex(sorted(price["date"].unique()), name="date")
    codes = pd.Index(sorted(price["code"].unique()), name="code")
    idx = pd.MultiIndex.from_product([dates, codes], names=["date", "code"])

    p = price.set_index(["date", "code"]).reindex(idx).reset_index()
    # 当日有真实行情才参与横截面；停牌日价格前向填充仅用于计算区间收益率
    p["tradable"] = p["close"].notna()
    if "volume" in p.columns:
        p["tradable"] &= p["volume"].fillna(0) > 0
    p["close"] = p.groupby("code", sort=False)["close"].ffill()

    p = p.merge(pe[["date", "code", "pe_ttm"]], on=["date", "code"], how="left")
    p["pe_ttm"] = p.groupby("code", sort=False)["pe_ttm"].ffill(limit=cfg.pe_ffill_days)

    if "date" in industry.columns:
        ind = industry[["date", "code", "industry"]].drop_duplicates(
            ["date", "code"], keep="last"
        )
        p = pd.merge_asof(
            p.sort_values("date"),
            ind.sort_values("date"),
            on="date",
            by="code",
            direction="backward",
        )
    else:
        ind = industry[["code", "industry"]].drop_duplicates("code", keep="last")
        p = p.merge(ind, on="code", how="left")

    return p.sort_values(["code", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 因子层
# --------------------------------------------------------------------------- #
def compute_factors(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """计算三因子原始值、达标/恶化标记与 0-100 因子得分矩阵。"""
    df = panel.copy()
    g = df.groupby("code", sort=False)

    # 1) 动量：20 交易日累计收益率
    prev1 = g["close"].shift(1)
    prev_n = g["close"].shift(cfg.mom_window)
    df["ret_1"] = df["close"] / prev1 - 1.0
    df["ret_20"] = df["close"] / prev_n - 1.0

    # 2) 波动率：10 日收益率标准差（年化）及其历史分位数
    df["vol_10"] = df.groupby("code", sort=False)["ret_1"].transform(
        lambda s: s.rolling(cfg.vol_window, min_periods=cfg.vol_window).std(ddof=1)
    )
    if cfg.annualize_vol:
        df["vol_10"] *= np.sqrt(TRADING_DAYS_PER_YEAR)

    def hist_quantile(s: pd.Series, q: float) -> pd.Series:
        base = s if cfg.vol_q_include_current else s.shift(1)
        return base.rolling(cfg.hist_window, min_periods=cfg.hist_min_periods).quantile(q)

    df["vol_q30"] = df.groupby("code", sort=False)["vol_10"].transform(
        lambda s: hist_quantile(s, cfg.vol_entry_q)
    )
    df["vol_q40"] = df.groupby("code", sort=False)["vol_10"].transform(
        lambda s: hist_quantile(s, cfg.vol_exit_q)
    )

    # 横截面动量百分位（0=最差，1=最好）
    tradable = df["tradable"].fillna(False).astype(bool)
    df["mom_pct"] = (
        df["ret_20"].where(tradable).groupby(df["date"], sort=False).rank(pct=True)
    )

    # 3) 估值：同日同行业 PE 均值（含自身；剔除亏损/无效 PE）
    pe_valid = df["pe_ttm"].where((df["pe_ttm"] > 0) & tradable)
    tmp = pd.DataFrame(
        {"date": df["date"], "industry": df["industry"], "_pe": pe_valid}
    )
    gb = tmp.groupby(["date", "industry"], sort=False)["_pe"]
    n_valid = gb.transform("count")
    agg = (lambda s: s.mean()) if cfg.industry_stat == "mean" else (lambda s: s.median())
    df["ind_pe"] = gb.transform(agg).where(n_valid >= cfg.min_industry_stocks)
    df["pe_ratio"] = df["pe_ttm"] / df["ind_pe"]

    # 可评估性：数据齐备才做判断，避免 NaN 触发误买/误卖
    df["eligible"] = (
        tradable
        & df["ret_20"].notna()
        & df["mom_pct"].notna()
        & df["pe_ttm"].gt(0)
        & df["ind_pe"].gt(0)
        & df["pe_ratio"].notna()
        & df["vol_10"].notna()
        & df["vol_q30"].notna()
    )
    df["f_mom"] = df["eligible"] & df["mom_pct"].ge(cfg.mom_entry_pct)
    df["f_val"] = df["eligible"] & df["pe_ratio"].lt(cfg.pe_entry_ratio)
    df["f_vol"] = df["eligible"] & df["vol_10"].lt(df["vol_q30"])
    df["qualified"] = df["f_mom"] & df["f_val"] & df["f_vol"]

    df["evaluable"] = (
        tradable
        & df["mom_pct"].notna()
        & df["pe_ratio"].notna()
        & df["vol_10"].notna()
        & df["vol_q40"].notna()
    )
    df["d_mom"] = df["evaluable"] & df["mom_pct"].lt(cfg.mom_exit_pct)
    df["d_val"] = df["evaluable"] & (
        df["pe_ratio"].ge(cfg.pe_exit_multiple) | df["pe_ttm"].le(0)
    )
    df["d_vol"] = df["evaluable"] & df["vol_10"].gt(df["vol_q40"])
    df["deteriorated"] = df["d_mom"] | df["d_val"] | df["d_vol"]

    # 因子得分矩阵：横截面百分位 * 100，方向统一为「越大越好」
    def cs_score(s: pd.Series) -> pd.Series:
        return s.groupby(df["date"], sort=False).rank(pct=True) * 100.0

    mask = df["eligible"]
    df["score_mom"] = cs_score(df["ret_20"].where(mask))
    df["score_val"] = cs_score((-df["pe_ratio"]).where(mask))
    df["score_vol"] = cs_score((-df["vol_10"]).where(mask))

    w = np.array([cfg.w_mom, cfg.w_val, cfg.w_vol], dtype=float)
    w = w / w.sum()
    parts = df[["score_mom", "score_val", "score_vol"]].to_numpy(dtype=float)
    valid_w = np.sum(~np.isnan(parts) * w, axis=1)
    df["score_total"] = np.divide(
        np.nansum(parts * w, axis=1),
        valid_w,
        out=np.full(len(df), np.nan),
        where=valid_w > 0,
    )
    return df


# --------------------------------------------------------------------------- #
# 策略层
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    scores: pd.DataFrame     # 每日全市场因子得分矩阵
    pool: pd.DataFrame       # 每日三因子同时达标的股票池
    orders: pd.DataFrame     # 买入 / 卖出订单
    positions: pd.DataFrame  # 每日持仓
    summary: pd.DataFrame    # 每日汇总

    def save(self, out_dir: Path, excel: bool = False) -> dict[str, Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tables = {
            "factor_scores": self.scores,
            "daily_pool": self.pool,
            "orders": self.orders,
            "positions": self.positions,
            "daily_summary": self.summary,
        }
        paths: dict[str, Path] = {}
        for name, table in tables.items():
            path = out_dir / f"{name}.csv"
            table.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.6f")
            paths[name] = path
        if excel:
            try:
                xlsx = out_dir / "strategy_output.xlsx"
                with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
                    for name, table in tables.items():
                        table.to_excel(writer, sheet_name=name[:31], index=False)
                paths["excel"] = xlsx
            except ImportError:
                print("[warn] 未安装 openpyxl，跳过 Excel 输出（pip install openpyxl）")
        return paths


SCORE_COLUMNS = [
    "date", "code", "industry", "close", "tradable",
    "ret_20", "mom_pct", "pe_ttm", "ind_pe", "pe_ratio",
    "vol_10", "vol_q30", "vol_q40",
    "f_mom", "f_val", "f_vol", "qualified",
    "d_mom", "d_val", "d_vol", "deteriorated", "evaluable", "eligible",
    "score_mom", "score_val", "score_vol", "score_total",
]


def run_strategy(factors: pd.DataFrame, cfg: Config) -> Result:
    """逐日状态机：先按恶化规则卖出，再按三因子规则买入。"""
    scores = factors[SCORE_COLUMNS].copy()
    dates = sorted(factors["date"].unique())
    by_date = {d: sub.set_index("code") for d, sub in factors.groupby("date", sort=True)}
    exec_date = {
        d: (dates[i + cfg.exec_lag] if i + cfg.exec_lag < len(dates) else pd.NaT)
        for i, d in enumerate(dates)
    }

    positions: set[str] = set()
    pool_rows: list[dict] = []
    order_rows: list[dict] = []
    pos_rows: list[dict] = []
    summary_rows: list[dict] = []

    for dt in dates:
        day = by_date[dt]

        # --- 卖出：持仓中任一因子恶化 ---
        sells: list[tuple[str, str]] = []
        for code in sorted(positions):
            if code not in day.index:
                continue  # 停牌 / 数据缺失，不误判卖出
            row = day.loc[code]
            if not bool(row["evaluable"]) or not bool(row["deteriorated"]):
                continue
            reasons = [
                name
                for name, flag in (
                    ("momentum<50%", row["d_mom"]),
                    ("pe>=industry", row["d_val"]),
                    ("vol>40th_pct", row["d_vol"]),
                )
                if bool(flag)
            ]
            sells.append((code, "+".join(reasons)))
        sell_codes = {c for c, _ in sells}
        positions -= sell_codes

        # --- 股票池：三因子同时达标 ---
        q = day[day["qualified"].fillna(False).astype(bool)].sort_values(
            "score_total", ascending=False
        )
        capacity = (
            len(q) if cfg.max_positions <= 0
            else max(cfg.max_positions - len(positions), 0)
        )
        buys = [c for c in q.index if c not in positions and c not in sell_codes][:capacity]
        buy_codes = set(buys)
        positions |= buy_codes

        for code, row in q.iterrows():
            pool_rows.append({
                "date": dt,
                "code": code,
                "industry": row["industry"],
                "score_total": row["score_total"],
                "score_mom": row["score_mom"],
                "score_val": row["score_val"],
                "score_vol": row["score_vol"],
                "action": "BUY" if code in buy_codes else ("HOLD" if code in positions else "WATCH"),
                "exec_date": exec_date[dt] if code in buy_codes else pd.NaT,
            })

        for code, reason in sells:
            order_rows.append({
                "signal_date": dt, "exec_date": exec_date[dt], "code": code,
                "side": "SELL", "reason": reason,
                "score_total": day.at[code, "score_total"] if code in day.index else np.nan,
            })
        for code in buys:
            order_rows.append({
                "signal_date": dt, "exec_date": exec_date[dt], "code": code,
                "side": "BUY", "reason": "momentum+valuation+low_vol",
                "score_total": q.at[code, "score_total"],
            })

        for code in sorted(positions):
            pos_rows.append({"date": dt, "code": code})

        summary_rows.append({
            "date": dt,
            "n_universe": int(day["tradable"].fillna(False).astype(bool).sum()),
            "n_eligible": int(day["eligible"].fillna(False).astype(bool).sum()),
            "n_qualified": int(len(q)),
            "n_buy": len(buys),
            "n_sell": len(sells),
            "n_positions": len(positions),
        })

    def to_df(rows: list[dict], columns: list[str]) -> pd.DataFrame:
        return pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)

    return Result(
        scores=scores,
        pool=to_df(pool_rows, [
            "date", "code", "industry", "score_total", "score_mom", "score_val",
            "score_vol", "action", "exec_date",
        ]),
        orders=to_df(order_rows, [
            "signal_date", "exec_date", "code", "side", "reason", "score_total",
        ]),
        positions=to_df(pos_rows, ["date", "code"]),
        summary=to_df(summary_rows, [
            "date", "n_universe", "n_eligible", "n_qualified", "n_buy", "n_sell", "n_positions",
        ]),
    )


# --------------------------------------------------------------------------- #
# 自检 / 演示数据
# --------------------------------------------------------------------------- #
def _factors_from_raw(price, pe, industry, cfg):
    return compute_factors(prepare_panel(price, pe, industry, cfg), cfg)


def assert_no_lookahead(price, pe, industry, cfg) -> None:
    """截断最后一个交易日后重算，重叠区间必须逐值一致（前向偏差检验）。"""
    price, pe = price.copy(), pe.copy()
    price["date"] = pd.to_datetime(price["date"])
    pe["date"] = pd.to_datetime(pe["date"])
    full = _factors_from_raw(price, pe, industry, cfg)
    last = full["date"].max()
    cut = _factors_from_raw(
        price[price["date"] < last], pe[pe["date"] < last], industry, cfg
    )
    a = full[full["date"] < last].reset_index(drop=True)
    b = cut.reset_index(drop=True)
    cols = [c for c in a.columns if c not in ("date", "code")]
    pd.testing.assert_frame_equal(
        a[cols], b[cols], check_dtype=False, rtol=1e-9, atol=1e-12
    )


def self_test(price, pe, industry, cfg, result: Result) -> None:
    assert_no_lookahead(price, pe, industry, cfg)

    flags = result.scores[["date", "code", "f_mom", "f_val", "f_vol",
                           "d_mom", "d_val", "d_vol", "qualified"]]
    orders = result.orders.copy()
    orders["date"] = orders["signal_date"]
    buys = orders[orders["side"] == "BUY"].merge(
        flags, on=["date", "code"], how="left"
    )
    if len(buys):
        assert buys[["f_mom", "f_val", "f_vol"]].fillna(False).all(axis=1).all(), \
            "买入信号存在未达标因子"
    sells = orders[orders["side"] == "SELL"].merge(
        flags, on=["date", "code"], how="left"
    )
    if len(sells):
        assert sells[["d_mom", "d_val", "d_vol"]].fillna(False).any(axis=1).all(), \
            "卖出信号缺少恶化因子"
    if len(result.pool):
        merged = result.pool.merge(flags[["date", "code", "qualified"]],
                                   on=["date", "code"], how="left")
        assert merged["qualified"].fillna(False).all(), "股票池混入未达标标的"
    if cfg.max_positions > 0:
        assert result.summary["n_positions"].max() <= cfg.max_positions, "持仓超限"
    dup = result.positions.duplicated(["date", "code"]).sum()
    assert dup == 0, "持仓表存在重复"
    print("[self-test] 前向偏差 / 买入条件 / 卖出条件 / 股票池 / 持仓上限：全部通过")


def make_demo_data(
    n_codes: int = 80, n_days: int = 420, seed: int = 20261108
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """生成【合成随机数据】，仅用于验证代码可运行，绝非真实行情或真实标的。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days, name="date")
    codes = [f"SYN{i:03d}" for i in range(1, n_codes + 1)]
    n_ind = 8
    industry = pd.DataFrame({
        "code": codes,
        "industry": [f"IND_{i % n_ind}" for i in range(n_codes)],
    })

    price_frames = []
    for j in range(n_codes):
        drift = rng.normal(0.0004, 0.0006)
        sigma = rng.uniform(0.010, 0.030)
        vol_scale = 1.0 + 0.8 * np.sin(np.arange(n_days) / 40.0 + j)
        shocks = rng.normal(drift, sigma, n_days) * vol_scale
        close = 10.0 * np.exp(np.cumsum(shocks))
        price_frames.append(pd.DataFrame({
            "date": dates, "code": codes[j], "close": np.round(close, 4),
            "volume": rng.integers(100_000, 1_000_000, n_days),
        }))
    price = pd.concat(price_frames, ignore_index=True)

    base_pe = {f"IND_{i}": float(rng.uniform(12, 40)) for i in range(n_ind)}
    pe_dates = dates[::5]  # 估值数据每 5 个交易日更新一次
    pe_frames = []
    for j, code in enumerate(codes):
        discount = rng.uniform(0.50, 1.45)
        path = base_pe[f"IND_{j % n_ind}"] * discount * (
            1.0 + np.cumsum(rng.normal(0, 0.005, len(pe_dates)))
        )
        pe_frames.append(pd.DataFrame({
            "date": pe_dates, "code": code, "pe_ttm": np.round(path, 4),
        }))
    pe = pd.concat(pe_frames, ignore_index=True)
    return price, pe, industry


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="三因子选股策略：动量 / 估值 / 波动率")
    ap.add_argument("--data-dir", type=Path, help="含 prices.csv / pe_ttm.csv / industry.csv 的目录")
    ap.add_argument("--demo", action="store_true", help="用合成随机数据跑通全流程（非真实行情）")
    ap.add_argument("--out-dir", type=Path, default=Path("./strategy_output"))
    ap.add_argument("--start", default=None, help="回测起始日 YYYY-MM-DD")
    ap.add_argument("--max-positions", type=int, default=0, help="最大持仓数，0=不限")
    ap.add_argument("--self-test", action="store_true", help="额外执行自检")
    ap.add_argument("--excel", action="store_true", help="额外输出多 sheet Excel")
    args = ap.parse_args(argv)
    if not args.demo and args.data_dir is None:
        ap.error("需指定 --demo 或 --data-dir")

    cfg = replace(Config(), max_positions=args.max_positions)
    if args.demo:
        price, pe, industry = make_demo_data()
        print("[demo] 使用合成随机数据（SYN*/IND_*），仅验证代码逻辑，非真实行情")
    else:
        price, pe, industry = load_inputs(args.data_dir)

    raw = _factors_from_raw(price, pe, industry, cfg)
    factors = raw[raw["date"] >= pd.Timestamp(args.start)].copy() if args.start else raw
    if factors.empty:
        raise ValueError("起始日之后无有效数据，请检查 --start 或数据区间")

    result = run_strategy(factors, cfg)
    if args.demo or args.self_test:
        check = result if not args.start else run_strategy(raw, cfg)
        self_test(price, pe, industry, cfg, check)
    paths = result.save(args.out_dir, excel=args.excel)

    n_buy = int((result.orders["side"] == "BUY").sum())
    n_sell = int((result.orders["side"] == "SELL").sum())
    print(f"交易日 {result.summary['date'].nunique()} | 得分矩阵 {len(result.scores)} 行 | "
          f"股票池 {len(result.pool)} 行 | BUY {n_buy} | SELL {n_sell}")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
