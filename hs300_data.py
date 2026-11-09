#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建 2018-2023 沪深300成分股回测数据集（真实行情，本地缓存）。

数据源
------
* 历史成分：baostock `query_hs300_stocks(date)`，按 2017-07 ~ 2023-12 逐月采样，
  按中证指数公布的实际生效日去重，得到「生效日 -> 成分股」期间表。
  => 避免用 2026 年当前成分回溯 2018 年产生的幸存者偏差 / 前视偏差。
* 个股日线：akshare `stock_zh_a_hist`（东方财富），前复权 qfq。
* 基准指数：akshare `index_zh_a_hist`（沪深300 000300）。
* 多取 2017-07-01 起半年作为指标热身期（RSI/MACD/BOLL 需要前置样本），
  回测统计区间仍严格为 2018-01-01 ~ 2023-12-31。

输出（./data/）
--------------
membership.csv : effective_date, code       成分股生效期间表
prices.csv     : date, code, open, high, low, close, volume
index.csv      : date, open, high, low, close
meta.json      : 数据源 / 口径 / 拉取时间 / 失败清单

用法：python hs300_data.py [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DATA_DIR = Path(__file__).resolve().parent / "data"
WARMUP_START = "2017-07-01"
BACKTEST_START = "2018-01-01"
END = "2023-12-31"


def build_membership() -> tuple[pd.DataFrame, str]:
    """按月采样历史成分，按实际生效日去重。失败时降级为当前成分快照。"""
    try:
        import baostock as bs

        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"baostock login failed: {lg.error_msg}")

        periods: dict[str, list[str]] = {}
        months = pd.date_range(WARMUP_START, "2023-12-01", freq="MS")
        for m in months:
            day = m.strftime("%Y-%m-%d")
            rs = bs.query_hs300_stocks(day)
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                continue
            effective = rows[0][0]  # 该期成分的实际生效日
            codes = [r[1].split(".")[-1] for r in rows]
            periods.setdefault(effective, codes)
        bs.logout()
        if len(periods) < 4:
            raise RuntimeError("历史成分期数过少")

        frames = [
            pd.DataFrame({"effective_date": [eff] * len(codes), "code": codes})
            for eff, codes in sorted(periods.items())
        ]
        out = pd.concat(frames, ignore_index=True)
        out = out.drop_duplicates(["effective_date", "code"])
        return out, "baostock_historical_monthly_snapshot"
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 历史成分获取失败（{type(exc).__name__}: {exc}），降级为当前成分快照")
        import akshare as ak

        cons = ak.index_stock_cons(symbol="000300")
        out = pd.DataFrame({
            "effective_date": [WARMUP_START] * len(cons),
            "code": cons["品种代码"].astype(str).str.zfill(6),
        })
        return out, "akshare_current_snapshot_FALLBACK_survivorship_bias"


def _fetch_one(code: str) -> tuple[str, pd.DataFrame | None]:
    import akshare as ak

    rename = {
        "日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
        "最低": "low", "成交量": "volume",
    }
    for attempt in range(3):
        try:
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=WARMUP_START.replace("-", ""),
                end_date=END.replace("-", ""),
                adjust="qfq",
            )
            if df is None or df.empty:
                return code, None
            df = df.rename(columns=rename)
            df["code"] = code
            df["date"] = pd.to_datetime(df["date"])
            return code, df[["date", "code", "open", "high", "low", "close", "volume"]]
        except Exception:
            time.sleep(0.4 * (attempt + 1))
    return code, None


def build_prices(codes: list[str]) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, (code, df) in enumerate(pool.map(_fetch_one, codes), start=1):
            if df is None:
                failed.append(code)
            else:
                frames.append(df)
            if i % 50 == 0:
                print(f"  已拉取 {i}/{len(codes)} 只")
    prices = pd.concat(frames, ignore_index=True)
    prices = prices.sort_values(["code", "date"]).reset_index(drop=True)
    return prices, failed


def build_index() -> pd.DataFrame:
    import akshare as ak

    df = ak.index_zh_a_hist(
        symbol="000300", period="daily",
        start_date=WARMUP_START.replace("-", ""), end_date=END.replace("-", ""),
    )
    df = df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close",
                            "最高": "high", "最低": "low"})
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "open", "high", "low", "close"]].sort_values("date").reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="忽略缓存重新拉取")
    args = ap.parse_args(argv)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    price_path = DATA_DIR / "prices.csv"
    if price_path.exists() and not args.force:
        print(f"缓存已存在：{price_path}（--force 可强制重建）")
        return 0

    print("[1/3] 构建历史成分股期间表 ...")
    membership, mode = build_membership()
    codes = sorted(membership["code"].unique())
    print(f"  成分生效期 {membership['effective_date'].nunique()} 个，去重股票 {len(codes)} 只，口径={mode}")

    print("[2/3] 拉取个股前复权日线 ...")
    prices, failed = build_prices(codes)
    print(f"  成功 {prices['code'].nunique()} 只 / 失败 {len(failed)} 只，共 {len(prices)} 行")

    print("[3/3] 拉取沪深300指数 ...")
    index = build_index()

    membership.to_csv(DATA_DIR / "membership.csv", index=False, encoding="utf-8-sig")
    prices.to_csv(DATA_DIR / "prices.csv", index=False, encoding="utf-8-sig")
    index.to_csv(DATA_DIR / "index.csv", index=False, encoding="utf-8-sig")

    meta = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "warmup_start": WARMUP_START,
        "backtest_start": BACKTEST_START,
        "backtest_end": END,
        "membership_mode": mode,
        "membership_effective_dates": sorted(membership["effective_date"].unique().tolist()),
        "price_source": "akshare stock_zh_a_hist (EastMoney), qfq",
        "index_source": "akshare index_zh_a_hist (000300)",
        "n_codes_requested": len(codes),
        "n_codes_ok": int(prices["code"].nunique()),
        "n_rows": int(len(prices)),
        "failed_codes": failed,
        "notes": "前复权价格；指标热身期 2017-07-01 起，绩效统计仅 2018-01-01~2023-12-31。",
    }
    (DATA_DIR / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"完成：{DATA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
