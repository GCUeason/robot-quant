"""公开行情数据读取。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass(frozen=True)
class MarketBundle:
    """模型运行所需的三组日线行情。"""

    etf: pd.DataFrame
    robot_index: pd.DataFrame
    benchmark: pd.DataFrame


class TencentDataSource:
    """从腾讯证券公开行情接口读取日线。"""

    endpoint = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"

    def __init__(self) -> None:
        retry = Retry(
            total=4,
            connect=4,
            read=4,
            status=4,
            backoff_factor=0.8,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/138.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json,text/plain,*/*",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Connection": "close",
                "Referer": "https://gu.qq.com/",
            }
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)

    def fetch_daily(
        self,
        symbol: str,
        start_date: date = date(2024, 1, 1),
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """按自然年分段读取，避免单次条数上限截断早期历史。"""
        final_date = end_date or date.today()
        if final_date < start_date:
            raise ValueError("行情结束日期不能早于开始日期")

        frames: list[pd.DataFrame] = []
        for year in range(start_date.year, final_date.year + 1):
            segment_start = max(start_date, date(year, 1, 1))
            segment_end = min(final_date, date(year, 12, 31))
            params = {
                "param": (
                    f"{symbol},day,{segment_start.isoformat()},"
                    f"{segment_end.isoformat()},400,qfq"
                )
            }
            try:
                response = self.session.get(
                    self.endpoint,
                    params=params,
                    timeout=(10, 30),
                )
                response.raise_for_status()
                frames.append(self._parse_payload(response.json(), symbol))
            except requests.RequestException as error:
                raise RuntimeError(f"获取行情失败: {symbol}") from error

        combined = pd.concat(frames).sort_index()
        return combined.loc[~combined.index.duplicated(keep="last")]

    @staticmethod
    def _parse_payload(payload: dict, symbol: str) -> pd.DataFrame:
        symbol_data = (payload.get("data") or {}).get(symbol) or {}
        rows = symbol_data.get("qfqday") or symbol_data.get("day")
        if payload.get("code") != 0 or not rows:
            raise ValueError(f"行情返回为空: {symbol}")

        columns = ["date", "open", "close", "high", "low", "volume"]
        frame = pd.DataFrame(rows, columns=columns)
        frame["date"] = pd.to_datetime(frame["date"])
        for column in columns[1:]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame.dropna(subset=["open", "close"]).set_index("date").sort_index()


def read_offline_bundle(directory: Path) -> MarketBundle:
    """读取可复现测试或离线运行使用的CSV行情。"""

    return MarketBundle(
        etf=_read_csv(directory / "etf.csv"),
        robot_index=_read_csv(directory / "robot_index.csv"),
        benchmark=_read_csv(directory / "benchmark.csv"),
    )


def fetch_live_bundle() -> MarketBundle:
    """读取159530和沪深300；直接预测可交易ETF的相对收益。"""

    source = TencentDataSource()
    etf = source.fetch_daily("sz159530")
    return MarketBundle(
        etf=etf,
        robot_index=etf.copy(),
        benchmark=source.fetch_daily("sh000300"),
    )


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    return frame.set_index("date").sort_index()
