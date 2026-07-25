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
    etf_source: str
    robot_index_source: str
    benchmark_source: str


def _retry_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/138.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Connection": "close",
        }
    )
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


class TencentDataSource:
    """从腾讯证券公开行情接口读取日线。"""

    endpoint = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"

    def __init__(self) -> None:
        self.session = _retry_session()
        self.session.headers.update(
            {
                "Referer": "https://gu.qq.com/",
            }
        )

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
                    f"{symbol},day,{segment_start.isoformat()},{segment_end.isoformat()},400,qfq"
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


class CnIndexDataSource:
    """从国证指数官网公开行情接口读取官方指数日线。"""

    endpoint = "https://hq.cnindex.com.cn/market/market/getIndexDailyDataWithDataFormat"

    def __init__(self) -> None:
        self.session = _retry_session()
        self.session.headers.update({"Referer": "https://www.cnindex.com.cn/"})

    def fetch_daily(
        self,
        index_code: str,
        start_date: date = date(2015, 1, 1),
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """读取官方指数历史；980022基日为2014-12-31。"""
        final_date = end_date or date.today()
        if final_date < start_date:
            raise ValueError("行情结束日期不能早于开始日期")
        try:
            response = self.session.get(
                self.endpoint,
                params={
                    "indexCode": index_code,
                    "startDate": start_date.isoformat(),
                    "endDate": final_date.isoformat(),
                    "frequency": "day",
                },
                timeout=(10, 30),
            )
            response.raise_for_status()
            return self._parse_payload(response.json(), index_code)
        except requests.RequestException as error:
            raise RuntimeError(f"获取国证指数行情失败: {index_code}") from error

    @staticmethod
    def _parse_payload(payload: dict, index_code: str) -> pd.DataFrame:
        rows = (payload.get("data") or {}).get("data") or []
        if payload.get("code") != 200 or not rows:
            raise ValueError(f"国证指数行情返回为空: {index_code}")
        if any(len(row) < 10 for row in rows):
            raise ValueError(f"国证指数行情字段不足: {index_code}")

        columns = [
            "date",
            "previous_close",
            "high",
            "open",
            "low",
            "close",
            "change",
            "percent",
            "amount",
            "volume",
        ]
        frame = pd.DataFrame([row[:10] for row in rows], columns=columns)
        frame["date"] = pd.to_datetime(frame["date"])
        numeric_columns = ["open", "close", "high", "low", "volume", "amount"]
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return (
            frame.loc[:, ["date", *numeric_columns]]
            .dropna(subset=["open", "close"])
            .set_index("date")
            .sort_index()
        )


def read_offline_bundle(directory: Path) -> MarketBundle:
    """读取可复现测试或离线运行使用的CSV行情。"""

    return MarketBundle(
        etf=_read_csv(directory / "etf.csv"),
        robot_index=_read_csv(directory / "robot_index.csv"),
        benchmark=_read_csv(directory / "benchmark.csv"),
        etf_source="offline_etf_csv",
        robot_index_source="offline_robot_index_csv",
        benchmark_source="offline_benchmark_csv",
    )


def fetch_live_bundle() -> MarketBundle:
    """读取159530、官方980022与沪深300长历史。"""

    tencent = TencentDataSource()
    cnindex = CnIndexDataSource()
    start_date = date(2015, 1, 1)
    return MarketBundle(
        etf=tencent.fetch_daily("sz159530"),
        robot_index=cnindex.fetch_daily("980022", start_date=start_date),
        benchmark=tencent.fetch_daily("sh000300", start_date=start_date),
        etf_source="tencent_qfq_sz159530",
        robot_index_source="cnindex_official_980022",
        benchmark_source="tencent_qfq_sh000300",
    )


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    return frame.set_index("date").sort_index()
