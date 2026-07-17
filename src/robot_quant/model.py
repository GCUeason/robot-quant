"""无前视偏差的月度滚动预测模型。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class PredictionConfig:
    """滚动预测参数。"""

    horizon_days: int = 10
    minimum_training_samples: int = 252
    half_position_probability: float = 0.50
    full_position_probability: float = 0.60
    logistic_c: float = 0.10


class WalkForwardPredictor:
    """使用当时已知数据按月重新训练逻辑回归模型。"""

    FEATURE_COLUMNS = (
        "return_5",
        "return_20",
        "return_60",
        "distance_ma20",
        "distance_ma60",
        "volatility_20",
        "volume_ratio_20",
        "relative_strength_20",
        "market_trend_120",
    )

    def __init__(self, config: PredictionConfig | None = None) -> None:
        self.config = config or PredictionConfig()

    def predict_history(
        self,
        robot_index: pd.DataFrame,
        benchmark: pd.DataFrame,
    ) -> pd.DataFrame:
        """生成逐日概率和目标仓位；训练标签必须在预测日前已经实现。"""
        dataset = self._build_dataset(robot_index, benchmark)
        records: list[dict[str, float | str | pd.Timestamp]] = []

        for _, monthly_data in dataset.groupby(dataset.index.to_period("M"), sort=True):
            training_cutoff = monthly_data.index[0]
            train_mask = dataset["label_known_date"].lt(training_cutoff)
            train_data = dataset.loc[train_mask].dropna(
                subset=[*self.FEATURE_COLUMNS, "label"]
            )
            model = self._fit_model(train_data)

            for date, row in monthly_data.iterrows():
                features = row.loc[list(self.FEATURE_COLUMNS)]
                if model is not None and features.notna().all():
                    feature_frame = features.to_frame().T.astype(float)
                    probability = float(
                        model.predict_proba(feature_frame)[0, 1]
                    )
                    model_kind = "logistic_regression"
                else:
                    probability = self._fallback_probability(row)
                    model_kind = "trend_fallback"

                target_weight = self._target_weight(
                    probability=probability,
                    market_trend=float(row["market_trend_120"]),
                )
                records.append(
                    {
                        "date": pd.Timestamp(date),
                        "probability": probability,
                        "target_weight": target_weight,
                        "model_kind": model_kind,
                        "realized_label": float(row["label"]),
                    }
                )

        return pd.DataFrame.from_records(records).set_index("date")

    def _build_dataset(
        self,
        robot_index: pd.DataFrame,
        benchmark: pd.DataFrame,
    ) -> pd.DataFrame:
        robot = robot_index.sort_index().copy()
        market = benchmark.sort_index().copy()
        common_index = robot.index.intersection(market.index)
        robot = robot.loc[common_index]
        market = market.loc[common_index]
        if robot.empty:
            raise ValueError("机器人指数与基准指数没有重合日期")

        robot_close = robot["close"].astype(float)
        market_close = market["close"].astype(float)
        robot_volume = (
            robot["volume"].astype(float)
            if "volume" in robot.columns
            else pd.Series(1.0, index=robot.index)
        )
        daily_return = robot_close.pct_change()

        dataset = pd.DataFrame(index=common_index)
        dataset["return_5"] = robot_close.pct_change(5)
        dataset["return_20"] = robot_close.pct_change(20)
        dataset["return_60"] = robot_close.pct_change(60)
        dataset["distance_ma20"] = robot_close / robot_close.rolling(20).mean() - 1.0
        dataset["distance_ma60"] = robot_close / robot_close.rolling(60).mean() - 1.0
        dataset["volatility_20"] = daily_return.rolling(20).std() * np.sqrt(252.0)
        dataset["volume_ratio_20"] = robot_volume / robot_volume.rolling(20).mean()
        dataset["relative_strength_20"] = (
            robot_close.pct_change(20) - market_close.pct_change(20)
        )
        dataset["market_trend_120"] = (
            market_close / market_close.rolling(120).mean() - 1.0
        )

        horizon = self.config.horizon_days
        future_robot_return = robot_close.shift(-horizon) / robot_close - 1.0
        future_market_return = market_close.shift(-horizon) / market_close - 1.0
        future_excess_return = future_robot_return - future_market_return
        dataset["label"] = (future_excess_return > 0.0).astype(float)
        dataset.loc[future_excess_return.isna(), "label"] = np.nan
        dataset["label_known_date"] = pd.Series(
            common_index,
            index=common_index,
        ).shift(-horizon)
        return dataset

    def _fit_model(self, training_data: pd.DataFrame) -> Pipeline | None:
        if len(training_data) < self.config.minimum_training_samples:
            return None
        labels = training_data["label"].astype(int)
        if labels.nunique() < 2:
            return None

        model = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=self.config.logistic_c,
                        class_weight="balanced",
                        max_iter=500,
                        random_state=0,
                    ),
                ),
            ]
        )
        model.fit(training_data.loc[:, self.FEATURE_COLUMNS], labels)
        return model

    def _fallback_probability(self, row: pd.Series) -> float:
        probability = 0.50
        probability += 0.08 if row.get("distance_ma20", np.nan) > 0.0 else -0.08
        probability += 0.08 if row.get("distance_ma60", np.nan) > 0.0 else -0.08
        probability += 0.08 if row.get("relative_strength_20", np.nan) > 0.0 else -0.08
        probability += 0.08 if row.get("market_trend_120", np.nan) > 0.0 else -0.08
        if row.get("volatility_20", np.nan) > 0.35:
            probability -= 0.05
        return float(np.clip(probability, 0.05, 0.95))

    def _target_weight(self, probability: float, market_trend: float) -> float:
        if (
            probability >= self.config.full_position_probability
            and market_trend > 0.0
        ):
            return 1.0
        if probability >= self.config.half_position_probability:
            return 0.5
        return 0.0
