"""无前视偏差的月度滚动预测模型。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class PredictionConfig:
    """滚动预测参数。"""

    horizon_days: int = 10
    minimum_training_samples: int = 252
    half_position_probability: float = 0.50
    full_position_probability: float = 0.60
    logistic_c: float = 0.01
    probability_shrinkage: float = 0.25
    calibration_splits: int = 5
    calibration_gap: int = 10
    ood_lower_quantile: float = 0.01
    ood_upper_quantile: float = 0.99
    quality_window_size: int = 50
    maximum_shadow_weight: float = 0.20


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
    MODEL_KIND = "shrunk_logistic_regression"
    MODEL_VERSION = "trend_risk_logistic_v3"
    MODEL_SELECTION_STATUS = "retrospective_challenger"

    def __init__(self, config: PredictionConfig | None = None) -> None:
        self.config = config or PredictionConfig()

    def predict_history(
        self,
        robot_index: pd.DataFrame,
        benchmark: pd.DataFrame,
    ) -> pd.DataFrame:
        """生成逐日概率和目标仓位；训练标签必须在预测日前已经实现。"""
        dataset = self._build_dataset(robot_index, benchmark)
        records: list[dict[str, bool | float | str | pd.Timestamp]] = []

        for _, monthly_data in dataset.groupby(dataset.index.to_period("M"), sort=True):
            training_cutoff = monthly_data.index[0]
            train_mask = dataset["label_known_date"].lt(training_cutoff)
            train_data = dataset.loc[train_mask].dropna(subset=[*self.FEATURE_COLUMNS, "label"])
            raw_model = self._fit_model(train_data)
            training_validation = self._training_validation(train_data)

            for date, row in monthly_data.iterrows():
                features = row.loc[list(self.FEATURE_COLUMNS)]
                if raw_model is not None and features.notna().all():
                    feature_frame = features.to_frame().T.astype(float)
                    raw_probability = float(raw_model.predict_proba(feature_frame)[0, 1])
                    probability = 0.5 + self.config.probability_shrinkage * (raw_probability - 0.5)
                    model_kind = self.MODEL_KIND
                else:
                    probability = self._fallback_probability(row)
                    raw_probability = probability
                    model_kind = "trend_fallback"

                research_target_weight = self._target_weight(
                    probability=probability,
                    market_trend=float(row["market_trend_120"]),
                )
                distribution_lower, distribution_upper = (
                    self._distribution_bounds(train_data)
                    if raw_model is not None and features.notna().all()
                    else (
                        pd.Series(np.nan, index=self.FEATURE_COLUMNS, dtype=float),
                        pd.Series(np.nan, index=self.FEATURE_COLUMNS, dtype=float),
                    )
                )
                ood_features = (
                    self._out_of_distribution_features(features, train_data)
                    if raw_model is not None and features.notna().all()
                    else []
                )
                signal_status = (
                    "out_of_distribution"
                    if ood_features
                    else (
                        "observation_only" if model_kind == self.MODEL_KIND else "model_unavailable"
                    )
                )
                record: dict[str, bool | float | str | pd.Timestamp] = {
                    "date": pd.Timestamp(date),
                    "raw_probability": raw_probability,
                    "probability": probability,
                    "target_weight": research_target_weight,
                    "research_target_weight": research_target_weight,
                    "model_kind": model_kind,
                    "model_version": self.MODEL_VERSION,
                    "model_selection_status": self.MODEL_SELECTION_STATUS,
                    "signal_status": signal_status,
                    "is_out_of_distribution": bool(ood_features),
                    "ood_features": ",".join(ood_features),
                    "ood_training_sample_count": int(len(train_data)),
                    "probability_shrinkage": (
                        self.config.probability_shrinkage if model_kind == self.MODEL_KIND else 0.0
                    ),
                    "training_oof_sample_count": training_validation["sample_count"],
                    "training_oof_brier": training_validation["brier"],
                    "training_oof_auc": training_validation["roc_auc"],
                    "training_oof_passed": training_validation["passed"],
                    "realized_label": float(row["label"]),
                    "label_known_date": row["label_known_date"],
                }
                record.update({feature: float(row[feature]) for feature in self.FEATURE_COLUMNS})
                for feature in self.FEATURE_COLUMNS:
                    record[f"{feature}_ood_lower"] = float(distribution_lower[feature])
                    record[f"{feature}_ood_upper"] = float(distribution_upper[feature])
                records.append(record)

        result = pd.DataFrame.from_records(records).set_index("date")
        return self._apply_mature_quality_gate(result)

    def _out_of_distribution_features(
        self,
        features: pd.Series,
        training_data: pd.DataFrame,
    ) -> list[str]:
        """仅用本月训练样本的分位边界识别不可安全外推的特征。"""
        lower, upper = self._distribution_bounds(training_data)
        lower_label = _quantile_label(self.config.ood_lower_quantile)
        upper_label = _quantile_label(self.config.ood_upper_quantile)
        breaches: list[str] = []
        for column in self.FEATURE_COLUMNS:
            value = float(features[column])
            if value < float(lower[column]):
                breaches.append(f"{column}:below_{lower_label}")
            elif value > float(upper[column]):
                breaches.append(f"{column}:above_{upper_label}")
        return breaches

    def _distribution_bounds(
        self,
        training_data: pd.DataFrame,
    ) -> tuple[pd.Series, pd.Series]:
        """返回与OOD判断完全一致的训练分位边界，供报告解释。"""
        training_features = training_data.loc[:, self.FEATURE_COLUMNS].astype(float)
        return (
            training_features.quantile(self.config.ood_lower_quantile),
            training_features.quantile(self.config.ood_upper_quantile),
        )

    def _training_validation(self, training_data: pd.DataFrame) -> dict:
        """只用训练集内部的带隔离时序折记录候选稳健性。"""
        if (
            len(training_data) < self.config.minimum_training_samples
            or training_data["label"].nunique() < 2
        ):
            return {
                "sample_count": 0,
                "brier": None,
                "roc_auc": None,
                "passed": False,
            }
        records: list[tuple[float, int]] = []
        try:
            splitter = TimeSeriesSplit(
                n_splits=self.config.calibration_splits,
                gap=self.config.calibration_gap,
            )
            for train_positions, validation_positions in splitter.split(training_data):
                fold_train = training_data.iloc[train_positions]
                fold_validation = training_data.iloc[validation_positions]
                model = self._fit_model(fold_train)
                if model is None:
                    continue
                raw = model.predict_proba(fold_validation.loc[:, self.FEATURE_COLUMNS])[:, 1]
                probability = 0.5 + self.config.probability_shrinkage * (raw - 0.5)
                records.extend(
                    (float(score), int(label))
                    for score, label in zip(
                        probability,
                        fold_validation["label"].astype(int),
                        strict=True,
                    )
                )
        except ValueError:
            records = []
        if not records:
            return {
                "sample_count": 0,
                "brier": None,
                "roc_auc": None,
                "passed": False,
            }
        scores = pd.Series([record[0] for record in records], dtype=float)
        labels = pd.Series([record[1] for record in records], dtype=int)
        brier = float(((scores - labels) ** 2).mean())
        auc = float(roc_auc_score(labels, scores)) if labels.nunique() == 2 else None
        return {
            "sample_count": len(records),
            "brier": brier,
            "roc_auc": auc,
            "passed": brier < 0.25 and auc is not None and auc > 0.5,
        }

    def _apply_mature_quality_gate(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """逐日仅用已经成熟的预测结果决定不可执行的影子仓位。"""
        result = predictions.sort_index().copy()
        gate_records: list[dict[str, bool | float | int | str]] = []
        for date, current in result.iterrows():
            matured = result[
                result["label_known_date"].lt(date)
                & result["realized_label"].notna()
                & result["model_kind"].eq(self.MODEL_KIND)
            ].tail(self.config.quality_window_size)
            sample_count = len(matured)
            if sample_count:
                labels = matured["realized_label"].astype(int)
                scores = matured["probability"].astype(float)
                brier = float(((scores - labels) ** 2).mean())
                auc = (
                    float(roc_auc_score(labels, scores))
                    if labels.nunique() == 2 and scores.nunique() > 1
                    else None
                )
            else:
                brier = None
                auc = None
            performance_passed = (
                sample_count >= self.config.quality_window_size
                and brier is not None
                and brier < 0.25
                and auc is not None
                and auc > 0.5
            )
            quality_gate_passed = (
                performance_passed
                and not bool(current["is_out_of_distribution"])
                and current["model_kind"] == self.MODEL_KIND
            )
            if bool(current["is_out_of_distribution"]):
                reason = f"当前训练分布外：{current['ood_features']}"
            elif sample_count < self.config.quality_window_size:
                reason = f"成熟样本不足{self.config.quality_window_size}条（当前{sample_count}条）"
            elif not performance_passed:
                reason = "最近成熟窗口未同时通过Brier<0.25与AUC>0.5"
            elif current["model_kind"] != self.MODEL_KIND:
                reason = "当前稳健模型不可用"
            else:
                reason = "最近成熟窗口通过，仅允许不可执行影子卫星仓"
            shadow_weight = (
                min(
                    float(current["research_target_weight"]),
                    self.config.maximum_shadow_weight,
                )
                if quality_gate_passed
                else 0.0
            )
            gate_records.append(
                {
                    "quality_gate_sample_count": sample_count,
                    "quality_gate_brier": brier,
                    "quality_gate_auc": auc,
                    "quality_gate_performance_passed": performance_passed,
                    "quality_gate_passed": quality_gate_passed,
                    "quality_gate_reason": reason,
                    "shadow_target_weight": shadow_weight,
                }
            )
        return result.join(pd.DataFrame(gate_records, index=result.index))

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
        dataset["relative_strength_20"] = robot_close.pct_change(20) - market_close.pct_change(20)
        dataset["market_trend_120"] = market_close / market_close.rolling(120).mean() - 1.0

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

        model = self._new_model()
        model.fit(training_data.loc[:, self.FEATURE_COLUMNS], labels)
        return model

    def _new_model(self) -> Pipeline:
        return Pipeline(
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
        if probability >= self.config.full_position_probability and market_trend > 0.0:
            return 1.0
        if probability >= self.config.half_position_probability:
            return 0.5
        return 0.0


def _quantile_label(quantile: float) -> str:
    return f"{quantile * 100:g}pct"
