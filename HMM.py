
import os
import importlib.util
import sysconfig
import pickle
import io
import contextlib
import sys
import math
from tqdm.auto import tqdm

sys.dont_write_bytecode = True



CPU_COUNT = os.cpu_count() or 1
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_COUNT))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_COUNT))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_COUNT))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(CPU_COUNT))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(CPU_COUNT))

import random
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

try:
    from hmmlearn.hmm import GaussianHMM, PoissonHMM
    HMM_AVAILABLE = True
except Exception:
    GaussianHMM = None
    PoissonHMM = None
    HMM_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False


def can_use_torch_compile():
    if DEVICE.type != "cuda":
        return False
    if os.name == "nt":
        return False
    include_dir = sysconfig.get_path("include")
    if not include_dir or not os.path.isfile(os.path.join(include_dir, "Python.h")):
        return False
    return importlib.util.find_spec("triton") is not None


# ============================================================
# CONFIGURATION
# ============================================================


# The reported experiment set intentionally excludes the Student-t HMM and
# rule-based volatility variant. Their implementations remain available for
# separate development experiments, but they are not part of this study.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILES = ["BTCUSD_daily.pkl", "ETHUSD_daily.pkl"]
DATA_PATHS = [os.path.join(SCRIPT_DIR, data_file) for data_file in DATA_FILES]
POOL_PROCESSES = 10
FEE_RATE = 0.0001
EPISODES = 300
N_ENVS = 300
BATCH_SIZE = 2048
REPLAY_CAPACITY = 500000
TRAIN_EVERY = 100
GRADIENT_STEPS = 1
WARMUP = 0
INITIAL_EQUITY = 1000.0
DRAWDOWN_ABS_PENALTY = 2
DRAWDOWN_INCREASE_PENALTY = 1
REWARD_CLIP = 1.0
TRAIN_PERIOD = ("2017-07-01", "2021-07-01")
TEST_PERIOD = ("2021-07-01", "2023-03-01")
FEATURE_LOOKBACK_BARS = 250
COMPILE_MODEL = False
State_n = 3
SEED = 0
LEAKAGE_PROTOCOL_VERSION = "chronological-regime-crossfit-v1"
REGIME_MIN_FIT_BARS = 252
REGIME_REFIT_BARS = 252







ACTION_BUY = 0
ACTION_SELL = 1
ACTION_HOLD = 2
N_ACTIONS = 3
REGIME_NONE = "none"
REGIME_RULE_VOL = "rule_vol"
REGIME_RULE_TREND = "rule_trend"
REGIME_KMEANS = "kmeans"
REGIME_GMM = "gmm"
REGIME_HMM = "hmm"
REGIME_POISSON_HMM = "poisson_hmm"
REGIME_XGBOOST = "xgboost"
REGIME_STUDENT_T_HMM = "student_t_hmm"
REGIME_RNN = "rnn"
REGIME_LSTM = "lstm"
REGIME_CNN = "cnn"



STICKY_T_HMM_FEATURES = [
    "ret1",
    "atr_pct",
    "vol_20",
    "mom_10",
    "range_pct",
    "trend_strength",
]

REGIME_FEATURES = list(STICKY_T_HMM_FEATURES)

SUPERVISED_REGIME_FEATURES = list(STICKY_T_HMM_FEATURES)

MARKET_STATE_COLUMNS = [
    "ret1",
    "log_ret",
    "ema10_dist",
    "ema20_dist",
    "ema50_dist",
    "rsi_14",
    "atr_pct",
    "vol_chg",
    "range_pct",
    "body_pct",
    "mom_5",
    "mom_10",
    "vol_10",
    "vol_20",
    "trend_strength",
]

REQUIRED_STATE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    *MARKET_STATE_COLUMNS,
]

# regime probabilities that are handed to the DDQN as part of its state
REGIME_PROBA_COLUMNS = [f"regime_p{k}" for k in range(State_n)]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = DEVICE.type == "cuda"
AMP_DTYPE = torch.float16





torch.set_num_threads(CPU_COUNT)
try:
    torch.set_num_interop_threads(max(1, min(4, CPU_COUNT)))
except RuntimeError:
    pass



def safe_div(a, b):
    return a / b if abs(b) > 1e-12 else 0.0


def max_drawdown(equity_curve: np.ndarray):
    if equity_curve.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peaks) / np.maximum(peaks, 1e-12)
    return float(drawdowns.min())


def sharpe_from_returns(returns: np.ndarray, annualization=252.0):
    if returns.size < 2:
        return 0.0
    std = returns.std()
    if std < 1e-12:
        return 0.0
    return float(returns.mean() / std * np.sqrt(annualization))


def load_pickle_any(path: str):
    obj = pd.read_pickle(path)
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if isinstance(obj, dict):
        for value in obj.values():
            if isinstance(value, pd.DataFrame):
                return value.copy()
    raise ValueError("No DataFrame found in pickle file.")


def ensure_datetime_index(df: pd.DataFrame):
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        for column in ("time", "date", "datetime"):
            if column in out.columns:
                out[column] = pd.to_datetime(out[column])
                out = out.set_index(column)
                break
        else:
            out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    if out.index.hasnans or out.index.has_duplicates:
        raise ValueError("Market timestamps must be valid and unique.")
    return out


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def safe_series_div(a: pd.Series, b: pd.Series):
    return (
        a.div(b.replace(0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )


def ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(series: pd.Series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    mean_up = up.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    mean_down = down.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    relative_strength = mean_up.div(mean_down.replace(0, np.nan))
    return 100.0 - 100.0 / (1.0 + relative_strength)


def atr(df: pd.DataFrame, period=14):
    previous_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def build_features_no_leak(df: pd.DataFrame):
    out = df.copy()
    out.columns = [column.lower() for column in out.columns]

    for column in ("open", "high", "low", "close", "volume"):
        if column not in out.columns:
            raise ValueError(f"Missing required column: {column}")

    close = out["close"]

    out["ret1"] = close.pct_change()
    out["log_ret"] = np.log(close.div(close.shift(1)))

    out["ema_10"] = ema(close, 10)
    out["ema_20"] = ema(close, 20)
    out["ema_50"] = ema(close, 50)

    out["ema10_dist"] = safe_series_div(close - out["ema_10"], out["ema_10"])
    out["ema20_dist"] = safe_series_div(close - out["ema_20"], out["ema_20"])
    out["ema50_dist"] = safe_series_div(close - out["ema_50"], out["ema_50"])

    out["rsi_14"] = rsi(close, 14)
    out["atr_14"] = atr(out, 14)
    out["atr_pct"] = safe_series_div(out["atr_14"], close)

    out["vol_chg"] = out["volume"].pct_change()
    out["range_pct"] = safe_series_div(out["high"] - out["low"], close)
    out["body_pct"] = safe_series_div((close - out["open"]).abs(), close)

    out["mom_5"] = close.pct_change(5)
    out["mom_10"] = close.pct_change(10)
    out["vol_10"] = out["ret1"].rolling(10, min_periods=10).std()
    out["vol_20"] = out["ret1"].rolling(20, min_periods=20).std()
    out["trend_strength"] = safe_series_div((out["ema_10"] - out["ema_50"]).abs(), close)

    return out.replace([np.inf, -np.inf], np.nan).ffill()


def prepare_train_test_features_no_leak(
    df: pd.DataFrame,
    train_period=None,
    test_period=None,
    lookback_bars=250,
):
    raw = ensure_datetime_index(df)
    train_period = TRAIN_PERIOD if train_period is None else train_period
    test_period = TEST_PERIOD if test_period is None else test_period

    train_start = pd.Timestamp(train_period[0])
    train_end = pd.Timestamp(train_period[1])
    test_start = pd.Timestamp(test_period[0])
    test_end = pd.Timestamp(test_period[1])
    if not train_start < train_end <= test_start < test_end:
        raise ValueError("Expected chronological, disjoint training and test periods.")
    if lookback_bars < 0:
        raise ValueError("Feature history length must not be negative.")

    train_raw = raw.loc[(raw.index >= train_start) & (raw.index < train_end)].copy()
    test_raw = raw.loc[(raw.index >= test_start) & (raw.index < test_end)].copy()

    if train_raw.empty:
        raise ValueError("Train split is empty.")
    if test_raw.empty:
        raise ValueError("Test split is empty.")

    train_features = build_features_no_leak(train_raw)

    history = raw.loc[raw.index < test_start].tail(lookback_bars)
    test_context = pd.concat([history, test_raw])
    test_context = test_context[~test_context.index.duplicated(keep="last")].sort_index()

    test_features = build_features_no_leak(test_context)
    test_features = test_features.loc[test_raw.index].copy()

    train_features = train_features.dropna(subset=REQUIRED_STATE_COLUMNS).copy()
    test_features = test_features.dropna(subset=REQUIRED_STATE_COLUMNS).copy()

    if train_features.empty:
        raise ValueError("Train features are empty after preprocessing.")
    if test_features.empty:
        raise ValueError("Test features are empty after preprocessing.")

    return train_features, test_features


# ============================================================
# REGIME MODELS
# ============================================================

def quantile_3class_labels_from_future_return(train_df: pd.DataFrame, horizon: int = 5):
    future_return = train_df["close"].shift(-horizon).div(train_df["close"]) - 1.0
    valid = future_return.dropna()

    if valid.empty:
        return pd.Series(index=train_df.index, dtype=np.float32)

    low = float(valid.quantile(1.0 / 3.0))
    high = float(valid.quantile(2.0 / 3.0))

    labels = pd.Series(index=train_df.index, dtype=np.int64)
    labels.loc[future_return <= low] = 0
    labels.loc[(future_return > low) & (future_return < high)] = 1
    labels.loc[future_return >= high] = 2
    return labels.dropna().astype(np.int64)


def build_label_map(raw_labels, aligned_df, n_components):
    statistics = []

    for cluster in range(n_components):
        subset = aligned_df.iloc[np.flatnonzero(raw_labels == cluster)]
        statistics.append(
            {
                "cluster": cluster,
                "ret_mean": float(subset["ret1"].mean()) if len(subset) else -np.inf,
                "vol_mean": float(subset["atr_pct"].mean()) if len(subset) else np.inf,
                "trend_mean": float(subset["trend_strength"].mean()) if len(subset) else -np.inf,
            }
        )

    ordered = pd.DataFrame(statistics).sort_values(["ret_mean", "trend_mean", "vol_mean"])

    return {
        int(cluster): min(rank, 2)
        for rank, cluster in enumerate(ordered["cluster"].tolist())
    }


def regime_proba_frame(regime_model, df, n_states=State_n):
    """
    Regime probabilities aligned with the mapped regime labels.

    This is the matrix/posterior information that is handed to the DDQN instead
    of a single hard regime id. Falls back to a one-hot encoding of the hard
    label when a model cannot produce soft probabilities.
    """
    columns = [f"regime_p{k}" for k in range(n_states)]
    neutral = min(1, n_states - 1)

    has_soft_proba = (
        hasattr(regime_model, "transform_proba_df")
        and type(regime_model).transform_proba_df is not BaseRegimeModel.transform_proba_df
    )

    if has_soft_proba:
        try:
            raw = regime_model.transform_proba_df(df)
            if raw is not None and not raw.empty:
                out = pd.DataFrame(0.0, index=df.index, columns=columns, dtype=np.float32)
                for column in columns:
                    if column in raw.columns:
                        out[column] = raw[column].to_numpy(dtype=np.float32)

                out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                row_sum = out.sum(axis=1)
                bad = row_sum <= 1e-12
                out = out.div(np.where(bad, 1.0, row_sum), axis=0)

                if bad.any():
                    out.loc[bad, :] = 0.0
                    out.loc[bad, columns[neutral]] = 1.0

                return out
        except Exception:
            pass

    labels = df["regime"].to_numpy() if "regime" in df.columns else regime_model.transform_df(df).to_numpy()
    out = pd.DataFrame(0.0, index=df.index, columns=columns, dtype=np.float32)

    for k in range(n_states):
        out[columns[k]] = (labels == k).astype(np.float32)

    return out


class BaseRegimeModel:
    def fit(self, train_df):
        return self

    def transform_df(self, df):
        return pd.Series(1, index=df.index, dtype=np.int64)

    def transform_proba_df(self, df):
        columns = [f"regime_p{k}" for k in range(State_n)]
        out = pd.DataFrame(0.0, index=df.index, columns=columns, dtype=np.float32)
        out[columns[min(1, State_n - 1)]] = 1.0
        return out


class NoRegimeModel(BaseRegimeModel):
    pass


class RuleBasedVolatilityRegime(BaseRegimeModel):
    def __init__(self, column="atr_pct", low_q=0.33, high_q=0.67):
        self.column = column
        self.low_q = low_q
        self.high_q = high_q

    def fit(self, train_df):
        values = train_df[self.column].dropna()
        self.low_threshold = float(values.quantile(self.low_q))
        self.high_threshold = float(values.quantile(self.high_q))
        return self

    def transform_df(self, df):
        values = df[self.column].ffill().fillna(0.0)
        result = pd.Series(1, index=df.index, dtype=np.int64)
        result.loc[values <= self.low_threshold] = 0
        result.loc[values >= self.high_threshold] = 2
        return result


class RuleBasedTrendRegime(BaseRegimeModel):
    def __init__(self, column="trend_strength", quantile=0.67):
        self.column = column
        self.quantile = quantile

    def fit(self, train_df):
        self.threshold = float(train_df[self.column].dropna().quantile(self.quantile))
        return self

    def transform_df(self, df):
        trend = df[self.column].ffill().fillna(0.0)
        direction = df["ema10_dist"].ffill().fillna(0.0)
        result = pd.Series(1, index=df.index, dtype=np.int64)
        result.loc[(trend >= self.threshold) & (direction <= 0)] = 0
        result.loc[(trend >= self.threshold) & (direction > 0)] = 2
        return result


class SklearnRegimeModel(BaseRegimeModel):
    def __init__(self, kind, n_components=State_n):
        self.kind = kind
        self.n_components = n_components
        self.scaler = StandardScaler()

    def fit(self, train_df):
        clean = train_df[REGIME_FEATURES].dropna()
        scaled = self.scaler.fit_transform(clean.to_numpy())

        if self.kind == REGIME_KMEANS:
            self.model = KMeans(n_clusters=self.n_components, random_state=SEED, n_init=3)
            raw_labels = self.model.fit_predict(scaled)
        elif self.kind == REGIME_GMM:
            self.model = GaussianMixture(
                n_components=self.n_components,
                covariance_type="diag",
                random_state=SEED,
            )
            raw_labels = self.model.fit_predict(scaled)
        elif self.kind == REGIME_HMM:
            if not HMM_AVAILABLE:
                raise ImportError("Install hmmlearn first.")
            self.model = GaussianHMM(
                n_components=self.n_components,
                covariance_type="diag",
                n_iter=200,
                random_state=SEED,
            )
            self.model.fit(scaled)
            raw_labels = self.model.predict(scaled)
        else:
            raise ValueError(self.kind)

        aligned = train_df.loc[clean.index]
        self.label_map = build_label_map(raw_labels, aligned, self.n_components)
        return self

    def _raw_proba(self, values):
        """
        Raw (unmapped) regime probabilities.

        For the HMM kind this uses causal forward filtering, so no future bar of
        the evaluated split influences the regime of the current bar.
        """
        if self.kind == REGIME_HMM:
            return hmm_filtered_proba(self.model, values)

        if self.kind == REGIME_GMM:
            return np.asarray(self.model.predict_proba(values), dtype=np.float64)

        return None

    def _remap_proba(self, raw_proba):
        raw_proba = np.asarray(raw_proba, dtype=np.float64)
        raw_proba = np.nan_to_num(raw_proba, nan=0.0, posinf=0.0, neginf=0.0)
        raw_proba = np.clip(raw_proba, 0.0, None)

        mapped = np.zeros((raw_proba.shape[0], self.n_components), dtype=np.float64)

        for raw_label in range(raw_proba.shape[1]):
            mapped_label = int(self.label_map.get(raw_label, min(1, self.n_components - 1)))
            mapped_label = min(max(mapped_label, 0), self.n_components - 1)
            mapped[:, mapped_label] += raw_proba[:, raw_label]

        row_sum = mapped.sum(axis=1, keepdims=True)
        return mapped / np.where(row_sum <= 1e-12, 1.0, row_sum)

    def _mapped_proba(self, values):
        raw_proba = self._raw_proba(values)

        if raw_proba is not None:
            return self._remap_proba(raw_proba)

        labels = self.model.predict(values)
        mapped = np.zeros((len(labels), self.n_components), dtype=np.float64)
        mapped[np.arange(len(labels)), [self.label_map[int(label)] for label in labels]] = 1.0
        return mapped

    def transform_df(self, df):
        features = df[REGIME_FEATURES]
        valid = features.notna().all(axis=1)
        result = pd.Series(1, index=df.index, dtype=np.int64)

        if not valid.any():
            return result

        values = self.scaler.transform(features.loc[valid].to_numpy())
        proba = self._mapped_proba(values)

        result.loc[valid] = np.argmax(proba, axis=1).astype(np.int64)

        return result.ffill().fillna(1).astype(np.int64)

    def transform_proba_df(self, df):
        features = df[REGIME_FEATURES]
        valid = features.notna().all(axis=1)
        columns = [f"regime_p{k}" for k in range(self.n_components)]
        neutral = min(1, self.n_components - 1)

        out = pd.DataFrame(0.0, index=df.index, columns=columns, dtype=np.float32)
        out[columns[neutral]] = 1.0

        if not valid.any():
            return out

        values = self.scaler.transform(features.loc[valid].to_numpy())
        out.loc[valid, :] = self._mapped_proba(values).astype(np.float32)

        out = out.ffill().fillna(0.0)
        row_sum = out.sum(axis=1)
        bad = row_sum <= 1e-12
        out = out.div(np.where(bad, 1.0, row_sum), axis=0)

        if bad.any():
            out.loc[bad, :] = 0.0
            out.loc[bad, columns[neutral]] = 1.0

        return out


import warnings
import numpy as np

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans





class StickyStudentTHMMRegimeModel(BaseRegimeModel):
    """
    Sticky Student-t HMM regime model for your current pipeline.

    Recommended:
        n_states = 3
        df       = 4.0
        sticky   = 8.0
        n_iter   = 30

    Pipeline API:
        fit(train_df)
        transform_df(df)
        transform_proba_df(df)
    """

    def __init__(
        self,
        n_states=3,
        n_iter=30,
        df=4.0,
        sticky=8.0,
        device="cpu",
        random_state=42,
        eps=1e-6,
        verbose=False,
        feature_cols=None,
        map_labels=True,
        **kwargs,
    ):
        self.n_states = int(n_states)
        self.n_components = int(n_states)
        self.n_iter = int(n_iter)
        self.df = float(df)
        self.sticky = float(sticky)
        self.random_state = int(random_state)
        self.eps = float(eps)
        self.verbose = verbose
        self.map_labels = map_labels

        if feature_cols is None:
            self.feature_cols = list(STICKY_T_HMM_FEATURES)
        else:
            self.feature_cols = list(feature_cols)

        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.scaler = RobustScaler()

        self.pi = None
        self.A = None
        self.loc = None
        self.scale = None
        self.fitted = False
        self.feature_cols_ = None
        self.label_map = None

    # ============================================================
    # Utils
    # ============================================================

    def _clean_np(self, X):
        X = np.asarray(X, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
        X = np.clip(X, -10.0, 10.0)
        return X.astype(np.float32)

    def _to_tensor(self, X):
        X = self._clean_np(X)
        return torch.tensor(X, dtype=torch.float32, device=self.device)

    def _prepare_df_features(self, df: pd.DataFrame):
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing StickyStudentTHMM features: {missing}")

        raw = df[self.feature_cols].copy()
        valid = raw.notna().all(axis=1)

        clean = raw.loc[valid].replace([np.inf, -np.inf], np.nan).dropna()

        return clean

    # ============================================================
    # Student-t emission
    # ============================================================

    def _student_t_log_prob(self, X):
        """
        X: [T, D]
        return: [T, K]
        """
        nu = torch.tensor(self.df, device=self.device, dtype=torch.float32)

        X_ = X[:, None, :]
        loc = self.loc[None, :, :]
        scale = self.scale[None, :, :].clamp_min(self.eps)

        z = (X_ - loc) / scale

        log_norm = (
            torch.lgamma((nu + 1.0) / 2.0)
            - torch.lgamma(nu / 2.0)
            - 0.5 * torch.log(nu * torch.pi)
            - torch.log(scale)
        )

        log_kernel = -((nu + 1.0) / 2.0) * torch.log1p((z ** 2) / nu)

        return (log_norm + log_kernel).sum(dim=-1)

    # ============================================================
    # Forward backward
    # ============================================================

    def _forward_backward(self, logB):
        T, K = logB.shape

        log_pi = torch.log(self.pi.clamp_min(self.eps))
        log_A = torch.log(self.A.clamp_min(self.eps))

        alpha = torch.empty((T, K), device=self.device, dtype=torch.float32)
        beta = torch.empty((T, K), device=self.device, dtype=torch.float32)

        alpha[0] = log_pi + logB[0]

        for t in range(1, T):
            alpha[t] = logB[t] + torch.logsumexp(
                alpha[t - 1][:, None] + log_A,
                dim=0,
            )

        beta[-1] = 0.0

        for t in range(T - 2, -1, -1):
            beta[t] = torch.logsumexp(
                log_A + logB[t + 1][None, :] + beta[t + 1][None, :],
                dim=1,
            )

        loglik = torch.logsumexp(alpha[-1], dim=0)

        log_gamma = alpha + beta - loglik
        gamma = torch.exp(log_gamma).clamp_min(self.eps)
        gamma = gamma / gamma.sum(dim=1, keepdim=True).clamp_min(self.eps)

        xi_sum = torch.zeros((K, K), device=self.device, dtype=torch.float32)

        for t in range(T - 1):
            log_xi_t = (
                alpha[t][:, None]
                + log_A
                + logB[t + 1][None, :]
                + beta[t + 1][None, :]
                - loglik
            )
            xi_sum += torch.exp(log_xi_t)

        xi_sum = xi_sum.clamp_min(self.eps)

        return gamma, xi_sum, loglik

    # ============================================================
    # Causal filtering (no look-ahead)
    # ============================================================

    @torch.no_grad()
    def _forward_filter(self, logB):
        """
        Forward-only filtered state probabilities p(state_t | x_1 ... x_t).

        _forward_backward() also runs the beta (backward) pass, which mixes
        future observations into the state belief of the current bar. The
        filtering pass below is strictly causal and is what the test split uses.
        """
        T, K = logB.shape

        log_pi = torch.log(self.pi.clamp_min(self.eps))
        log_A = torch.log(self.A.clamp_min(self.eps))
        uniform = torch.full((K,), -float(np.log(K)), device=self.device, dtype=torch.float32)

        proba = torch.empty((T, K), device=self.device, dtype=torch.float32)

        log_alpha = log_pi + logB[0]
        normalizer = torch.logsumexp(log_alpha, dim=0)
        log_alpha = log_alpha - normalizer if torch.isfinite(normalizer) else uniform
        proba[0] = torch.exp(log_alpha)

        for t in range(1, T):
            log_alpha = logB[t] + torch.logsumexp(log_alpha[:, None] + log_A, dim=0)
            normalizer = torch.logsumexp(log_alpha, dim=0)

            if torch.isfinite(normalizer):
                log_alpha = log_alpha - normalizer
            else:
                fallback = torch.logsumexp(logB[t], dim=0)
                log_alpha = logB[t] - fallback if torch.isfinite(fallback) else uniform

            proba[t] = torch.exp(log_alpha)

        proba = torch.nan_to_num(proba, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        row_sum = proba.sum(dim=1, keepdim=True)
        bad = row_sum.squeeze(-1) <= self.eps
        row_sum = torch.where(row_sum <= self.eps, torch.ones_like(row_sum), row_sum)
        proba = proba / row_sum

        if bool(bad.any()):
            proba[bad, :] = 0.0
            proba[bad, min(1, K - 1)] = 1.0

        return proba

    # ============================================================
    # Init
    # ============================================================

    def _initialize_params(self, X_np):
        rng = np.random.RandomState(self.random_state)

        T, D = X_np.shape
        K = self.n_states

        if T < K * 5:
            raise ValueError("Not enough rows for Sticky Student-t HMM.")

        km = KMeans(
            n_clusters=K,
            random_state=self.random_state,
            n_init=10,
        )
        labels = km.fit_predict(X_np)
        centers = km.cluster_centers_.astype(np.float32)

        loc = np.zeros((K, D), dtype=np.float32)
        scale = np.ones((K, D), dtype=np.float32)

        global_std = X_np.std(axis=0).astype(np.float32) + 1e-3

        for k in range(K):
            mask = labels == k
            if mask.sum() > 5:
                loc[k] = np.median(X_np[mask], axis=0).astype(np.float32)
                scale[k] = X_np[mask].std(axis=0).astype(np.float32) + 1e-3
            else:
                loc[k] = centers[k]
                scale[k] = global_std

        pi = np.ones(K, dtype=np.float32) / K

        A = np.ones((K, K), dtype=np.float32)
        A += np.eye(K, dtype=np.float32) * self.sticky
        A = A / A.sum(axis=1, keepdims=True)

        self.pi = torch.tensor(pi, device=self.device, dtype=torch.float32)
        self.A = torch.tensor(A, device=self.device, dtype=torch.float32)
        self.loc = torch.tensor(loc, device=self.device, dtype=torch.float32)
        self.scale = torch.tensor(scale, device=self.device, dtype=torch.float32).clamp_min(1e-3)

    # ============================================================
    # Core array fit/predict
    # ============================================================

    def fit_array(self, X):
        X = self._clean_np(X)

        X_scaled = self.scaler.fit_transform(X).astype(np.float32)
        X_scaled = self._clean_np(X_scaled)

        self._initialize_params(X_scaled)

        Xt = self._to_tensor(X_scaled)
        T, D = Xt.shape
        K = self.n_states

        prev_loglik = None

        for it in range(self.n_iter):
            logB = self._student_t_log_prob(Xt)
            gamma, xi_sum, loglik = self._forward_backward(logB)

            self.pi = gamma[0]
            self.pi = self.pi / self.pi.sum().clamp_min(self.eps)

            A = xi_sum / xi_sum.sum(dim=1, keepdim=True).clamp_min(self.eps)

            sticky_eye = torch.eye(K, device=self.device, dtype=torch.float32) * self.sticky
            A = A + sticky_eye
            A = A / A.sum(dim=1, keepdim=True).clamp_min(self.eps)
            self.A = A.clamp_min(self.eps)

            nu = torch.tensor(self.df, device=self.device, dtype=torch.float32)

            diff = Xt[:, None, :] - self.loc[None, :, :]
            var = self.scale[None, :, :].pow(2).clamp_min(self.eps)

            maha = (diff.pow(2) / var).sum(dim=-1)
            t_weight = (nu + D) / (nu + maha)

            resp = gamma * t_weight
            resp_sum = resp.sum(dim=0).clamp_min(self.eps)

            loc_new = (
                resp[:, :, None] * Xt[:, None, :]
            ).sum(dim=0) / resp_sum[:, None]

            diff_new = Xt[:, None, :] - loc_new[None, :, :]

            var_new = (
                resp[:, :, None] * diff_new.pow(2)
            ).sum(dim=0) / gamma.sum(dim=0).clamp_min(self.eps)[:, None]

            self.loc = loc_new
            self.scale = torch.sqrt(var_new.clamp_min(self.eps)).clamp_min(1e-3)

            if self.verbose:
                pass

            if prev_loglik is not None:
                if torch.abs(loglik - prev_loglik) < 1e-4:
                    break

            prev_loglik = loglik.detach()

        self.fitted = True
        return self

    @torch.no_grad()
    def predict_proba_array(self, X, causal=True):
        if not causal:
            raise ValueError("Non-causal smoothing is disabled for regime inference.")
        if not self.fitted:
            raise RuntimeError("StickyStudentTHMMRegimeModel is not fitted.")

        X = self._clean_np(X)
        X_scaled = self.scaler.transform(X).astype(np.float32)
        X_scaled = self._clean_np(X_scaled)

        Xt = self._to_tensor(X_scaled)
        logB = self._student_t_log_prob(Xt)

        gamma = self._forward_filter(logB)

        return gamma.detach().cpu().numpy()

    @torch.no_grad()
    def predict_array(self, X, causal=True):
        probs = self.predict_proba_array(X, causal=causal)
        return np.argmax(probs, axis=1).astype(np.int64)

    def _remap_proba(self, raw_proba):
        raw_proba = np.asarray(raw_proba, dtype=np.float64)
        raw_proba = np.nan_to_num(raw_proba, nan=0.0, posinf=0.0, neginf=0.0)
        raw_proba = np.clip(raw_proba, 0.0, None)

        mapped = np.zeros((raw_proba.shape[0], self.n_states), dtype=np.float64)
        label_map = self.label_map if self.label_map is not None else {k: k for k in range(self.n_states)}

        for raw_label in range(raw_proba.shape[1]):
            mapped_label = int(label_map.get(raw_label, min(1, self.n_states - 1)))
            mapped_label = min(max(mapped_label, 0), self.n_states - 1)
            mapped[:, mapped_label] += raw_proba[:, raw_label]

        row_sum = mapped.sum(axis=1, keepdims=True)
        return mapped / np.where(row_sum <= 1e-12, 1.0, row_sum)

    # ============================================================
    # Pipeline API
    # ============================================================

    def fit(self, train_df):
        self.feature_cols_ = list(self.feature_cols)

        clean = self._prepare_df_features(train_df)

        if len(clean) < self.n_states * 10:
            raise ValueError("Not enough clean rows for Sticky Student-t HMM.")

        X = clean.to_numpy(dtype=np.float32)
        self.fit_array(X)

        raw_states = self.predict_array(X)

        if self.map_labels:
            self.label_map = build_label_map(
                raw_states,
                train_df.loc[clean.index],
                self.n_states,
            )
        else:
            self.label_map = {k: k for k in range(self.n_states)}

        return self

    def transform_df(self, df):
        if not self.fitted:
            raise RuntimeError("Fit the Student-t regime model on past training data first.")

        clean = self._prepare_df_features(df)
        result = pd.Series(1, index=df.index, dtype=np.int64)

        if clean.empty:
            return result

        X = clean.to_numpy(dtype=np.float32)
        proba = self._remap_proba(self.predict_proba_array(X, causal=True))

        result.loc[clean.index] = np.argmax(proba, axis=1).astype(np.int64)

        return result.ffill().fillna(1).astype(np.int64)

    def transform_proba_df(self, df):
        if not self.fitted:
            raise RuntimeError("Fit the Student-t regime model on past training data first.")

        clean = self._prepare_df_features(df)
        result = pd.DataFrame(
            0.0,
            index=df.index,
            columns=[f"regime_p{k}" for k in range(self.n_states)],
            dtype=np.float32,
        )

        if clean.empty:
            result["regime_p1"] = 1.0
            return result

        X = clean.to_numpy(dtype=np.float32)
        probs = self._remap_proba(self.predict_proba_array(X, causal=True))

        result.loc[clean.index, :] = probs.astype(np.float32)
        result = result.ffill().fillna(0.0)

        row_sum = result.sum(axis=1).replace(0, np.nan)
        result = result.div(row_sum, axis=0).fillna(0.0)

        return result

    def add_regime_columns(self, df, prefix="regime"):
        out = df.copy()
        probs = self.transform_proba_df(out)

        for k in range(self.n_states):
            out[f"{prefix}_p{k}"] = probs[f"regime_p{k}"].values

        out[prefix] = self.transform_df(out).values

        return out




import logging
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


# ============================================================
# Robust HMM validation / repair helpers
# ============================================================

def _valid_hmm_model(model, eps=1e-8):
    """
    Strict validation for hmmlearn HMM models.
    Rejects models with invalid startprob_, transmat_, or lambdas_.
    """
    attrs = ["startprob_", "transmat_"]

    for attr in attrs:
        if not hasattr(model, attr):
            return False

        arr = getattr(model, attr)

        if arr is None:
            return False

        arr = np.asarray(arr)

        if not np.all(np.isfinite(arr)):
            return False

        if np.any(arr < 0):
            return False

    start_sum = np.sum(model.startprob_)
    if not np.isfinite(start_sum) or start_sum <= eps:
        return False

    row_sums = np.sum(model.transmat_, axis=1)

    if not np.all(np.isfinite(row_sums)):
        return False

    if np.any(row_sums <= eps):
        return False

    # PoissonHMM-specific parameter in hmmlearn
    if hasattr(model, "lambdas_"):
        lambdas = np.asarray(model.lambdas_)

        if not np.all(np.isfinite(lambdas)):
            return False

        if np.any(lambdas <= 0):
            return False

    return True


def repair_hmm_transmat(model, eps=1e-6):
    """
    Repairs degenerate HMM transition/start probabilities.

    If a row of transmat_ has zero sum, replace it with uniform probability.
    Also normalizes startprob_.
    """
    trans = np.asarray(model.transmat_, dtype=np.float64)
    n = trans.shape[0]

    trans = np.nan_to_num(trans, nan=0.0, posinf=0.0, neginf=0.0)
    trans = np.clip(trans, 0.0, None)

    row_sums = trans.sum(axis=1)

    for i in range(n):
        if row_sums[i] <= eps:
            trans[i, :] = 1.0 / n
        else:
            trans[i, :] = trans[i, :] / row_sums[i]

    model.transmat_ = trans

    start = np.asarray(model.startprob_, dtype=np.float64)
    start = np.nan_to_num(start, nan=0.0, posinf=0.0, neginf=0.0)
    start = np.clip(start, 0.0, None)

    s = start.sum()
    if s <= eps:
        start[:] = 1.0 / len(start)
    else:
        start /= s

    model.startprob_ = start

    return model


# ============================================================
# Causal (forward-only) HMM inference
# ============================================================

def _logsumexp(values):
    values = np.asarray(values, dtype=np.float64)
    peak = float(np.max(values))
    if not np.isfinite(peak):
        return -np.inf
    return float(peak + np.log(np.sum(np.exp(values - peak))))


def hmm_filtered_proba(model, x, eps=1e-12):
    """
    Causal (forward-only) filtered state probabilities for hmmlearn HMM models.

    model.predict() / model.predict_proba() run Viterbi / forward-backward over the
    whole sequence. On the test split that means the regime of bar t is inferred
    with information from bars t + 1 ... T, i.e. look-ahead leakage.

    This helper propagates information forward in time only:

        p(state_t | x_1 ... x_t)

    so the regime signal stays strictly causal and can be used inside the test
    backtest without the model "seeing the future".
    """
    x = np.asarray(x, dtype=np.float64)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    n_states = int(model.n_components)
    proba = np.full((len(x), n_states), 1.0 / n_states, dtype=np.float64)

    if len(x) == 0:
        return proba

    trans = np.nan_to_num(np.asarray(model.transmat_, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    trans = np.clip(trans, 0.0, None)
    row_sums = trans.sum(axis=1, keepdims=True)
    trans = np.where(row_sums <= eps, 1.0 / n_states, trans / np.maximum(row_sums, eps))
    log_trans = np.log(np.clip(trans, eps, None))

    start = np.nan_to_num(np.asarray(model.startprob_, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    start = np.clip(start, 0.0, None)
    if start.sum() <= eps:
        start = np.full(n_states, 1.0 / n_states, dtype=np.float64)
    else:
        start = start / start.sum()
    log_start = np.log(np.clip(start, eps, None))

    if hasattr(model, "_compute_log_likelihood"):
        log_emit = np.asarray(model._compute_log_likelihood(x), dtype=np.float64)
        log_emit = np.nan_to_num(log_emit, nan=-np.inf, posinf=0.0, neginf=-np.inf)
    else:
        # Fallback: per-row (single observation) decode, still strictly causal.
        log_emit = np.full((len(x), n_states), -np.inf, dtype=np.float64)
        for row in range(len(x)):
            log_emit[row, int(model.predict(x[row : row + 1])[0])] = 0.0

    uniform = -np.log(n_states)

    log_alpha = log_start + log_emit[0]
    normalizer = _logsumexp(log_alpha)
    log_alpha = log_alpha - normalizer if np.isfinite(normalizer) else np.full(n_states, uniform)
    proba[0] = np.exp(log_alpha)

    for row in range(1, len(x)):
        predicted = log_trans + log_alpha[:, None]
        peak = np.max(predicted, axis=0)
        safe_peak = np.where(np.isfinite(peak), peak, 0.0)
        summed = np.sum(np.exp(predicted - safe_peak), axis=0)
        predicted = np.where(np.isfinite(peak), safe_peak + np.log(np.maximum(summed, eps)), -np.inf)

        log_alpha = predicted + log_emit[row]
        normalizer = _logsumexp(log_alpha)

        if np.isfinite(normalizer):
            log_alpha = log_alpha - normalizer
        else:
            normalizer = _logsumexp(predicted)
            log_alpha = predicted - normalizer if np.isfinite(normalizer) else np.full(n_states, uniform)

        proba[row] = np.exp(np.clip(log_alpha, -60.0, 0.0))

    row_sums = proba.sum(axis=1, keepdims=True)
    proba = proba / np.where(row_sums <= eps, 1.0, row_sums)

    return proba


# ============================================================
# Full corrected PoissonHMM regime model
# ============================================================

class PoissonHMMRegimeModel(BaseRegimeModel):
    """
    Robust hmmlearn PoissonHMM regime detector using the shared context features:

        ret1, atr_pct, vol_20, mom_10, range_pct, trend_strength

    Notes:
    - PoissonHMM expects non-negative count-like inputs.
    - Market features are continuous, so we robust-scale them,
      shift them to positive space, multiply by scale_factor,
      then round to integer counts.
    - PoissonHMM can easily create dead states on market data,
      so this class suppresses hmmlearn spam logs, repairs transition
      matrices, and validates each restart.
    """

    def __init__(
        self,
        n_components=3,
        scale_factor=10.0,
        n_restarts=20,
        n_iter=100,
        tol=1e-3,
        feature_cols=None,
        eps=1e-6,
    ):
        self.n_components = int(n_components)
        self.scale_factor = float(scale_factor)
        self.n_restarts = int(n_restarts)
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.feature_cols = list(feature_cols) if feature_cols is not None else list(STICKY_T_HMM_FEATURES)
        self.eps = float(eps)

        self.scaler = RobustScaler()
        self.shift_ = None
        self.model = None
        self.label_map = None

    def _clean_features(self, df):
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing PoissonHMM features: {missing}")

        clean = df[self.feature_cols].copy()
        clean = clean.replace([np.inf, -np.inf], np.nan).dropna()
        return clean

    def _to_poisson_train(self, clean_df):
        """
        Fit scaler and convert continuous features to count-like inputs.
        """
        x = clean_df.to_numpy(dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        x = np.clip(x, -10.0, 10.0)

        x_scaled = self.scaler.fit_transform(x).astype(np.float32)
        x_scaled = np.nan_to_num(x_scaled, nan=0.0, posinf=10.0, neginf=-10.0)
        x_scaled = np.clip(x_scaled, -10.0, 10.0)

        self.shift_ = (-x_scaled.min(axis=0) + 1.0).astype(np.float32)

        x_pos = x_scaled + self.shift_[None, :]
        x_pos = np.clip(x_pos, self.eps, None)

        x_count = np.rint(x_pos * self.scale_factor).astype(np.int64)
        x_count = np.clip(x_count, 0, None)

        return x_count

    def _to_poisson_transform(self, clean_df):
        """
        Transform new data using already-fitted scaler and shift.
        """
        if self.shift_ is None:
            raise RuntimeError("PoissonHMMRegimeModel is not fitted yet: shift_ is None.")

        x = clean_df.to_numpy(dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        x = np.clip(x, -10.0, 10.0)

        x_scaled = self.scaler.transform(x).astype(np.float32)
        x_scaled = np.nan_to_num(x_scaled, nan=0.0, posinf=10.0, neginf=-10.0)
        x_scaled = np.clip(x_scaled, -10.0, 10.0)

        x_pos = x_scaled + self.shift_[None, :]
        x_pos = np.clip(x_pos, self.eps, None)

        x_count = np.rint(x_pos * self.scale_factor).astype(np.int64)
        x_count = np.clip(x_count, 0, None)

        return x_count

    def fit(self, train_df):
        if not HMM_AVAILABLE or PoissonHMM is None:
            raise ImportError("Installed hmmlearn does not provide PoissonHMM.")

        clean = self._clean_features(train_df)

        if len(clean) < self.n_components * 10:
            raise ValueError(
                f"Not enough clean rows for PoissonHMM. "
                f"clean_rows={len(clean)}, required={self.n_components * 10}"
            )

        x = self._to_poisson_train(clean)

        best_model = None
        best_score = -np.inf
        failed = 0

        # hmmlearn often logs transmat_ zero-sum messages through logging,
        # not only warnings. So suppress its logger during restarts.
        hmm_logger = logging.getLogger("hmmlearn")
        old_level = hmm_logger.level
        hmm_logger.setLevel(logging.ERROR)

        try:
            for run in range(self.n_restarts):
                try:
                    model = PoissonHMM(
                        n_components=self.n_components,
                        n_iter=self.n_iter,
                        tol=self.tol,
                        random_state=SEED + run,
                        verbose=False,
                        init_params="stl",
                        params="stl",
                    )

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(x)

                    model = repair_hmm_transmat(model)

                    if not _valid_hmm_model(model):
                        failed += 1
                        continue

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        score = float(model.score(x))

                    if not np.isfinite(score):
                        failed += 1
                        continue

                    if score > best_score:
                        best_score = score
                        best_model = model

                except Exception as e:
                    failed += 1
                    pass
                    continue

        finally:
            hmm_logger.setLevel(old_level)

        if best_model is None:
            raise RuntimeError(
                f"PoissonHMM failed in all {self.n_restarts} restarts. "
                f"Try lower scale_factor, lower n_components, or use Student-t HMM."
            )

        self.model = best_model

        labels = self.model.predict(x)

        self.label_map = build_label_map(
            labels,
            train_df.loc[clean.index],
            self.n_components,
        )

        return self

    def fit_df(self, train_df):
        """
        API compatibility with other regime models.
        """
        return self.fit(train_df)

    def transform_df(self, df):
        if self.model is None:
            raise RuntimeError("PoissonHMMRegimeModel must be fitted before transform_df().")

        clean = self._clean_features(df)
        result = pd.Series(1, index=df.index, dtype=np.int64)

        if clean.empty:
            return result

        x = self._to_poisson_transform(clean)

        try:
            proba = self._remap_proba(hmm_filtered_proba(self.model, x))
        except Exception:
            # If hmmlearn fails on prediction for any numerical reason,
            # return neutral regime instead of crashing the whole RL pipeline.
            return result.ffill().fillna(1).astype(np.int64)

        mapped = np.argmax(proba, axis=1).astype(np.int64)

        result.loc[clean.index] = mapped
        return result.ffill().fillna(1).astype(np.int64)

    def _remap_proba(self, raw_proba):
        raw_proba = np.asarray(raw_proba, dtype=np.float64)
        raw_proba = np.nan_to_num(raw_proba, nan=0.0, posinf=0.0, neginf=0.0)
        raw_proba = np.clip(raw_proba, 0.0, None)

        mapped = np.zeros((raw_proba.shape[0], self.n_components), dtype=np.float64)

        for raw_label in range(raw_proba.shape[1]):
            mapped_label = int(self.label_map.get(raw_label, 1)) if self.label_map else raw_label
            mapped_label = min(max(mapped_label, 0), self.n_components - 1)
            mapped[:, mapped_label] += raw_proba[:, raw_label]

        row_sum = mapped.sum(axis=1, keepdims=True)
        return mapped / np.where(row_sum <= self.eps, 1.0, row_sum)

    def transform_proba_df(self, df):
        """
        Causal (forward-only) regime probabilities aligned with the mapped labels.

        These probabilities are the transition-matrix based belief state that is
        handed to the DDQN, instead of a single hard regime id.
        """
        if self.model is None:
            raise RuntimeError("PoissonHMMRegimeModel must be fitted before transform_proba_df().")

        clean = self._clean_features(df)

        columns = [f"regime_p{k}" for k in range(self.n_components)]
        neutral_col = min(1, self.n_components - 1)

        proba = pd.DataFrame(0.0, index=df.index, columns=columns, dtype=np.float32)
        proba[columns[neutral_col]] = 1.0

        if clean.empty:
            return proba

        x = self._to_poisson_transform(clean)

        try:
            post = self._remap_proba(hmm_filtered_proba(self.model, x)).astype(np.float32)
        except Exception:
            return proba

        bad = (~np.isfinite(post).all(axis=1)) | (post.sum(axis=1) <= self.eps)
        if bad.any():
            post = np.where(np.isfinite(post), post, 0.0)
            post[bad, :] = 0.0
            post[bad, neutral_col] = 1.0

        proba.loc[clean.index, :] = post
        proba = proba.ffill().fillna(0.0)

        row_sum = proba.sum(axis=1)
        bad_rows = row_sum <= self.eps
        proba = proba.div(np.where(bad_rows, 1.0, row_sum), axis=0)

        if bad_rows.any():
            proba.loc[bad_rows, :] = 0.0
            proba.loc[bad_rows, columns[neutral_col]] = 1.0

        return proba

class XGBoostRegimeModel(BaseRegimeModel):
    def __init__(self, horizon=5, feature_cols=None):
        self.horizon = horizon
        self.feature_cols = list(feature_cols) if feature_cols is not None else list(SUPERVISED_REGIME_FEATURES)
        self.scaler = StandardScaler()

    def fit(self, train_df):
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is not installed.")

        labels = quantile_3class_labels_from_future_return(train_df, horizon=self.horizon)

        features = train_df.loc[labels.index, self.feature_cols].copy()

        valid = features.notna().all(axis=1)
        features = features.loc[valid]
        labels = labels.loc[features.index]

        x = self.scaler.fit_transform(features.to_numpy(dtype=np.float32))
        y = labels.to_numpy(dtype=np.int64)

        self.model = XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softmax",
            num_class=3,
            random_state=SEED,
            n_jobs=CPU_COUNT,
            eval_metric="mlogloss",
        )

        self.model.fit(x, y)
        return self

    def transform_df(self, df):
        features = df[self.feature_cols].copy()
        valid = features.notna().all(axis=1)
        result = pd.Series(1, index=df.index, dtype=np.int64)

        if not valid.any():
            return result

        x = self.scaler.transform(features.loc[valid].to_numpy(dtype=np.float32))
        pred = self.model.predict(x).astype(np.int64)

        result.loc[valid] = pred
        return result.ffill().fillna(1).astype(np.int64)

    def transform_proba_df(self, df):
        features = df[self.feature_cols].copy()
        valid = features.notna().all(axis=1)
        columns = [f"regime_p{k}" for k in range(3)]

        out = pd.DataFrame(0.0, index=df.index, columns=columns, dtype=np.float32)
        out[columns[1]] = 1.0

        if not valid.any():
            return out

        x = self.scaler.transform(features.loc[valid].to_numpy(dtype=np.float32))
        proba = np.asarray(self.model.predict_proba(x), dtype=np.float32)
        proba = np.nan_to_num(proba, nan=0.0, posinf=0.0, neginf=0.0)

        out.loc[valid, :] = proba
        out = out.ffill().fillna(0.0)

        row_sum = out.sum(axis=1)
        bad = row_sum <= 1e-12
        out = out.div(np.where(bad, 1.0, row_sum), axis=0)

        if bad.any():
            out.loc[bad, :] = 0.0
            out.loc[bad, columns[1]] = 1.0

        return out


class SequenceDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class SequenceRegimeNet(nn.Module):
    def __init__(self, model_type: str, input_dim: int, hidden_dim: int = 64, n_classes: int = 3):
        super().__init__()
        self.model_type = model_type

        if model_type == REGIME_RNN:
            self.seq = nn.RNN(input_dim, hidden_dim, batch_first=True)
            self.head = nn.Linear(hidden_dim, n_classes)
        elif model_type == REGIME_LSTM:
            self.seq = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.head = nn.Linear(hidden_dim, n_classes)
        elif model_type == REGIME_CNN:
            self.conv = nn.Sequential(
                nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Linear(64, n_classes)
        else:
            raise ValueError(model_type)

    def forward(self, x):
        if self.model_type in (REGIME_RNN, REGIME_LSTM):
            out, _ = self.seq(x)
            last = out[:, -1, :]
            return self.head(last)

        x = x.transpose(1, 2)
        feat = self.conv(x).squeeze(-1)
        return self.head(feat)


class NeuralSequenceRegimeModel(BaseRegimeModel):
    def __init__(
        self,
        model_type: str,
        seq_len: int = 32,
        horizon: int = 5,
        epochs: int = 20,
        batch_size: int = 256,
        feature_cols=None,
    ):
        self.model_type = model_type
        self.seq_len = seq_len
        self.horizon = horizon
        self.epochs = epochs
        self.batch_size = batch_size
        self.feature_cols = list(feature_cols) if feature_cols is not None else list(SUPERVISED_REGIME_FEATURES)
        self.scaler = StandardScaler()

    def _make_sequences(self, df: pd.DataFrame, labels: Optional[pd.Series] = None):
        feats = df[self.feature_cols].copy()
        valid_rows = feats.notna().all(axis=1)
        feats = feats.loc[valid_rows].copy()

        if feats.empty:
            return None, None, None

        feats_scaled = self.scaler.transform(feats.to_numpy(dtype=np.float32))

        xs = []
        ys = []
        idxs = []

        index_list = feats.index.tolist()
        feat_values = feats_scaled

        label_map = None
        if labels is not None:
            label_map = labels.to_dict()

        for i in range(self.seq_len - 1, len(feat_values)):
            current_idx = index_list[i]
            window = feat_values[i - self.seq_len + 1 : i + 1]

            if labels is not None:
                if current_idx not in label_map:
                    continue
                ys.append(int(label_map[current_idx]))

            xs.append(window)
            idxs.append(current_idx)

        if not xs:
            return None, None, None

        x = np.asarray(xs, dtype=np.float32)
        y = None if labels is None else np.asarray(ys, dtype=np.int64)
        idxs = pd.Index(idxs)

        return x, y, idxs

    def fit(self, train_df):
        labels = quantile_3class_labels_from_future_return(train_df, horizon=self.horizon)

        features = train_df.loc[:, self.feature_cols].copy()
        valid = features.notna().all(axis=1)
        features = features.loc[valid]

        if features.empty:
            raise ValueError(f"No valid features for {self.model_type}")

        self.scaler.fit(features.to_numpy(dtype=np.float32))

        x, y, _ = self._make_sequences(train_df, labels=labels)

        if x is None or len(x) == 0:
            raise ValueError(f"No valid sequences for {self.model_type}")

        dataset = SequenceDataset(x, y)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=USE_CUDA,
        )

        self.model = SequenceRegimeNet(
            model_type=self.model_type,
            input_dim=len(self.feature_cols),
            hidden_dim=64,
            n_classes=3,
        ).to(DEVICE)

        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        self.model.train()

        for epoch in range(self.epochs):
            epoch_loss = 0.0

            for batch_x, batch_y in loader:
                batch_x = batch_x.to(DEVICE, non_blocking=True)
                batch_y = batch_y.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)

                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item()) * len(batch_x)

            avg_loss = epoch_loss / max(len(dataset), 1)
            # print(f"{self.model_type} epoch {epoch + 1}/{self.epochs} loss={avg_loss:.6f}")

        self.model.eval()
        return self

    def transform_df(self, df):
        result = pd.Series(1, index=df.index, dtype=np.int64)

        x, _, idxs = self._make_sequences(df, labels=None)

        if x is None or len(x) == 0:
            return result

        self.model.eval()
        preds = []

        with torch.no_grad():
            for start in range(0, len(x), self.batch_size):
                batch = torch.tensor(
                    x[start : start + self.batch_size],
                    dtype=torch.float32,
                    device=DEVICE,
                )

                logits = self.model(batch)
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                preds.append(pred)

        preds = np.concatenate(preds).astype(np.int64)

        result.loc[idxs] = preds
        return result.ffill().fillna(1).astype(np.int64)

    def transform_proba_df(self, df):
        columns = [f"regime_p{k}" for k in range(3)]

        out = pd.DataFrame(0.0, index=df.index, columns=columns, dtype=np.float32)
        out[columns[1]] = 1.0

        x, _, idxs = self._make_sequences(df, labels=None)

        if x is None or len(x) == 0:
            return out

        self.model.eval()
        chunks = []

        with torch.no_grad():
            for start in range(0, len(x), self.batch_size):
                batch = torch.tensor(
                    x[start : start + self.batch_size],
                    dtype=torch.float32,
                    device=DEVICE,
                )

                logits = self.model(batch)
                chunks.append(torch.softmax(logits, dim=1).cpu().numpy())

        proba = np.concatenate(chunks, axis=0).astype(np.float32)
        proba = np.nan_to_num(proba, nan=0.0, posinf=0.0, neginf=0.0)

        out.loc[idxs, :] = proba
        out = out.ffill().fillna(0.0)

        row_sum = out.sum(axis=1)
        bad = row_sum <= 1e-12
        out = out.div(np.where(bad, 1.0, row_sum), axis=0)

        if bad.any():
            out.loc[bad, :] = 0.0
            out.loc[bad, columns[1]] = 1.0

        return out


def build_regime_model(kind):
    if kind == REGIME_NONE:
        return NoRegimeModel()

    if kind == REGIME_RULE_VOL:
        return RuleBasedVolatilityRegime()

    if kind == REGIME_RULE_TREND:
        return RuleBasedTrendRegime()

    if kind in (REGIME_KMEANS, REGIME_GMM, REGIME_HMM):
        return SklearnRegimeModel(
            kind,
            n_components=State_n,
        )

    if kind == REGIME_POISSON_HMM:
        return PoissonHMMRegimeModel(
            n_components=3,
            scale_factor=20.0,
            n_restarts=20,
            n_iter=100,
            tol=1e-3,
            feature_cols=STICKY_T_HMM_FEATURES,
        )

    if kind == REGIME_STUDENT_T_HMM:
        return StickyStudentTHMMRegimeModel(
            n_states=State_n,
            df=4.0,
            sticky=8.0,
            n_iter=30,
            device="cuda" if torch.cuda.is_available() else "cpu",
            random_state=SEED,
            verbose=False,
            feature_cols=STICKY_T_HMM_FEATURES,
            map_labels=True,
        )

    if kind == REGIME_XGBOOST:
        return XGBoostRegimeModel(
            horizon=20,
            feature_cols=SUPERVISED_REGIME_FEATURES,
        )

    if kind == REGIME_RNN:
        return NeuralSequenceRegimeModel(
            model_type=REGIME_RNN,
            seq_len=32,
            horizon=20,
            epochs=100,
            feature_cols=SUPERVISED_REGIME_FEATURES,
        )

    if kind == REGIME_LSTM:
        return NeuralSequenceRegimeModel(
            model_type=REGIME_LSTM,
            seq_len=32,
            horizon=20,
            epochs=100,
            feature_cols=SUPERVISED_REGIME_FEATURES,
        )

    if kind == REGIME_CNN:
        return NeuralSequenceRegimeModel(
            model_type=REGIME_CNN,
            seq_len=32,
            horizon=20,
            epochs=100,
            feature_cols=SUPERVISED_REGIME_FEATURES,
        )

    raise ValueError(f"Unknown regime type: {kind}")

# ============================================================
# FAST ARRAY DATA
# ============================================================

def regime_probability_matrix(df, n_states=State_n):
    """
    Regime probability matrix used in the DDQN state.

    Uses the soft regime probabilities (which carry the HMM transition-matrix
    belief) when they are present, otherwise falls back to a one-hot encoding of
    the hard regime label.
    """
    columns = [f"regime_p{k}" for k in range(n_states)]
    neutral = min(1, n_states - 1)

    if all(column in df.columns for column in columns):
        values = df[columns].to_numpy(dtype=np.float32)
    else:
        regimes = df["regime"].to_numpy(dtype=np.int64)
        values = np.zeros((len(df), n_states), dtype=np.float32)
        values[np.arange(len(df)), np.clip(regimes, 0, n_states - 1)] = 1.0

    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, None)

    row_sum = values.sum(axis=1, keepdims=True)
    bad = row_sum.squeeze(-1) <= 1e-12
    values = values / np.where(row_sum <= 1e-12, 1.0, row_sum)

    if bad.any():
        values[bad, :] = 0.0
        values[bad, neutral] = 1.0

    return values.astype(np.float32)


@dataclass
class FastMarketData:
    market: np.ndarray
    open: np.ndarray
    close: np.ndarray
    regime: np.ndarray
    regime_proba: np.ndarray
    index: pd.Index

    @classmethod
    def from_frame(cls, df):
        market = df[MARKET_STATE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
        market[:, 5] /= 100.0

        return cls(
            market=np.ascontiguousarray(market),
            open=np.ascontiguousarray(df["open"].to_numpy(dtype=np.float32)),
            close=np.ascontiguousarray(df["close"].to_numpy(dtype=np.float32)),
            regime=np.ascontiguousarray(df["regime"].to_numpy(dtype=np.int64)),
            regime_proba=np.ascontiguousarray(regime_probability_matrix(df)),
            index=df.index.copy(),
        )

    def __len__(self):
        return len(self.open)


class FastStateScaler:
    def __init__(self, mean, scale):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.scale[self.scale < 1e-12] = 1.0

    def transform(self, states):
        return (states - self.mean) / self.scale


def build_states(
    data: FastMarketData,
    row_indices: np.ndarray,
    side: np.ndarray,
    entry_price: np.ndarray,
    equity: np.ndarray,
    peak_equity: np.ndarray,
):
    batch_size = row_indices.shape[0]
    states = np.empty((batch_size, 22), dtype=np.float32)

    states[:, :15] = data.market[row_indices]

    position_open = side != 0
    states[:, 15] = position_open
    states[:, 16] = side

    prices = data.close[row_indices]
    unrealized = np.zeros(batch_size, dtype=np.float32)
    unrealized[position_open] = (
        (prices[position_open] - entry_price[position_open])
        / np.maximum(entry_price[position_open], 1e-12)
        * side[position_open]
    )
    states[:, 17] = unrealized

    states[:, 18] = np.maximum(
        0.0,
        1.0 - equity / np.maximum(peak_equity, 1e-12),
    )

    states[:, 19:22] = data.regime_proba[row_indices]

    return states


def fit_fast_state_scaler(data: FastMarketData, warmup):
    indices = np.arange(warmup, len(data) - 1, dtype=np.int64)
    count = len(indices)

    side = np.zeros(count, dtype=np.int8)
    entry = np.zeros(count, dtype=np.float32)
    equity = np.ones(count, dtype=np.float32)
    peak = np.ones(count, dtype=np.float32)

    states = build_states(data, indices, side, entry, equity, peak)

    mean = states.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = states.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-12] = 1.0

    return FastStateScaler(mean, scale), states.shape[1]


# ============================================================
# GPU REPLAY BUFFER
# ============================================================

class TensorReplayBuffer:
    def __init__(self, capacity, state_dim, pin_memory=True):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.size = 0
        self.position = 0

        pin_memory = bool(pin_memory and USE_CUDA)

        self.states = torch.empty((capacity, state_dim), dtype=torch.float32, pin_memory=pin_memory)
        self.next_states = torch.empty((capacity, state_dim), dtype=torch.float32, pin_memory=pin_memory)
        self.actions = torch.empty(capacity, dtype=torch.int64, pin_memory=pin_memory)
        self.rewards = torch.empty(capacity, dtype=torch.float32, pin_memory=pin_memory)
        self.dones = torch.empty(capacity, dtype=torch.float32, pin_memory=pin_memory)

    def add_batch(self, states, actions, rewards, next_states, dones):
        states = torch.from_numpy(np.asarray(states, dtype=np.float32))
        next_states = torch.from_numpy(np.asarray(next_states, dtype=np.float32))
        actions = torch.from_numpy(np.asarray(actions, dtype=np.int64))
        rewards = torch.from_numpy(np.asarray(rewards, dtype=np.float32))
        dones = torch.from_numpy(np.asarray(dones, dtype=np.float32))

        batch_size = states.shape[0]
        if batch_size >= self.capacity:
            states = states[-self.capacity:]
            actions = actions[-self.capacity:]
            rewards = rewards[-self.capacity:]
            next_states = next_states[-self.capacity:]
            dones = dones[-self.capacity:]
            batch_size = self.capacity

        first = min(batch_size, self.capacity - self.position)
        second = batch_size - first
        end = self.position + first

        self.states[self.position:end].copy_(states[:first])
        self.actions[self.position:end].copy_(actions[:first])
        self.rewards[self.position:end].copy_(rewards[:first])
        self.next_states[self.position:end].copy_(next_states[:first])
        self.dones[self.position:end].copy_(dones[:first])

        if second:
            self.states[:second].copy_(states[first:])
            self.actions[:second].copy_(actions[first:])
            self.rewards[:second].copy_(rewards[first:])
            self.next_states[:second].copy_(next_states[first:])
            self.dones[:second].copy_(dones[first:])

        self.position = (self.position + batch_size) % self.capacity
        self.size = min(self.capacity, self.size + batch_size)

    def sample(self, batch_size, device):
        indices = torch.randint(0, self.size, (batch_size,), device="cpu")
        return (
            self.states[indices].to(device, non_blocking=True),
            self.actions[indices].to(device, non_blocking=True),
            self.rewards[indices].to(device, non_blocking=True),
            self.next_states[indices].to(device, non_blocking=True),
            self.dones[indices].to(device, non_blocking=True),
        )

    def __len__(self):
        return self.size


# ============================================================
# DDQN
# ============================================================

class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, action_dim),
        )

    def forward(self, states):
        return self.net(states)


class BatchedDDQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate=1e-3,
        gamma=0.99,
        target_update_steps=250,
        compile_model=True,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.target_update_steps = target_update_steps
        self.update_count = 0

        self.online = QNet(state_dim, action_dim).to(DEVICE)
        self.target = QNet(state_dim, action_dim).to(DEVICE)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.AdamW(
            self.online.parameters(),
            lr=learning_rate,
            fused=USE_CUDA,
        )
        self.loss_function = nn.SmoothL1Loss()

        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=USE_CUDA)

        self.compile_enabled = False

        if compile_model and can_use_torch_compile():
            try:
                self.online = torch.compile(self.online, mode="reduce-overhead")
                self.target = torch.compile(self.target, mode="reduce-overhead")
                self.compile_enabled = True
                pass
            except Exception as error:
                pass

    @torch.inference_mode()
    def act_batch(self, states, epsilon=0.0):
        states_tensor = torch.as_tensor(states, dtype=torch.float32, device=DEVICE)

        with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_CUDA):
            greedy_actions = self.online(states_tensor).argmax(dim=1)

        actions = greedy_actions.cpu().numpy()

        if epsilon > 0:
            random_mask = np.random.random(len(actions)) < epsilon
            actions[random_mask] = np.random.randint(0, self.action_dim, size=random_mask.sum())

        return actions.astype(np.int64, copy=False)

    def train_step(self, replay, batch_size):
        if len(replay) < batch_size:
            return None

        states, actions, rewards, next_states, dones = replay.sample(batch_size, DEVICE)

        self.online.train()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_CUDA):
            current_q = self.online(states).gather(1, actions.unsqueeze(1))

            with torch.no_grad():
                next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target(next_states).gather(1, next_actions)
                target_q = rewards.unsqueeze(1) + self.gamma * next_q * (1.0 - dones.unsqueeze(1))

            loss = self.loss_function(current_q, target_q)

        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        self.update_count += 1
        if self.update_count % self.target_update_steps == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.detach().item())


def epsilon_by_episode(
    episode,
    episodes,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=5.0,
):
    if episodes <= 1:
        return epsilon_end

    progress = episode / max(episodes - 1, 1)
    return epsilon_end + (epsilon_start - epsilon_end) * np.exp(-epsilon_decay * progress)


# ============================================================
# BATCHED TRAINING ENVIRONMENT
# ============================================================

def calculate_batch_reward(
    equity_before,
    next_equity,
    peak_before,
    next_peak,
    drawdown_abs_penalty,
    drawdown_increase_penalty,
    reward_clip,
):
    step_return = (next_equity - equity_before) / np.maximum(equity_before, 1e-12)

    current_drawdown = np.maximum(0.0, 1.0 - equity_before / np.maximum(peak_before, 1e-12))
    next_drawdown = np.maximum(0.0, 1.0 - next_equity / np.maximum(next_peak, 1e-12))
    drawdown_increase = np.maximum(0.0, next_drawdown - current_drawdown)

    reward = (
        step_return
        - drawdown_abs_penalty * next_drawdown
        - drawdown_increase_penalty * drawdown_increase
    )

    if reward_clip is not None:
        np.clip(reward, -reward_clip, reward_clip, out=reward)

    return reward.astype(np.float32)


def execute_actions_batch(
    actions,
    open_price,
    side,
    entry_price,
    quantity,
    account_base,
    equity_at_entry,
    fee_rate,
):
    desired_side = np.zeros_like(side)
    desired_side[actions == ACTION_BUY] = 1
    desired_side[actions == ACTION_SELL] = -1

    active_action = actions != ACTION_HOLD
    same_position = (side != 0) & (side == desired_side)
    change_position = active_action & ~same_position

    closing = change_position & (side != 0)

    if closing.any():
        close_equity = (
            account_base[closing]
            + (open_price[closing] - entry_price[closing]) * quantity[closing] * side[closing]
        )
        exit_fee = np.abs(quantity[closing] * open_price[closing]) * fee_rate
        account_base[closing] = close_equity - exit_fee

    opening = change_position

    if opening.any():
        equity_before_entry = account_base[opening].copy()
        entry_fee = equity_before_entry * fee_rate
        tradable_equity = equity_before_entry - entry_fee

        valid = tradable_equity > 0
        opening_indices = np.flatnonzero(opening)
        valid_indices = opening_indices[valid]

        side[opening_indices] = 0
        quantity[opening_indices] = 0.0
        entry_price[opening_indices] = 0.0
        equity_at_entry[opening_indices] = 0.0

        side[valid_indices] = desired_side[valid_indices]
        entry_price[valid_indices] = open_price[valid_indices]
        quantity[valid_indices] = tradable_equity[valid] / np.maximum(open_price[valid_indices], 1e-12)
        account_base[valid_indices] = tradable_equity[valid]
        equity_at_entry[valid_indices] = equity_before_entry[valid]


def mark_to_market_batch(price, side, entry_price, quantity, account_base):
    return account_base + (price - entry_price) * quantity * side


def train_batched_ddqn(
    train_data: FastMarketData,
    episodes=100,
    n_envs=64,
    fee_rate=0.0,
    warmup=180,
    initial_equity=10_000.0,
    batch_size=2048,
    replay_capacity=1_000_000,
    train_every=4,
    gradient_steps=1,
    drawdown_abs_penalty=10.0,
    drawdown_increase_penalty=80.0,
    reward_clip=10.0,
    compile_model=True,
):
    if len(train_data) <= warmup + 2:
        raise ValueError("Training data is too small for warmup.")

    scaler, state_dim = fit_fast_state_scaler(train_data, warmup)

    agent = BatchedDDQNAgent(
        state_dim=state_dim,
        action_dim=N_ACTIONS,
        learning_rate=1e-3,
        gamma=0.99,
        target_update_steps=10,
        compile_model=compile_model,
    )

    replay = TensorReplayBuffer(
        capacity=replay_capacity,
        state_dim=state_dim,
        pin_memory=True,
    )

    episode_groups = (episodes + n_envs - 1) // n_envs
    completed_episodes = 0
    global_step = 0

    # progress = tqdm(total=episodes, desc="Batched DDQN")

    for group in range(episode_groups):
        active_envs = min(n_envs, episodes - completed_episodes)
        episode_ids = np.arange(active_envs) + completed_episodes

        epsilons = np.array(
            [epsilon_by_episode(ep, episodes) for ep in episode_ids],
            dtype=np.float32,
        )

        side = np.zeros(active_envs, dtype=np.int8)
        entry_price = np.zeros(active_envs, dtype=np.float32)
        quantity = np.zeros(active_envs, dtype=np.float32)
        account_base = np.full(active_envs, initial_equity, dtype=np.float32)
        equity_at_entry = np.zeros(active_envs, dtype=np.float32)
        peak_equity = np.full(active_envs, initial_equity, dtype=np.float32)
        episode_reward = np.zeros(active_envs, dtype=np.float64)
        latest_losses = []

        for idx in range(warmup + 1, len(train_data) - 1):
            open_price = np.full(active_envs, train_data.open[idx], dtype=np.float32)

            equity_before = mark_to_market_batch(
                open_price,
                side,
                entry_price,
                quantity,
                account_base,
            )
            peak_equity = np.maximum(peak_equity, equity_before)
            peak_before = peak_equity.copy()

            row_indices = np.full(active_envs, idx - 1, dtype=np.int64)
            raw_states = build_states(
                train_data,
                row_indices,
                side,
                entry_price,
                equity_before,
                peak_equity,
            )
            states = scaler.transform(raw_states)

            greedy_actions = agent.act_batch(states, epsilon=0.0)
            random_mask = np.random.random(active_envs) < epsilons
            random_actions = np.random.randint(0, N_ACTIONS, size=active_envs)
            actions = np.where(random_mask, random_actions, greedy_actions)

            execute_actions_batch(
                actions=actions,
                open_price=open_price,
                side=side,
                entry_price=entry_price,
                quantity=quantity,
                account_base=account_base,
                equity_at_entry=equity_at_entry,
                fee_rate=fee_rate,
            )

            # End the transition at the next decision time, not a later close.
            next_price = np.full(active_envs, train_data.open[idx + 1], dtype=np.float32)
            next_equity = mark_to_market_batch(
                next_price,
                side,
                entry_price,
                quantity,
                account_base,
            )
            next_peak = np.maximum(peak_equity, next_equity)

            if idx == len(train_data) - 2:
                exit_fee = np.abs(quantity * next_price) * fee_rate
                next_equity = next_equity - exit_fee
                account_base[:] = next_equity
                side[:] = 0
                quantity[:] = 0.0
                entry_price[:] = 0.0
                equity_at_entry[:] = 0.0

            rewards = calculate_batch_reward(
                equity_before=equity_before,
                next_equity=next_equity,
                peak_before=peak_before,
                next_peak=next_peak,
                drawdown_abs_penalty=drawdown_abs_penalty,
                drawdown_increase_penalty=drawdown_increase_penalty,
                reward_clip=reward_clip,
            )

            next_indices = np.full(active_envs, idx, dtype=np.int64)
            next_raw_states = build_states(
                train_data,
                next_indices,
                side,
                entry_price,
                next_equity,
                next_peak,
            )
            next_states = scaler.transform(next_raw_states)

            done_value = float(idx >= len(train_data) - 2)
            dones = np.full(active_envs, done_value, dtype=np.float32)

            replay.add_batch(states, actions, rewards, next_states, dones)

            if global_step % train_every == 0:
                for _ in range(gradient_steps):
                    loss = agent.train_step(replay, batch_size)
                    if loss is not None:
                        latest_losses.append(loss)

            episode_reward += rewards
            peak_equity = next_peak
            global_step += 1

        final_equity = mark_to_market_batch(
            np.full(active_envs, train_data.open[-1], dtype=np.float32),
            side,
            entry_price,
            quantity,
            account_base,
        )

        completed_episodes += active_envs
        # progress.update(active_envs)
        # progress.set_postfix(
        #     reward=f"{episode_reward.mean():.3f}",
        #     equity=f"{final_equity.mean():.2f}",
        #     loss=(f"{np.mean(latest_losses):.5f}" if latest_losses else "warming"),
        #     replay=len(replay),
        # )

    # progress.close()
    return agent, scaler


# ============================================================
# BACKTEST
# ============================================================

@dataclass
class Trade:
    side: int
    entry_time: Any
    exit_time: Any
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    pnl_pct: float
    fee_paid: float
    bars_held: int
    equity_before: float
    equity_after: float
    reason: str


import numpy as np
import pandas as pd


def calculate_metrics(equity_list, risk_free_rate=0.0, periods_per_year=252):
    equity = pd.Series(equity_list)
    returns = equity.pct_change().dropna()

    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1

    peak = equity.expanding(min_periods=1).max()
    drawdown = (equity - peak) / peak
    max_drawdown = drawdown.min()

    n_periods = len(equity)
    annualized_return = ((1 + total_return) ** (periods_per_year / n_periods)) - 1

    mean_return = returns.mean() * periods_per_year
    std_return = returns.std() * np.sqrt(periods_per_year)
    sharpe_ratio = (mean_return - risk_free_rate) / std_return if std_return != 0 else 0

    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(periods_per_year)
    sortino_ratio = (mean_return - risk_free_rate) / downside_std if downside_std != 0 else 0

    return {
        "Total Return": f"{total_return:.2%}",
        "Annualized Return": f"{annualized_return:.2%}",
        "Maximum Drawdown": f"{max_drawdown:.2%}",
        "Sharpe Ratio": round(sharpe_ratio, 2),
        "Sortino Ratio": round(sortino_ratio, 2)
    }


def run_fast_backtest(
    data: FastMarketData,
    agent: BatchedDDQNAgent,
    scaler: FastStateScaler,
    fee_rate=0.0,
    initial_equity=10_000.0,
    warmup=180,
):
    if len(data) <= warmup + 2:
        raise ValueError("Test data is too small for warmup.")

    side = 0
    entry_price = 0.0
    quantity = 0.0
    account_base = initial_equity
    entry_equity = 0.0
    entry_fee = 0.0
    entry_index = -1

    peak_equity = initial_equity
    trades: List[Trade] = []
    actions_output = []

    equity_values = [initial_equity]
    equity_times = [data.index[warmup]]

    for idx in range(warmup + 1, len(data) - 1):
        open_price = float(data.open[idx])

        current_equity = account_base + (open_price - entry_price) * quantity * side
        peak_equity = max(peak_equity, current_equity)

        raw_state = build_states(
            data,
            np.array([idx - 1]),
            np.array([side], dtype=np.int8),
            np.array([entry_price], dtype=np.float32),
            np.array([current_equity], dtype=np.float32),
            np.array([peak_equity], dtype=np.float32),
        )
        state = scaler.transform(raw_state)
        action = int(agent.act_batch(state, epsilon=0.0)[0])

        desired_side = 1 if action == ACTION_BUY else -1 if action == ACTION_SELL else 0
        should_change = action != ACTION_HOLD and desired_side != side

        if should_change and side != 0:
            equity_before_fee = current_equity
            exit_fee = abs(quantity * open_price) * fee_rate
            equity_after = equity_before_fee - exit_fee
            pnl = equity_after - entry_equity

            trades.append(
                Trade(
                    side=side,
                    entry_time=data.index[entry_index],
                    exit_time=data.index[idx],
                    entry_price=entry_price,
                    exit_price=open_price,
                    qty=quantity,
                    pnl=pnl,
                    pnl_pct=safe_div(pnl, entry_equity),
                    fee_paid=entry_fee + exit_fee,
                    bars_held=idx - entry_index + 1,
                    equity_before=equity_before_fee,
                    equity_after=equity_after,
                    reason="reverse_to_buy" if desired_side == 1 else "reverse_to_sell",
                )
            )

            account_base = equity_after
            side = 0
            quantity = 0.0

        if should_change:
            entry_equity = account_base
            entry_fee = account_base * fee_rate
            account_base -= entry_fee
            entry_price = open_price
            quantity = account_base / max(open_price, 1e-12)
            side = desired_side
            entry_index = idx

        # Mark at the next open, before the next action.
        next_price = float(data.open[idx + 1])
        next_equity = account_base + (next_price - entry_price) * quantity * side

        peak_equity = max(peak_equity, next_equity)
        equity_values.append(next_equity)
        equity_times.append(data.index[idx + 1])

        actions_output.append(
            {
                "time": data.index[idx],
                "action": "BUY" if action == ACTION_BUY else "SELL" if action == ACTION_SELL else "HOLD",
                "equity_before": current_equity,
                "equity_after": next_equity,
                "position_side": side if side else None,
                "regime": int(data.regime[idx - 1]),
            }
        )

    if side != 0:
        idx = len(data) - 1
        exit_price = float(data.open[idx])

        equity_before_fee = account_base + (exit_price - entry_price) * quantity * side
        exit_fee = abs(quantity * exit_price) * fee_rate
        final_equity = equity_before_fee - exit_fee
        pnl = final_equity - entry_equity

        trades.append(
            Trade(
                side=side,
                entry_time=data.index[entry_index],
                exit_time=data.index[idx],
                entry_price=entry_price,
                exit_price=exit_price,
                qty=quantity,
                pnl=pnl,
                pnl_pct=safe_div(pnl, entry_equity),
                fee_paid=entry_fee + exit_fee,
                bars_held=idx - entry_index + 1,
                equity_before=equity_before_fee,
                equity_after=final_equity,
                reason="final_forced_close",
            )
        )

        # The final open is already the last mark. Apply its exit fee
        # to that mark instead of adding a duplicate timestamp.
        equity_values[-1] = final_equity
        if actions_output:
            actions_output[-1]["equity_after"] = final_equity

    equity_curve = np.asarray(equity_values, dtype=np.float64)
    returns = np.diff(equity_curve) / np.maximum(equity_curve[:-1], 1e-12)

    final_equity = float(equity_curve[-1])
    net_profit = final_equity - initial_equity

    wins = [trade for trade in trades if trade.pnl > 0]
    losses = [trade for trade in trades if trade.pnl <= 0]

    gross_profit = float(sum(trade.pnl for trade in wins))
    gross_loss = float(-sum(trade.pnl for trade in losses))

    summary = {
        # "initial_equity": initial_equity,
        "final_equity": final_equity,
        # "net_profit": net_profit,
        # "total_return": safe_div(net_profit, initial_equity),
        # "max_drawdown": max_drawdown(equity_curve),
        # "sharpe": sharpe_from_returns(returns),
        # "num_trades": len(trades),
        # "win_rate": safe_div(len(wins), len(trades)),
        # "profit_factor": safe_div(gross_profit, gross_loss) if gross_loss > 0 else np.inf,
        # "gross_profit": gross_profit,
        # "gross_loss": gross_loss,
    }

    trades_df = pd.DataFrame([trade.__dict__ for trade in trades])

    equity_df = pd.DataFrame({"time": equity_times, "equity": equity_curve}).set_index("time")

    actions_df = pd.DataFrame(actions_output)
    if not actions_df.empty:
        actions_df = actions_df.set_index("time")

    return summary, trades_df, equity_df, actions_df


# ============================================================
# EXPERIMENTS
# ============================================================

def causal_regime_probabilities(regime_model, history_df, prediction_df):
    if prediction_df.empty:
        return pd.DataFrame(index=prediction_df.index, columns=REGIME_PROBA_COLUMNS, dtype=np.float32)
    if not history_df.empty and history_df.index.max() >= prediction_df.index.min():
        raise ValueError("Regime history must end strictly before the prediction block.")
    context = pd.concat([history_df, prediction_df])
    context["regime"] = regime_model.transform_df(context)
    probabilities = regime_proba_frame(regime_model, context)
    return probabilities.loc[prediction_df.index, REGIME_PROBA_COLUMNS].copy()


def prepare_crossfitted_regimes(base_train, base_test, regime_kind):
    if base_train.index.max() >= base_test.index.min():
        raise ValueError("Regime training and test periods must not overlap.")
    if REGIME_MIN_FIT_BARS < 64 or REGIME_REFIT_BARS < 1:
        raise ValueError("Invalid chronological regime fitting settings.")
    if len(base_train) <= REGIME_MIN_FIT_BARS + 2:
        raise ValueError("Not enough training bars after the initial regime-fitting period.")

    prediction_index = base_train.index[REGIME_MIN_FIT_BARS:]
    train_proba = pd.DataFrame(
        np.nan, index=prediction_index, columns=REGIME_PROBA_COLUMNS, dtype=np.float32,
    )
    fit_windows = []

    for block_start in range(REGIME_MIN_FIT_BARS, len(base_train), REGIME_REFIT_BARS):
        block_end = min(block_start + REGIME_REFIT_BARS, len(base_train))
        fit_df = base_train.iloc[:block_start].copy()
        prediction_df = base_train.iloc[block_start:block_end].copy()
        regime_model = build_regime_model(regime_kind)
        regime_model.fit(fit_df)
        block_proba = causal_regime_probabilities(regime_model, fit_df, prediction_df)
        train_proba.loc[prediction_df.index, :] = block_proba.to_numpy(dtype=np.float32)
        fit_windows.append({
            "stage": "ddqn_training",
            "fit_start": fit_df.index.min().isoformat(),
            "fit_end": fit_df.index.max().isoformat(),
            "predict_start": prediction_df.index.min().isoformat(),
            "predict_end": prediction_df.index.max().isoformat(),
            "target_horizon_bars": int(getattr(regime_model, "horizon", 0)),
        })

    if not np.isfinite(train_proba.to_numpy()).all():
        raise ValueError("Chronological regime predictions are incomplete or non-finite.")

    regime_model = build_regime_model(regime_kind)
    regime_model.fit(base_train.copy())
    test_proba = causal_regime_probabilities(regime_model, base_train, base_test)
    if not np.isfinite(test_proba.to_numpy()).all():
        raise ValueError("Test regime predictions are non-finite.")
    fit_windows.append({
        "stage": "held_out_test",
        "fit_start": base_train.index.min().isoformat(),
        "fit_end": base_train.index.max().isoformat(),
        "predict_start": base_test.index.min().isoformat(),
        "predict_end": base_test.index.max().isoformat(),
        "target_horizon_bars": int(getattr(regime_model, "horizon", 0)),
    })

    train_df = base_train.loc[prediction_index].copy()
    test_df = base_test.copy()
    for frame, probabilities in ((train_df, train_proba), (test_df, test_proba)):
        frame["regime"] = np.argmax(probabilities.to_numpy(), axis=1).astype(np.int64)
        for column in REGIME_PROBA_COLUMNS:
            frame[column] = probabilities[column].to_numpy(dtype=np.float32)
    return train_df, test_df, fit_windows


def run_all_experiments(
    df,
    fee_rate=0.0,
    episodes=100,
    n_envs=64,
    warmup=180,
    initial_equity=10_000.0,
    batch_size=2048,
    replay_capacity=500_000,
    train_every=4,
    gradient_steps=1,
    drawdown_abs_penalty=10.0,
    drawdown_increase_penalty=80.0,
    reward_clip=10.0,
    train_period=None,
    test_period=None,
    feature_lookback_bars=250,
    compile_model=True,
):
    train_period = TRAIN_PERIOD if train_period is None else train_period
    test_period = TEST_PERIOD if test_period is None else test_period
    base_train, base_test = prepare_train_test_features_no_leak(
        df,
        train_period=train_period,
        test_period=test_period,
        lookback_bars=feature_lookback_bars,
    )

    experiments = [
        ("DDQN + PoissonHMM regime", REGIME_POISSON_HMM),
        ("DDQN + XGBoost regime", REGIME_XGBOOST),
        ("DDQN + KMeans", REGIME_KMEANS),
        ("DDQN + HMM regime", REGIME_HMM),
        ("DDQN + LSTM regime", REGIME_LSTM),
        ("DDQN-only", REGIME_NONE),
        ("DDQN + RNN regime", REGIME_RNN),
        ("DDQN + CNN regime", REGIME_CNN),
    ]

    results = []
    outputs: Dict[str, dict] = {}
    regime_result = {}
    for experiment_name, regime_kind in experiments:

        if regime_kind in (REGIME_HMM, REGIME_POISSON_HMM) and not HMM_AVAILABLE:
            continue
            continue

        if regime_kind == REGIME_POISSON_HMM and PoissonHMM is None:
            continue
            continue

        if regime_kind == REGIME_XGBOOST and not XGBOOST_AVAILABLE:
            continue
            continue

        # print("\n" + "=" * 90)
        # print(experiment_name)
        # # print("=" * 90)

        train_df, test_df, fit_windows = prepare_crossfitted_regimes(
            base_train, base_test, regime_kind,
        )

        train_data = FastMarketData.from_frame(train_df)
        test_data = FastMarketData.from_frame(test_df)

        agent, scaler = train_batched_ddqn(
            train_data=train_data,
            episodes=episodes,
            n_envs=n_envs,
            fee_rate=fee_rate,
            warmup=warmup,
            initial_equity=initial_equity,
            batch_size=batch_size,
            replay_capacity=replay_capacity,
            train_every=train_every,
            gradient_steps=gradient_steps,
            drawdown_abs_penalty=drawdown_abs_penalty,
            drawdown_increase_penalty=drawdown_increase_penalty,
            reward_clip=reward_clip,
            compile_model=compile_model,
        )

        summary, trades_df, equity_df, actions_df = run_fast_backtest(
            data=test_data,
            agent=agent,
            scaler=scaler,
            fee_rate=fee_rate,
            initial_equity=initial_equity,
            warmup=warmup,
        )

        summary.update(
            {
                # "experiment": experiment_name,
                # "regime_kind": regime_kind,
                # "train_start": train_period[0],
                # "train_end": train_period[1],
                # "test_start": test_period[0],
                # "test_end": test_period[1],
            }
        )

        results.append(summary)
        outputs[experiment_name] = {
            "summary": summary,
            "trades_df": trades_df,
            "equity_df": equity_df,
            "actions_df": actions_df,
            "train_regimes": train_df["regime"].copy(),
            "test_regimes": test_df["regime"].copy(),
            "regime_fit_windows": fit_windows,
        }

        regime_result[experiment_name] = calculate_metrics(list(equity_df['equity']))

        # print("Backtest:")
        # for key, value in summary.items():
        #     if isinstance(value, float):
        #         print(f"{key:20s}: {value:.6f}")
        #     else:
        #         print(f"{key:20s}: {value}")

        if USE_CUDA:
            torch.cuda.empty_cache()


    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values("final_equity", ascending=False).reset_index(drop=True)

    return results_df, outputs,regime_result




# equity_data = list(equity_df['equity'])
# metrics = calculate_metrics(equity_data)
# print(metrics)

def set_seed(SEED):

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# ============================================================
# MAIN
# ============================================================
import multiprocessing as mp
from collections import defaultdict





def run_single_seed(job):
    data_path, seed = job
    global SEED
    SEED = seed
    set_seed(SEED)
    df = load_pickle_any(data_path)

    results_df, outputs, regime_result = run_all_experiments(
        df=df,
        fee_rate=FEE_RATE,
        episodes=EPISODES,
        n_envs=N_ENVS,
        warmup=WARMUP,
        initial_equity=INITIAL_EQUITY,
        batch_size=BATCH_SIZE,
        replay_capacity=REPLAY_CAPACITY,
        train_every=TRAIN_EVERY,
        gradient_steps=GRADIENT_STEPS,
        drawdown_abs_penalty=DRAWDOWN_ABS_PENALTY,
        drawdown_increase_penalty=DRAWDOWN_INCREASE_PENALTY,
        reward_clip=REWARD_CLIP,
        train_period=TRAIN_PERIOD,
        test_period=TEST_PERIOD,
        feature_lookback_bars=FEATURE_LOOKBACK_BARS,
        compile_model=COMPILE_MODEL,
    )

    normalized_equity = {}
    for experiment_name, bundle in outputs.items():
        equity = bundle["equity_df"]["equity"].copy()
        if not equity.empty and float(equity.iloc[0]) != 0.0:
            normalized_equity[experiment_name] = equity / float(equity.iloc[0])

    return {
        "seed": seed,
        "metrics": regime_result,
        "normalized_equity": normalized_equity,
        "regime_fit_windows": {
            name: bundle["regime_fit_windows"] for name, bundle in outputs.items()
        },
    }


def parse_percent(value):
    if isinstance(value, str):
        return float(value.strip().replace("%", "")) / 100.0
    return float(value)


def summarize_seed_results(all_seed_outputs):
    """Average the metrics that appear in the paper's final table."""
    values = defaultdict(list)
    for seed_output in all_seed_outputs:
        for regime, metric in seed_output["metrics"].items():
            values[regime].append(metric)

    rows = []
    for regime, metrics in values.items():
        returns = [parse_percent(metric["Total Return"]) * 100.0 for metric in metrics]
        sharpes = [float(metric["Sharpe Ratio"]) for metric in metrics]
        sortinos = [
            float(metric["Sortino Ratio"])
            for metric in metrics
            if math.isfinite(float(metric["Sortino Ratio"]))
        ]
        drawdowns = [
            abs(parse_percent(metric["Maximum Drawdown"])) * 100.0
            for metric in metrics
        ]
        rows.append(
            {
                "regime": regime,
                "mean_total_return": float(np.mean(returns)),
                "mean_sharpe": float(np.mean(sharpes)),
                "mean_sortino": float(np.mean(sortinos)) if sortinos else float("nan"),
                "mean_mdd_pct": float(np.mean(drawdowns)),
                "seeds": len(metrics),
            }
        )

    return pd.DataFrame(rows)


MODEL_LABELS = {
    "DDQN + PoissonHMM regime": "Poisson HMM",
    "DDQN + XGBoost regime": "XGBoost",
    "DDQN + KMeans": "K-means",
    "DDQN + HMM regime": "Gaussian HMM",
    "DDQN + LSTM regime": "LSTM",
    "DDQN-only": "No learned context",
    "DDQN + RNN regime": "RNN",
    "DDQN + CNN regime": "1D-CNN",
}

MODEL_ORDER = [
    "DDQN + PoissonHMM regime",
    "DDQN + XGBoost regime",
    "DDQN + KMeans",
    "DDQN + HMM regime",
    "DDQN + LSTM regime",
    "DDQN-only",
    "DDQN + RNN regime",
    "DDQN + CNN regime",
]


def format_final_table(summary_df):
    by_model = {row["regime"]: row for row in summary_df.to_dict("records")}
    rows = []
    for model in MODEL_ORDER:
        if model not in by_model:
            continue
        row = by_model[model]
        rows.append(
            {
                "Context variant": MODEL_LABELS[model],
                "Return (%)": row["mean_total_return"],
                "Sharpe": row["mean_sharpe"],
                "Sortino": row["mean_sortino"],
                "MDD (%)": row["mean_mdd_pct"],
            }
        )

    headers = ["Context variant", "Return (%)", "Sharpe", "Sortino", "MDD (%)"]
    widths = [len(header) for header in headers]
    for row in rows:
        widths[0] = max(widths[0], len(row[headers[0]]))
        for column, header in enumerate(headers[1:], start=1):
            value = row[header]
            text = "nan" if not math.isfinite(value) else f"{value:.2f}"
            widths[column] = max(widths[column], len(text))

    def format_row(row):
        cells = [str(row[headers[0]]).ljust(widths[0])]
        for column, header in enumerate(headers[1:], start=1):
            value = row[header]
            text = "nan" if not math.isfinite(value) else f"{value:.2f}"
            cells.append(text.rjust(widths[column]))
        return "  ".join(cells)

    header_line = "  ".join(
        header.ljust(widths[index]) if index == 0 else header.rjust(widths[index])
        for index, header in enumerate(headers)
    )
    rule = "  ".join("-" * width for width in widths)
    body = [format_row(row) for row in rows]
    return "\n".join([header_line, rule, *body])


def quiet_run_single_seed(job):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return run_single_seed(job)


def main():
    seeds = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    asset_labels = {
        "BTCUSD_daily": "Bitcoin",
        "ETHUSD_daily": "Ether",
    }

    for data_path in DATA_PATHS:
        symbol = os.path.splitext(os.path.basename(data_path))[0]
        jobs = [(data_path, seed) for seed in seeds]

        with tqdm(
            total=len(jobs),
            desc=f"{asset_labels.get(symbol, symbol)} seeds",
            unit="seed",
            dynamic_ncols=True,
        ) as progress:
            if USE_CUDA:
                all_seed_outputs = []
                for job in jobs:
                    all_seed_outputs.append(quiet_run_single_seed(job))
                    progress.update(1)
            else:
                with mp.get_context("spawn").Pool(processes=POOL_PROCESSES) as pool:
                    all_seed_outputs = []
                    for output in pool.imap_unordered(quiet_run_single_seed, jobs, chunksize=1):
                        all_seed_outputs.append(output)
                        progress.update(1)

        all_seed_outputs.sort(key=lambda output: output["seed"])
        summary_df = summarize_seed_results(all_seed_outputs)
        print(f"Final results: {asset_labels.get(symbol, symbol)}")
        print(format_final_table(summary_df))


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
