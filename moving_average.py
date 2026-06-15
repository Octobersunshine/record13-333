import pandas as pd
import numpy as np


def moving_average_smooth(data, window_size, center=False, min_periods=1):
    """
    移动平均平滑服务

    Parameters
    ----------
    data : array-like
        输入的时间序列数据，可以是列表、NumPy 数组或 Pandas Series
    window_size : int
        移动窗口大小，必须为正整数
    center : bool, optional
        是否使用中心窗口，默认为 False（使用右对齐窗口）
    min_periods : int, optional
        窗口中最少需要的非空值数量，默认为 1

    Returns
    -------
    pd.Series
        平滑后的时间序列

    Raises
    ------
    ValueError
        如果 window_size 不是正整数
    """
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("window_size 必须为正整数")

    series = pd.Series(data)

    smoothed = series.rolling(
        window=window_size,
        center=center,
        min_periods=min_periods
    ).mean()

    return smoothed


def weighted_moving_average_smooth(data, window_size, weights=None, center=False):
    """
    加权移动平均平滑服务

    Parameters
    ----------
    data : array-like
        输入的时间序列数据
    window_size : int
        移动窗口大小
    weights : array-like, optional
        权重数组，长度必须等于 window_size，默认为线性递减权重
    center : bool, optional
        是否使用中心窗口，默认为 False

    Returns
    -------
    pd.Series
        平滑后的时间序列
    """
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("window_size 必须为正整数")

    if weights is None:
        weights = np.arange(1, window_size + 1)
    else:
        weights = np.asarray(weights)
        if len(weights) != window_size:
            raise ValueError("权重数组长度必须等于 window_size")

    weights = weights / weights.sum()

    series = pd.Series(data)

    def _weighted_mean(window):
        return np.dot(window, weights)

    smoothed = series.rolling(
        window=window_size,
        center=center
    ).apply(_weighted_mean, raw=True)

    return smoothed


def exponential_moving_average_smooth(data, span=None, alpha=None):
    """
    指数移动平均平滑服务（EMA）

    Parameters
    ----------
    data : array-like
        输入的时间序列数据
    span : float, optional
        指数衰减跨度，与 alpha 二选一
    alpha : float, optional
        平滑因子，0 < alpha <= 1，与 span 二选一

    Returns
    -------
    pd.Series
        平滑后的时间序列
    """
    if span is None and alpha is None:
        raise ValueError("必须指定 span 或 alpha 其中之一")
    if span is not None and alpha is not None:
        raise ValueError("span 和 alpha 只能指定一个")

    series = pd.Series(data)
    smoothed = series.ewm(span=span, alpha=alpha, adjust=False).mean()

    return smoothed


if __name__ == "__main__":
    np.random.seed(42)
    data = np.sin(np.linspace(0, 4 * np.pi, 100)) + np.random.normal(0, 0.3, 100)

    print("原始数据（前10个）：", data[:5].round(4))
    print()

    ma_simple = moving_average_smooth(data, window_size=5)
    print("简单移动平均（窗口=5，前10个）：", ma_simple[:5].round(4).tolist())
    print()

    ma_center = moving_average_smooth(data, window_size=5, center=True)
    print("中心移动平均（窗口=5，前10个）：", ma_center[:5].round(4).tolist())
    print()

    wma = weighted_moving_average_smooth(data, window_size=5)
    print("加权移动平均（窗口=5，前10个）：", wma[:5].round(4).tolist())
    print()

    ema = exponential_moving_average_smooth(data, span=5)
    print("指数移动平均（span=5，前10个）：", ema[:5].round(4).tolist())
