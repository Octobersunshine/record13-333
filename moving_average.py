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


def weighted_moving_average_smooth(data, window_size, weights=None, center=False, min_periods=1):
    """
    加权移动平均平滑服务

    Parameters
    ----------
    data : array-like
        输入的时间序列数据
    window_size : int
        移动窗口大小
    weights : array-like, optional
        权重数组，长度必须等于 window_size，默认为线性递增权重
    center : bool, optional
        是否使用中心窗口，默认为 False
    min_periods : int, optional
        窗口中最少需要的非空值数量，默认为 1

    Returns
    -------
    pd.Series
        平滑后的时间序列
    """
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("window_size 必须为正整数")

    if not isinstance(min_periods, int) or min_periods <= 0 or min_periods > window_size:
        raise ValueError("min_periods 必须为 1 到 window_size 之间的正整数")

    if weights is None:
        weights = np.arange(1, window_size + 1)
    else:
        weights = np.asarray(weights)
        if len(weights) != window_size:
            raise ValueError("权重数组长度必须等于 window_size")

    weights_full = weights
    series = pd.Series(data)

    def _weighted_mean(window):
        n = len(window)
        if n < min_periods:
            return np.nan
        w = weights_full[-n:]
        w = w / w.sum()
        return np.dot(window, w)

    smoothed = series.rolling(
        window=window_size,
        center=center,
        min_periods=min_periods
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

    print("原始数据（前5个）：", data[:5].round(4))
    print()

    ma_simple = moving_average_smooth(data, window_size=5)
    print("简单移动平均（窗口=5，前5个）：", ma_simple[:5].round(4).tolist())
    print()

    ma_center = moving_average_smooth(data, window_size=5, center=True)
    print("中心移动平均（窗口=5，前5个）：", ma_center[:5].round(4).tolist())
    print()

    wma = weighted_moving_average_smooth(data, window_size=5)
    print("加权移动平均（窗口=5，前5个）：", wma[:5].round(4).tolist())
    print()

    wma_verify = weighted_moving_average_smooth([1, 2, 3, 4, 5], window_size=5, weights=[1, 2, 3, 4, 5])
    print("加权移动平均验证（数据=[1,2,3,4,5], 权重=[1,2,3,4,5]）：", wma_verify.round(4).tolist())
    expected = (1*1 + 2*2 + 3*3 + 4*4 + 5*5) / (1+2+3+4+5)
    print("预期最后一个值：", round(expected, 4))
    print()

    ema = exponential_moving_average_smooth(data, span=5)
    print("指数移动平均（span=5，前5个）：", ema[:5].round(4).tolist())

    print()
    print("=" * 60)
    print("边界处理验证：")
    print("-" * 60)
    test_data = [10, 20, 30, 40, 50]
    print("测试数据：", test_data)
    print()

    wma_test = weighted_moving_average_smooth(test_data, window_size=3, weights=[1, 2, 3])
    print("加权移动平均（窗口=3, 权重=[1,2,3]）：", wma_test.round(4).tolist())

    print("  第1个值(10): 仅1个有效数据 -> 10 * (3/3) = 10")
    print("  第2个值(20): 有效数据[10,20], 权重[2,3] -> (10*2 + 20*3)/(2+3) =", round((10*2 + 20*3)/5, 4))
    print("  第3个值(30): 有效数据[10,20,30], 权重[1,2,3] -> (10*1 + 20*2 + 30*3)/6 =", round((10*1 + 20*2 + 30*3)/6, 4))
