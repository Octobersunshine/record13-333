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


def exponential_moving_average_smooth(
    data,
    span=None,
    alpha=None,
    com=None,
    halflife=None,
    adjust=False,
    bias_correction=False,
    min_periods=1
):
    """
    指数加权移动平均平滑服务（EWMA / EMA）

    提供四种参数化方式（四选一）：
    - alpha: 直接指定平滑因子，0 < alpha <= 1
    - span: 衰减跨度，alpha = 2 / (span + 1)
    - com: 质心，alpha = 1 / (com + 1)
    - halflife: 半衰期，alpha = 1 - exp(-ln(2) / halflife)

    Parameters
    ----------
    data : array-like
        输入的时间序列数据
    span : float, optional
        指数衰减跨度，如 span=5 则 alpha≈0.333
    alpha : float, optional
        平滑因子，0 < alpha <= 1
    com : float, optional
        质心参数，com >= 0
    halflife : float, optional
        半衰期，halflife > 0
    adjust : bool, optional
        是否使用调整后的权重（True：使用递归公式的精确形式；False：使用近似形式），默认为 False
    bias_correction : bool, optional
        是否进行偏差校正（仅在 adjust=False 时有效），默认为 False。
        对初期小样本进行偏差校正，使估计更准确。
    min_periods : int, optional
        最少观测值数量，默认为 1

    Returns
    -------
    pd.Series
        平滑后的时间序列
    """
    param_count = sum(x is not None for x in [span, alpha, com, halflife])
    if param_count == 0:
        raise ValueError("必须指定 span、alpha、com、halflife 其中之一")
    if param_count > 1:
        raise ValueError("span、alpha、com、halflife 只能指定一个")

    if alpha is not None and not (0 < alpha <= 1):
        raise ValueError("alpha 必须满足 0 < alpha <= 1")
    if span is not None and span <= 0:
        raise ValueError("span 必须为正数")
    if com is not None and com < 0:
        raise ValueError("com 必须 >= 0")
    if halflife is not None and halflife <= 0:
        raise ValueError("halflife 必须为正数")
    if not isinstance(min_periods, int) or min_periods <= 0:
        raise ValueError("min_periods 必须为正整数")

    series = pd.Series(data)

    ewm_kwargs = {
        "adjust": adjust,
        "min_periods": min_periods,
    }
    if span is not None:
        ewm_kwargs["span"] = span
    if alpha is not None:
        ewm_kwargs["alpha"] = alpha
    if com is not None:
        ewm_kwargs["com"] = com
    if halflife is not None:
        ewm_kwargs["halflife"] = halflife

    smoothed = series.ewm(**ewm_kwargs).mean()

    if bias_correction and not adjust:
        if alpha is None:
            if span is not None:
                alpha = 2 / (span + 1)
            elif com is not None:
                alpha = 1 / (com + 1)
            elif halflife is not None:
                alpha = 1 - np.exp(-np.log(2) / halflife)
        n = np.arange(1, len(series) + 1)
        bias = 1 - (1 - alpha) ** n
        smoothed = smoothed / bias

    return smoothed


def ewma_smooth(data, **kwargs):
    """
    指数加权移动平均（EWMA），为 exponential_moving_average_smooth 的别名函数。

    Parameters
    ----------
    data : array-like
        输入的时间序列数据
    **kwargs
        传递给 exponential_moving_average_smooth 的其他参数

    Returns
    -------
    pd.Series
        平滑后的时间序列
    """
    return exponential_moving_average_smooth(data, **kwargs)


def smooth(data, method="simple", **kwargs):
    """
    统一平滑服务入口，支持多种平滑方法

    Parameters
    ----------
    data : array-like
        输入的时间序列数据
    method : str, optional
        平滑方法，可选值：
        - "simple" / "ma" / "sma": 简单移动平均
        - "weighted" / "wma": 加权移动平均
        - "exponential" / "ema" / "ewma": 指数加权移动平均
        默认为 "simple"
    **kwargs
        传递给对应平滑函数的参数，如 window_size, span, alpha 等

    Returns
    -------
    pd.Series
        平滑后的时间序列

    Raises
    ------
    ValueError
        如果 method 不是有效值
    """
    method_lower = method.lower()

    if method_lower in ("simple", "ma", "sma"):
        return moving_average_smooth(data, **kwargs)
    elif method_lower in ("weighted", "wma"):
        return weighted_moving_average_smooth(data, **kwargs)
    elif method_lower in ("exponential", "ema", "ewma"):
        return exponential_moving_average_smooth(data, **kwargs)
    else:
        raise ValueError(
            f"不支持的平滑方法: {method}。可选方法: simple/ma/sma, weighted/wma, exponential/ema/ewma"
        )


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

    print("=" * 60)
    print("EWMA / 指数加权移动平均 功能测试")
    print("-" * 60)

    ema_span = exponential_moving_average_smooth(data, span=5)
    print("EWMA (span=5，前5个)：", ema_span[:5].round(4).tolist())

    ema_alpha = exponential_moving_average_smooth(data, alpha=0.3)
    print("EWMA (alpha=0.3，前5个)：", ema_alpha[:5].round(4).tolist())

    ema_com = exponential_moving_average_smooth(data, com=2)
    print("EWMA (com=2，前5个)：", ema_com[:5].round(4).tolist())

    ema_halflife = exponential_moving_average_smooth(data, halflife=3)
    print("EWMA (halflife=3，前5个)：", ema_halflife[:5].round(4).tolist())
    print()

    ewma_adj = ewma_smooth([1, 2, 3, 4, 5], alpha=0.5, adjust=True)
    print("EWMA adjust=True (alpha=0.5, 数据=[1,2,3,4,5])：", ewma_adj.round(4).tolist())

    ewma_bias = ewma_smooth([1, 2, 3, 4, 5], alpha=0.5, adjust=False, bias_correction=True)
    print("EWMA with bias correction (alpha=0.5)：", ewma_bias.round(4).tolist())
    print()

    print("手动验证 alpha=0.5 bias correction:")
    for i in range(1, 6):
        raw = ewma_smooth([1,2,3,4,5], alpha=0.5, adjust=False)[i-1]
        bias = 1 - (0.5)**i
        corrected = raw / bias
        print(f"  第{i}个: raw={raw:.4f}, bias={bias:.4f}, corrected={corrected:.4f}")
    print()

    print("=" * 60)
    print("统一入口 smooth() 函数测试")
    print("-" * 60)
    test_data_simple = [1, 2, 3, 4, 5]
    print("测试数据：", test_data_simple)
    print()

    r1 = smooth(test_data_simple, method="sma", window_size=3)
    print("smooth method='sma', window_size=3:", r1.round(4).tolist())

    r2 = smooth(test_data_simple, method="WMA", window_size=3, weights=[1,2,3])
    print("smooth method='WMA', window_size=3:", r2.round(4).tolist())

    r3 = smooth(test_data_simple, method="EWMA", alpha=0.5)
    print("smooth method='EWMA', alpha=0.5:", r3.round(4).tolist())
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
