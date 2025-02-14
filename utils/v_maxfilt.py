import numpy as np

def v_maxfilt(x, f=1, n=None, d=None, x0=None):
    """
    V_MAXFILT - 计算指数加权滑动窗口的最大值 [Y, K, Y0] = (X, F, n, D, X0)

    用法:
        (1) y = v_maxfilt(x)   # 沿着第一个非单一维度进行最大值滤波
        (2) y = v_maxfilt(x, 0.95) # 使用遗忘因子 0.95（时间常数为 -1/log(0.95)=19.5 个样本）
        (3) 两种等效方法（即可以分块处理 x）：
                 y = v_maxfilt(np.hstack([u, v]))
                 yu, ku, x0 = v_maxfilt(u)
                 yv = v_maxfilt(v, f, n, d, x0)
                 y = np.hstack([yu, yv])

    输入:
        X: 输入数据的向量或矩阵
        F: 指数遗忘因子，范围从 0（非常健忘）到 1（不遗忘）
           F = exp(-1/T) 给出时间常数 T 个样本 [默认值 = 1]
        n: 滑动窗口的长度 [默认值 = Inf（相当于 []）]
        D: 沿着哪个维度进行计算 [默认值 = 第一个非单一维度]
        X0: 放置在 X 数据前面的初始值

    输出:
        Y: 输出矩阵 - 与 X 大小相同
        K: 索引数组：Y = X(K)。（注意，如果存在输入 X0，这些值可能 <= 0）
        Y0: 最后 nn-1 个值（用于初始化后续调用 v_maxfilt()）（如果 n=Inf，则为最后一个输出）

    该函数计算 y(p) = max(f^r * x(p-r), r=0:n-1)，其中 x(r) = -inf 对于 r < 1
    y = x(k) 在输出中

    示例：找到 x 中在 +-w 个样本内未被超过的所有峰值
    w = 4; m = 100; x = np.random.rand(m, 1)
    y, k = v_maxfilt(x, 1, 2*w+1)
    p = np.where((np.arange(1, m+1) - k) == w)[0]
    plt.plot(np.arange(1, m+1), x, '-', p-w, x[p-w], '+')
    """

    x = np.asarray(x)  # 确保输入是numpy数组
    input_is_1d = x.ndim == 1
    
    # 如果是1D输入，转换为2D列向量
    if input_is_1d:
        x = x.reshape(-1)  # 确保是1D
        s = (len(x),)
        d = 0
    else:
        s = x.shape
        if d is None:
            d = np.where(np.array(s) > 1)[0][0] if np.any(np.array(s) > 1) else 0
        elif d >= len(s):
            d = 0
    
    # 处理x0的情况
    if x0 is not None:
        if np.isscalar(x0):
            if input_is_1d:
                x0 = np.array([x0])
            else:
                x0_shape = list(s)
                x0_shape[d] = 1
                x0 = np.full(x0_shape, x0)
        y = np.concatenate((x0, x), axis=d)
        nx0 = 1 if np.isscalar(x0) else x0.shape[d]
    else:
        y = x
        nx0 = 0
    
    # 将数据移到第一个维度进行处理
    if not input_is_1d:
        y = np.moveaxis(y, d, 0)
    
    s = y.shape
    s1 = len(y) if input_is_1d else s[0]
    
    if n is None:
        n0 = np.inf
    else:
        n0 = max(n, 1)
    
    nn = n0
    if n0 < np.inf:
        ny0 = min(s1, nn - 1)
    else:
        ny0 = min(s1, 1)
    
    # 处理y0
    if ny0 <= 0 or n0 == np.inf:
        y0 = np.array([]) if input_is_1d else np.zeros(s[1:])
    else:
        y0 = y[-ny0:] if input_is_1d else y[-ny0:]
    
    # 初始化k
    k = np.arange(s1)
    if not input_is_1d:
        k = k.reshape(-1, *([1] * (len(s) - 1)))
        k = np.tile(k, [1] + list(s[1:]))
    
    if nn > 1:
        j = 1
        j2 = 1
        while j > 0:
            g = f ** j
            if input_is_1d:
                m = np.where(y[j:s1] <= g * y[:s1-j])[0]
            else:
                m = np.where(y[j:s1] <= g * y[:s1-j])[0]
            m = m + j * (m // (s1 - j))
            y[m + j] = g * y[m]
            k[m + j] = k[m]
            j2 += j
            j = min(j2, nn - j2)
    
    # 处理输出
    if nx0 > 0:
        y = y[nx0:]
        k = k[nx0:] - nx0
    
    # 如果不是1D输入，需要还原维度
    if not input_is_1d:
        y = np.moveaxis(y, 0, d)
        k = np.moveaxis(k, 0, d)
        if ny0 > 0:
            y0 = np.moveaxis(y0, 0, d)
    
    return y, k, y0

if __name__ == "__main__":
    x = np.random.rand(10800)  # 一维数组
    y, k, y0 = v_maxfilt(x, f=0.95, n=10, d=1, x0=-np.inf)
    print(y.shape)  # 应该是(10800,)
    print(k.shape)  # 应该是(10800,)
    print(y0.shape)  # 应该是(9,) 因为n=10