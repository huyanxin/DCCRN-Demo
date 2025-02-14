import numpy as np
from scipy.signal import lfilter, cheby1, cheby2, ellip, bilinear, freqz
from utils.v_maxfilt import v_maxfilt
from utils.v_stdspectrum import v_stdspectrum

# from v_maxfilt import v_maxfilt
# from v_stdspectrum import v_stdspectrum

def v_activlev(sp, fs, mode=' '):
    """
    测量语音活动水平，基于ITU-T P.56标准
    输入:
        sp: 语音信号 (信噪比应大于20dB)
        fs: 采样频率 (Hz)
        mode: 模式字符串，包含以下选项的组合:
            0 - 完全省略高通滤波器 (即包含DC)
            3 - 高通滤波器截止频率为30 Hz (允许通过电源噪声)
            4 - 高通滤波器截止频率为40 Hz (允许通过电源噪声)
            1 - 使用Chebyshev 1滤波器
            2 - 使用Chebyshev 2滤波器 (默认)
            e - 使用椭圆滤波器
            h - 省略低通滤波器 (5.5, 12或18 kHz)
            w - 使用宽带滤波器频率范围: 70 Hz 到 12 kHz
            W - 使用超宽带滤波器频率范围: 30 Hz 到 18 kHz
            d - 输出单位为dB而不是功率
            n - 输出归一化的语音信号作为第一个输出
            N - 输出归一化的滤波后语音信号作为第一个输出
            l - 输出活动功率和长期功率水平
            a - 包含A加权滤波器
            i - 包含ITU-R-BS.468/ITU-T-J.16加权滤波器
            z - 不进行0.35秒的零填充
    输出:
        lev: 语音活动水平 (功率单位或dB，取决于mode)
        af: 活动因子 (占空比)
        fso: 中间信息结构体，允许分块处理语音信号
        vad: 与sp长度相同的布尔向量，作为近似语音活动检测器
    """
    # 初始化变量
    nbin = 20  # 60 dB范围，每个bin 3dB
    thresh = 15.9  # 阈值 (dB)

    # 高通滤波器的s域零点和极点
    c25zp = np.array([[0.37843443673309j, 0.23388534441447j],
                      [-0.20640255179496 + 0.73942185906851j, -0.54036889596392 + 0.45698784092898j]])
    c25zp = np.hstack((np.array([[0], [-0.66793268833792]]), c25zp, np.conj(c25zp)))

    c15zp = np.array([[-0.659002835294875 + 1.195798636925079j, -0.123261821596263 + 0.947463030958881j]])
    c15zp = np.hstack((np.zeros((1, 5)), np.array([[-2.288586431066945]]), c15zp, np.conj(c15zp)))

    e5zp = np.array([[0.406667680649209j, 0.613849362744881j],
                     [-0.538736390607201 + 1.130245082677107j, -0.092723126159100 + 0.958193646330194j]])
    e5zp = np.hstack((np.array([[0], [-1.964538608244084]]), e5zp, np.conj(e5zp)))

    if not isinstance(fs, dict):  # 如果没有状态向量
        fso = {}
        fso['ffs'] = fs  # 采样频率
        ti = 1 / fs
        g = np.exp(-ti / 0.03)  # 包络滤波器的极点位置
        fso['ae'] = np.array([1, -2 * g, g ** 2]) / (1 - g) ** 2  # 包络滤波器系数 (DC增益 = 1)
        fso['ze'] = np.zeros(2)
        fso['nh'] = int(np.ceil(0.2 / ti)) + 1  # 挂起时间 (样本数)
        fso['zx'] = -np.inf  # v_maxfilt()的初始值
        fso['emax'] = -np.inf  # 最大指数
        fso['ns'] = 0
        fso['ssq'] = 0
        fso['ss'] = 0
        fso['sf'] = 1  # 形成能量直方图时的比例因子
        fso['sfdb'] = 0  # dB比例因子
        fso['kc'] = np.zeros(nbin)  # 累积占用计数

        # 选择滤波器类型
        if '1' in mode:
            szp = c15zp  # Chebyshev 1
        elif 'e' in mode:
            szp = e5zp  # Elliptic
        else:
            szp = c25zp  # Chebyshev 2

        flh = [200, 5500]  # 默认频率范围 +- 0.25 dB
        if 'w' in mode:
            flh = [70, 12000]  # 超宽带 (Annex B of [2])
        elif 'W' in mode:
            flh = [30, 18000]  # 全频带 (Annex C of [2])

        if '3' in mode:
            flh[0] = 30  # 强制30 Hz HPF截止频率
        if '4' in mode:
            flh[0] = 40  # 强制40 Hz HPF截止频率

        if 'r' in mode:  # 向后兼容
            mode = '0h' + mode  # 取消两个滤波器
        elif fs < flh[1] * 2.2:
            mode = 'h' + mode  # 在低采样率下取消低通滤波器

        fso['fmd'] = mode  # 保存模式标志

        if '0' not in mode:  # 实现HPF为双二阶滤波器以避免舍入误差
            zl = 2 / (1 - szp * np.tan(flh[0] * np.pi / fs)) - 1  # 使用双线性变换转换s域极点/零点
            abl = np.hstack((np.ones((2, 1)), -zl[:, 0:1], -2 * np.real(zl[:, 1:3]), np.abs(zl[:, 1:3]) ** 2))  # 双二阶系数
            hfg = (abl @ np.array([1, -1, 0, 0, 0, 0]).T) * (abl @ np.array([1, 0, -1, 0, 1, 0]).T) * (abl @ np.array([1, 0, 0, -1, 0, 1]).T)
            abl = abl[:, [0, 1, 0, 2, 4, 0, 3, 5]]  # 重新排序为双二阶
            abl[0, 0:2] = abl[0, 0:2] * hfg[1] / hfg[0]  # 强制Nyquist增益等于1
            fso['abl'] = abl
            fso['zl'] = np.zeros((5,1))  # 确保是2维数组

        if 'h' not in mode:
            zh = 2 / (szp / np.tan(flh[1] * np.pi / fs) - 1) + 1  # 使用双线性变换转换s域极点/零点
            ah = np.real(np.poly(zh[1, :]))
            bh = np.real(np.poly(zh[0, :]))
            fso['bh'] = bh * np.sum(ah) / np.sum(bh)
            fso['ah'] = ah
            fso['zh'] = np.zeros(5)

        if 'a' in mode or 'i' in mode:
            fso['bw'], fso['aw'] = v_stdspectrum(2, 'z', fs) if 'a' in mode else v_stdspectrum(8, 'z', fs)
            fso['zw'] = np.zeros(max(len(fso['bw']), len(fso['aw'])) - 1)
    else:
        fso = fs  # 使用现有结构体

    md = fso['fmd']  # md用于确定所有选项，除了'z'使用mode
    nsp = len(sp)  # 原始语音长度
    if 'z' not in mode:
        nz = int(np.ceil(0.35 * fso['ffs']))  # 要附加的零的数量
        sp = np.hstack((sp, np.zeros(nz)))
    else:
        nz = 0

    ns = len(sp)
    if ns:  # 处理此语音块
        # 应用输入滤波器到语音
        if '0' not in md:  # 实现HPF为双二阶滤波器以避免舍入误差
            sq, fso['zl'][0] = lfilter(fso['abl'][0, 0:2], fso['abl'][1, 0:2], sp, zi=fso['zl'][0])  # 高通滤波器: 实极点/零点
            sq, fso['zl'][1:3] = lfilter(fso['abl'][0, 2:5], fso['abl'][1, 2:5], sq, zi=fso['zl'][1:3])  # 高通滤波器: 双二阶1
            sq, fso['zl'][3:5] = lfilter(fso['abl'][0, 5:8], fso['abl'][1, 5:8], sq, zi=fso['zl'][3:5])  # 高通滤波器: 双二阶2
        else:
            sq = sp

        if 'h' not in md:
            sq, fso['zh'] = lfilter(fso['bh'], fso['ah'], sq, zi=fso['zh'])  # 低通滤波器

        if 'a' in md or 'i' in md:
            sq, fso['zw'] = lfilter(fso['bw'], fso['aw'], sq, zi=fso['zw'])  # 加权滤波器

        fso['ns'] += ns  # 计数语音样本数
        fso['ss'] += np.sum(sq)  # 语音样本的总和 (内部未使用，但在fso输出中可用)
        ssq = np.sum(sq ** 2)  # 新语音样本的平方和
        if ssq > 0 and fso['ssq'] == 0:  # 如果这些是第一个非零语音样本
            fso['sf'] = ns / ssq  # 比例因子，将此块的均方功率归一化为1
            fso['sfdb'] = 10 * np.log10(fso['sf'])  # dB比例因子

        fso['ssq'] += ssq  # 所有语音样本的平方和
        s, fso['ze'] = lfilter([1], fso['ae'], np.abs(sq), zi=fso['ze'])  # 包络滤波器

        # 使用比例因子fso.sf确保当sp乘以常数时直方图分箱不变
        qf, qe = np.frexp(fso['sf'] * s ** 2)  # 取高效的log2函数，2^qe是bin的上限
        qe = qe.astype(float)  # 将qe转换为浮点数类型
        qe[qf == 0] = -np.inf  # 修正零值
        qe, qk, fso['zx'] = v_maxfilt(qe, 1, fso['nh'], 1, fso['zx'])  # 应用0.2秒挂起
        oemax = fso['emax']  # 前一个qe+1的最大值
        fso['emax'] = max(oemax, np.max(qe) + 1)  # 更新qe+1的最大值

        if fso['emax'] == -np.inf:  # 如果所有样本都是零
            fso['kc'][0] += ns
        else:  # 如果有非零样本
            qe = np.minimum(fso['emax'] - qe, nbin)  # 强制在1:nbin范围内
            qe = qe.astype(np.int64)  # 将qe转换为整数类型
            wqe = np.ones(len(qe))
            kc = np.cumsum(np.bincount(qe, wqe, minlength=nbin))  # 累积占用计数
            esh = fso['emax'] - oemax  # 向下移动前一个bin计数的量
            if esh < nbin - 1:  # 如果前一个bin值得保留
                kc[esh + 1:nbin - 1] += fso['kc'][0:nbin - esh - 1]
                kc[nbin - 1] += np.sum(fso['kc'][nbin - esh - 1:nbin])
            else:
                kc[nbin - 1] += np.sum(fso['kc'])  # 否则将所有旧计数加到最后一个 (最低) bin
            fso['kc'] = kc

    if fso['ns'] > nz:  # 现在计算输出值
        if fso['ssq'] > 0:
            kc = fso['kc']

            # 避免除以零的情况
            kc[kc == 0] = 1e-10
            aj = 10 * np.log10(fso['ssq'] / kc)  # 计算活动水平

            aj = 10 * np.log10(fso['ssq'] / kc)  # 计算活动水平
            cj = 10 * np.log10(2) * (fso['emax'] - np.arange(0, nbin + 1) - 1) - fso['sfdb']  # bin下限
            mj = aj - cj - thresh
            jj = np.where((mj[:-1] < 0) & (mj[1:] >= 0))[0]  # 找到通过阈值的正过渡
            
            if len(jj) == 0:  # 如果从未跨越阈值
                if mj[-1] <= 0:  # 如果最终低于
                    jj = len(mj) - 1  # 取阈值为最后一个bin的底部
                    jf = 1
                else:  # 如果始终高于
                    jj = 0  # 取阈值为第一个bin的底部
                    jf = 0
            else:
                jj = jj[0]  # 取第一个过渡点
                jf = 1 / (1 - mj[jj + 1] / mj[jj])  # 线性插值
            
            lev = aj[jj] + jf * (aj[jj + 1] - aj[jj])  # 活动水平 (dB)
            lp = 10 ** (lev / 10)  # 活动水平 (功率)
            if 'd' in md:  # 'd'选项 -> 输出单位为dB
                lev = np.array([lev, 10 * np.log10(fso['ssq'] / fso['ns'])])
            else:  # ~'d'选项 -> 输出单位为功率
                lev = np.array([lp, fso['ssq'] / fso['ns']])

            af = fso['ssq'] / ((fso['ns'] - nz) * lp)
        else:  # 如果所有样本都为零
            af = 0
            if 'd' in md:  # 'd'选项 -> 输出单位为dB
                lev = np.array([-np.inf, -np.inf])  # 活动水平为0 dB
            else:  # ~'d'选项 -> 输出单位为功率
                lev = np.array([0, 0])  # 活动水平为0功率

        if 'l' not in md:
            lev = lev[0]  # 除非'l'选项，否则只输出lev的第一个元素

        # 计算VAD
        vad = v_maxfilt(s[:nsp], 1, fso['nh'], 1)
        vad = vad[0] > (np.sqrt(lp) / 10 ** (thresh / 20))
        return lev, af, fso, vad

    # 如果没有足够的样本
    vad = v_maxfilt(s, 1, fso['nh'], 1)
    vad = vad > (np.sqrt(lp) / 10 ** (thresh / 20))
    levdb = 10 * np.log10(lp)
    return levdb, af, fso, vad

if __name__ == '__main__':
    fs = 16000
    t = np.arange(0, 1, 1/fs)
    sp = np.sin(2 * np.pi * 1000 * t) + 0.1 * np.random.randn(len(t))
    lev, af, fso, vad = v_activlev(sp, fs, '0')  # 添加 mode='0' 参数
    print(vad)

    '''
    # 示例
    fs = 8000
    t = np.arange(0, 1, 1 / fs)
    sp = np.sin(2 * np.pi * 1000 * t) + 0.1 * np.random.randn(len(t))
    lev = v_activlev(sp, fs)
    print(f'活动水平 = {lev:.1f} dB (ITU-T P.56)')
    '''