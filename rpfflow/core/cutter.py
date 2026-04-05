import numpy as np



def transition_probability(delta_g, T=3000):
    """
    计算从状态0到状态b的转移概率权重
    delta_g: 状态b与状态0的能量差 (E_b - E_0)
    T: 温度 (Kelvin)
    """
    kb = 8.617333e-5  # 玻尔兹曼常数 eV/K
    kt = kb * T

    # 如果是下坡反应 (delta_g <= 0)，转移概率极高，设为权重 1
    if delta_g <= 0:
        return 1.0
    else:
        # 如果是上坡反应，计算指数衰减
        # 为了防止 delta_g 过大导致数值溢出，可以做个截断
        prob = np.exp(-delta_g / kt)
        return prob