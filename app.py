import streamlit as st
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 设置网页标题和布局
st.set_page_config(page_title="Hodgkin-Huxley 模型仿真器", layout="wide")
st.title("🧠 Hodgkin-Huxley (HH) 模型神经元仿真器")
st.markdown("通过左侧面板调整参数，实时观察神经元膜电位和离子通道门控变量的变化。")

# ================= 侧边栏：参数输入区域 =================
st.sidebar.header("⚙️ 仿真参数设置")

st.sidebar.subheader("1. 刺激电流 (External Current)")
I_amp = st.sidebar.slider("电流强度 (μA/cm²)", min_value=0.0, max_value=50.0, value=10.0, step=0.5)
t_start = st.sidebar.number_input("刺激开始时间 (ms)", value=10.0)
t_end = st.sidebar.number_input("刺激结束时间 (ms)", value=40.0)

st.sidebar.subheader("2. 离子通道最大电导 (mS/cm²)")
g_Na = st.sidebar.number_input("钠离子电导 (g_Na)", value=120.0)
g_K = st.sidebar.number_input("钾离子电导 (g_K)", value=36.0)
g_L = st.sidebar.number_input("漏电流电导 (g_L)", value=0.3)

st.sidebar.subheader("3. 翻转电位 (Reversal Potentials, mV)")
E_Na = st.sidebar.number_input("钠离子翻转电位 (E_Na)", value=50.0)
E_K = st.sidebar.number_input("钾离子翻转电位 (E_K)", value=-77.0)
E_L = st.sidebar.number_input("漏电流翻转电位 (E_L)", value=-54.38)

st.sidebar.subheader("4. 细胞膜参数")
C_m = st.sidebar.number_input("膜电容 (C_m, μF/cm²)", value=1.0)
t_max = st.sidebar.number_input("总仿真时间 (ms)", value=100.0)

# ================= 数学模型：HH 方程 =================
# 定义刺激电流函数
def I_ext(t):
    if t_start <= t <= t_end:
        return I_amp
    return 0.0

# 门控变量的速率函数 (使用现代的静息电位约定，约 -65 mV)
# 为了防止分母为0，加入一个微小的常数 1e-5
def alpha_m(V):
    vt = V + 40.0
    return np.where(np.abs(vt) < 1e-5, 1.0, 0.1 * vt / (1.0 - np.exp(-vt / 10.0)))

def beta_m(V):
    return 4.0 * np.exp(-(V + 65.0) / 18.0)

def alpha_h(V):
    return 0.07 * np.exp(-(V + 65.0) / 20.0)

def beta_h(V):
    return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

def alpha_n(V):
    vt = V + 55.0
    return np.where(np.abs(vt) < 1e-5, 0.1, 0.01 * vt / (1.0 - np.exp(-vt / 10.0)))

def beta_n(V):
    return 0.125 * np.exp(-(V + 65.0) / 80.0)

# 定义微分方程组
def hh_derivatives(Y, t):
    V, m, h, n = Y
    
    # 计算各项离子电流
    I_Na = g_Na * (m**3) * h * (V - E_Na)
    I_K = g_K * (n**4) * (V - E_K)
    I_Leak = g_L * (V - E_L)
    
    # 膜电位变化率 dV/dt
    dVdt = (I_ext(t) - I_Na - I_K - I_Leak) / C_m
    
    # 门控变量变化率 dm/dt, dh/dt, dn/dt
    dmdt = alpha_m(V) * (1.0 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1.0 - h) - beta_h(V) * h
    dndt = alpha_n(V) * (1.0 - n) - beta_n(V) * n
    
    return[dVdt, dmdt, dhdt, dndt]

# ================= 求解器与仿真执行 =================
# 时间向量
t = np.linspace(0, t_max, 10000)

# 初始条件 (系统在静息电位时的稳态值)
V0 = -65.0
m0 = alpha_m(V0) / (alpha_m(V0) + beta_m(V0))
h0 = alpha_h(V0) / (alpha_h(V0) + beta_h(V0))
n0 = alpha_n(V0) / (alpha_n(V0) + beta_n(V0))
Y0 =[V0, m0, h0, n0]

# 使用 scipy.odeint 求解常微分方程
sol = odeint(hh_derivatives, Y0, t)
V = sol[:, 0]
m = sol[:, 1]
h = sol[:, 2]
n = sol[:, 3]

# 获取注入电流随时间的变化（用于绘图）
I_inj = [I_ext(ti) for ti in t]

# ================= 绘图与界面输出 =================
col1, col2 = st.columns(2)

with col1:
    st.subheader("⚡ 膜电位与刺激电流")
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # 膜电位图
    ax1.plot(t, V, color='r', lw=2)
    ax1.set_ylabel('Membrane Potential (mV)')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Action Potential')
    
    # 电流图
    ax2.plot(t, I_inj, color='k', lw=2)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Current (μA/cm²)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig1)

with col2:
    st.subheader("🚪 离子通道门控变量 (m, h, n)")
    fig2, ax3 = plt.subplots(figsize=(8, 6))
    
    ax3.plot(t, m, label='m (Na+ Activation)', color='blue')
    ax3.plot(t, h, label='h (Na+ Inactivation)', color='green')
    ax3.plot(t, n, label='n (K+ Activation)', color='orange')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Probability')
    ax3.set_title('Gating Variables')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig2)

st.success("仿真成功！试着在左侧修改参数，比如将电流强度设为 2.0（看是否发生动作电位），或者将钠离子电导 g_Na 降为 0（模拟河豚毒素 TTX 的阻断效果）。")