import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

def outer_product_(u, v):
    m = np.zeros((u.shap[0], v.shape[0]))
    for i in range(u.shape[0]):
        for j in range(v.shape[0]):
            m[i,j] = u[i]*v[j]
    return m

def add_(u,v):
    if u.shape != v.shape[0]:
        ValueError('input vectors are not in the same size')
    m = np.zeros(u.shape[0])
    for i in range(u.shape[0]):
        m[i] = u[i] + v[i]
    return m

def qft(x):
    return x

def ft_plot(ft_x, x, t, ft_type):
    amp_x = abs(ft_x)/len(x)*2 # 纵坐标变换
    amp = amp_x[0:int(len(x)/2)] # 选取前半段计算结果
    label_x = np.arange(int(len(x)/2)) # 生成频率坐标
    fs = 1/(t[2]-t[1]) # 计算采样频率
    fre = label_x /len(x)*fs # 频率坐标变换
    pha = np.unwrap(np.angle(ft_x)) # 计算相位角并去除2pi跃变
    
    if not os.path.exists('figures'):
        os.mkdir('figures')
    plt.figure()
    plt.plot(t,x)
    plt.title('Time Signal')
    plt.xlabel('Frequence / Hz')
    plt.ylabel('Amplitute / a.u.')
    plt.savefig(os.path.join('figures', 'Time_Signal.png'))
    plt.close()

    plt.figure()
    plt.plot(fre,amp)
    plt.title('{} Frequence Signal'.format(ft_type))
    plt.xlabel('Frequence / Hz')
    plt.ylabel('Amplitute / a.u.')
    plt.savefig(os.path.join('figures', '{}_Frequence_Signal.png'.format(ft_type)))
    plt.close()

def demo(n):
    N = 2**n
    # generate sine data
    t = np.linspace(0, 5*np.pi, N)
    x = np.sin(2*np.pi*t)
    # traditional fft
    fft_x = fft(x)
    # myQFT
    qft_x = qft(x)    
    # plot
    ft_plot(fft_x, x, t, 'FFT')
    ft_plot(qft_x, x, t, 'QFT')

if __name__ == '__main__':
    demo(n=8)