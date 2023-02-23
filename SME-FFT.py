import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from cmath import sin, cos, pi

'''
SME基本运算
1. 外积: outer_product_
2. 累加: add_
3. 基于这两个运算, 我们可以很容易定义出内积: dot_
'''
def add_(u, v):
    '''
    parameters:
    u: list
        first vector
    v: list
        second vector
    returns:
    m: list
        the sum of u and v
    '''
    m = [u[i] + v[i] for i in range(len(u))]
    return m

def outer_(u, v):
    '''
    parameters:
    u: list
        first vector
    v: list
        second vector
    returns:
    m: 2D list
        the outer product of u and v
    '''
    m = []
    for i in range(len(u)):
        temp = [u[i] * v[j] for j in range(len(v))]
        m.append(temp)
    return m

def dot_(u, v):
    '''
    parameters:
    u: list
        first vector
    v: list
        second vector
    returns:
    m: list, whose length is 1
        the cross product of u and v
    '''
    m = [0]
    for i in range(len(u)):
        temp = outer_([u[i]], v)[i]
        m = add_(m, temp)
    return m


class SME_FFT():
    """
    SME_FFT
    adapted from https://www.jianshu.com/p/0bd1ddae41c4

    Here, we use the original FFT based on the Butterfly Algorithm, which can be learned from or this blog https://blog.csdn.net/m0_38139533/article/details/100942095 for easier understanding.
    
    In our code, we follow the logic of the Butterfly Algrithm, but change the basic operation(add, multiply/cross product) based on SME framework, which only support outer product and add between vectors.
    """
    def __init__(self, _list=[], N=0):
        '''
        _list 是传入的待计算的离散序列，N是序列采样点数，对于本方法，点数必须是2^n才可以得到正确结果
        '''
        self.list = _list  # 初始化数据
        self.N = N
        self.total_m = 0  # 序列的总层数
        self._reverse_list = []  # 位倒序列表
        self.output =  []  # 计算结果存储列表
        self._W = []  # 系数因子列表
        for _ in range(len(self.list)):
            self._reverse_list.append(self.list[self._reverse_pos(_)])
        self.output = self._reverse_list.copy()
        for _ in range(self.N):
            self._W.append((cos(2 * pi / N) - sin(2 * pi / N) * 1j) ** _)  # 提前计算W值，降低算法复杂度

    def _reverse_pos(self, num) -> int:
        '''
        得到位倒序后的索引
        '''
        out = 0
        bits = 0
        _i = self.N
        data = num
        while (_i != 0):
            _i = _i // 2
            bits += 1
        for i in range(bits - 1):
            out = out << 1
            out |= (data >> i) & 1
        self.total_m = bits - 1
        return out

    def FFT(self, _list, N, abs=True) -> list:
        '''
        计算给定序列的傅里叶变换结果，返回一个列表，结果是没有经过归一化处理的
        abs=True 表示输出结果是否取得绝对值
        '''
        self.__init__(_list, N)
        for m in range(self.total_m):
            _split = self.N // 2 ** (m + 1)
            num_each = self.N // _split
            for k in range(_split):
                for l in range(num_each // 2):
                    # temp = self.output[k * num_each + l]
                    temp = self.output[add_(dot_([k],[num_each]), [l])[0]]
                    # temp2 = self.output[k * num_each + l + num_each // 2] * self._W[l * 2 ** (self.total_m - m - 1)]
                    temp2 = dot_([self.output[add_(add_(dot_([k], [num_each]), [l]), [num_each // 2])[0]]], [self._W[dot_([l], [2**(self.total_m - m - 1)])[0]]])[0]
                    # self.output[k * num_each + l] = (temp + temp2)
                    self.output[add_(dot_([k],[num_each]), [l])[0]] = (add_([temp], [temp2])[0])
                    # self.output[k * num_each + l + num_each // 2] = (temp - temp2)
                    self.output[add_(add_(dot_([k], [num_each]), [l]), [num_each // 2])[0]] = (add_([temp], [-temp2])[0])
        if abs == True:
            for k in range(len(self.output)):
                self.output[k] = self.output[k].__abs__()
        return self.output

    def FFT_normalized(self, _list, N) -> list:
        '''
        计算给定序列的傅里叶变换结果，返回一个列表，结果经过归一化处理
        '''
        self.FFT(_list, N)
        max = 0   # 存储元素最大值
        for k in range(len(self.output)):
            if max < self.output[k]:
                max = self.output[k]
        for k in range(len(self.output)):
            self.output[k] /= max
        return self.output

    def IFFT(self, _list, N) -> list:
        '''
        计算给定序列的傅里叶逆变换结果，返回一个列表
        '''
        self.__init__(_list, N)
        for k in range(self.N):
            self._W[k] = (cos(2 * pi / N) - sin(2 * pi / N) * 1j) ** (-k)
        for m in range(self.total_m):
            _split = self.N // 2 ** (m + 1)
            num_each = self.N // _split
            for k in range(_split):
                for l in range(num_each // 2):
                    temp = self.output[k * num_each + l]
                    temp2 = self.output[k * num_each + l + num_each // 2] * self._W[l * 2 ** (self.total_m - m - 1)]
                    self.output[k * num_each + l] = (temp + temp2)
                    self.output[k * num_each + l + num_each // 2] = (temp - temp2)
        for k in range(self.N):  # 根据IFFT计算公式对所有计算列表中的元素进行*1/N的操作
            self.output[k] /= self.N
            self.output[k] = self.output[k].__abs__()
        return self.output

    def DFT(self, _list, N) -> list:
        '''
        计算给定序列的离散傅里叶变换结果，算法复杂度较大，返回一个列表，结果没有经过归一化处理
        '''
        self.__init__(_list, N)
        origin = self.list.copy()
        for i in range(self.N):
            temp = 0
            for j in range(self.N):
                temp += origin[j] * (((cos(2 * pi / self.N) - sin(2 * pi / self.N) * 1j)) ** (i * j))
            self.output[i] = temp.__abs__()
        return self.output

def ft_plot(ft_x, x, t, ft_type):
    '''
    plot the result of Fourier Transform X and the raw data x
    
    parameters:
    ft_x: list or array
        the Fourier transform X from x by a certain FT method
    x: array
        the raw data x
    t: array
        the indice of the raw data x
    ft_type: str
        the name of FT method
    '''
    if type(ft_x) == list:
        ft_x = np.array(ft_x)
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

def demo(x):
    '''
    Compute the Fast Fourier Transform by our SME method, and compare the result with standard FFT method included in scipy module
    
    x: list
        the raw data used for Fourier Transform
    '''
    x = np.array(x)
    N = x.shape[0]
    t = np.arange(N)

    # traditional fft
    fft_x = fft(x)
    # SME FFT
    sme_x = SME_FFT().FFT(x, N, False)    
    # plot
    ft_plot(fft_x, x, t, 'FFT')
    ft_plot(sme_x, x, t, 'SME')

if __name__ == '__main__':
    # generate raw data
    # x = [1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0]
    N = 256
    t = np.linspace(0, 5*np.pi, N)
    x = np.sin(2*np.pi*t)
    # compute the FFT result by SME and standard FFT
    demo(x)