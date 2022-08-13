import imp
from brian2 import *
import os
import os.path as osp
import matplotlib.pyplot as plt

def IO_curves(type="LIF"):
    n = 1000
    duration = 1*second
    tau = 10*ms
    if type == 'LIF':
        eqs = '''
        dv/dt = (v0 - v) / tau : volt (unless refractory)
        v0 : volt
        '''
    elif type == 'QIF': # don't know why it didn't work.
        eqs = '''
        dv/dt = (v0 + v**2) / tau : volt (unless refractory)
        v0 : volt
        '''
        
    group = NeuronGroup(n, eqs, threshold='v > 10*mV', reset='v = 0*mV',
                        refractory=5*ms, method='exact')
    group.v = 0*mV
    group.v0 = '20*mV * i / (n-1)'

    monitor = SpikeMonitor(group)

    run(duration)
    plt.plot(group.v0/mV, monitor.count / duration)
    plt.xlabel('v0 (mV)')
    plt.ylabel('Firing rate (sp/s)')
    # plt.show()
    save_path = './figures' + '/Kameron_Harris'
    if not osp.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(osp.join(save_path, 'IO_curve_{}'.format('LIF') + '.png'))
    plt.close()
    
if __name__ == '__main__':
    IO_curves(type='QIF')