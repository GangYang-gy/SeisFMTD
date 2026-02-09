# -------------------------------------------------------------------
# Paste the synthetic waveform (syn) on the data.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# Modified By: Gang Yang (gangy.yang@mail.utoronto.ca)
# -------------------------------------------------------------------


from pyCAPLunar.DMisfit import DMisfit
import numpy as np


def DPaste(data, n_t_data, n_data_lens, syn, n_t_syn, n_syn_lens, n_max_shift, type='Zhu1996'):
    '''
    Paste the synthetic waveform on the data,
    The syn is sliced into phases while the data keeps the original traces.

    ****
    * The data are expected to be longer than the syn.
    ****

    :param data:        The data trace. 1C.
    :param n_t_data:    The index indicating the approximate arrival time of a phase. INT
    :param n_data_lens: The length of the data. INT.
    :param syn:         The synthetic waveform of the phase. 1C.
    :param n_t_syn:     The approximate arrival time of the phase on syn. waveform. INT.
    :param n_syn_lens   The length of the synthetic wave. INT.
    :param n_max_shift: The maximum allowing misfit. INT.
    :return:
            n_shift: the time shift.
            misfit: the misfit.
    '''
    #print(n_t_data, n_data_lens, n_t_syn, n_syn_lens, n_max_shift)
    raw_data = data.copy()
    n_max_shift_2 = int(2.0 * n_max_shift)+1
    # compute tiem shift
    i_shift = 0
    _var_cc = 0
    max_cc = 0
    for i in range(n_max_shift_2):
        # idx on the data
        idx = n_t_data - n_t_syn - n_max_shift + i
        if idx + n_syn_lens <= n_data_lens:
            if idx < 0:
                # if window is too short, there may be error
                _var_cc = np.dot(syn, np.hstack([np.zeros((-1 * idx)), data[:idx + n_syn_lens]]))
            else:
                _var_cc = np.dot(syn, data[idx:idx + n_syn_lens])
        else:
            if idx < 0:
                _var_cc = np.dot(syn, np.hstack([np.zeros(-1 * idx), data, np.zeros(n_syn_lens + idx - n_data_lens)]))
            else:
                _var_cc = np.dot(syn, np.hstack([data[idx:], np.zeros(n_syn_lens - (n_data_lens - idx))]))
        if 0 == i:
            max_cc = _var_cc
        else:
            if max_cc < _var_cc:
                max_cc = _var_cc
                i_shift = i
    n_shift = i_shift - n_max_shift
    plot = False
    # calculate the waveform misfit and cc.
    n0_idx = n_t_data - n_t_syn + n_shift
    if n0_idx >= 0:
        if n0_idx + n_syn_lens > n_data_lens:           
            tmp = data[n0_idx:]
            misfit = DMisfit(syn, np.hstack([tmp, np.zeros(n_syn_lens-len(tmp))]),raw_data, type = type)
            cc = np.corrcoef(syn, np.hstack([tmp, np.zeros(n_syn_lens-len(tmp))]))[0][1]  
            if plot: 
                import matplotlib.pyplot as plt
                fig,ax=plt.subplots(1)
                ax.plot(syn)
                ax.plot(np.hstack([tmp, np.zeros(n_syn_lens-len(tmp))]),'--')                 
                plt.savefig('1.png')  
                exit()  
            
        else:
            misfit = DMisfit(syn, data[n0_idx:n0_idx+n_syn_lens],raw_data, type = type)
            cc = np.corrcoef(syn, data[n0_idx:n0_idx+n_syn_lens])[0][1]
            if plot:
                import matplotlib.pyplot as plt
                fig,ax=plt.subplots(1)
                ax.plot(syn)
                ax.plot(data[n0_idx:n0_idx+n_syn_lens],'--')                 
                plt.savefig('1.png') 
                exit()   
            
    else:
        if n0_idx + n_syn_lens > n_data_lens:
            tmp = np.hstack([np.zeros(-1 * n0_idx), data])
            misfit = DMisfit(syn, np.hstack([tmp, np.zeros(n_syn_lens - len(tmp))]),raw_data, type = type)
            cc = np.corrcoef(syn, np.hstack([tmp, np.zeros(n_syn_lens - len(tmp))]))[0][1]
            if plot:
                import matplotlib.pyplot as plt
                fig,ax=plt.subplots(1)
                ax.plot(syn)
                ax.plot(np.hstack([tmp, np.zeros(n_syn_lens - len(tmp))]),'--')                 
                plt.savefig('1.png')  
                exit()  
            
        else:
            misfit = DMisfit(syn, np.hstack([np.zeros(-1 * n0_idx), data[:n_syn_lens + n0_idx]]),raw_data, type = type)
            cc = np.corrcoef(syn, np.hstack([np.zeros(-1 * n0_idx), data[:n_syn_lens + n0_idx]]))[0][1]
            if plot:
                import matplotlib.pyplot as plt
                fig,ax=plt.subplots(1)
                ax.plot(syn)
                ax.plot(np.hstack([np.zeros(-1 * n0_idx), data[:n_syn_lens + n0_idx]]),'--')                 
                plt.savefig('1.png',dpi=30)
                exit()    
            

    return int(n_shift), misfit, cc



def DPaste_shift_syn(data, n_data_lens, syn, n_max_shift, type='Zhu1996'):
    """
    Shift the synthetic waveform (syn) to align with the observed data.
    Returns the shift amount, misfit, cc, and the shifted syn.
    """

    raw_data = data.copy()
    n_max_shift_2 = int(2.0 * n_max_shift) + 1

    i_shift = 0
    max_cc = -1e10

    for i in range(n_max_shift_2):
        n_shift = i - n_max_shift
        # Pad syn with zeros before/after to obtain the shifted syn
        if n_shift >= 0:
            syn_shifted = np.hstack([np.zeros(n_shift), syn])
            syn_shifted = syn_shifted[:n_data_lens]   # Keep aligned with data length
        else:
            syn_shifted = syn[-n_shift:]              # Left shift: remove first |n_shift| points
            syn_shifted = np.hstack([syn_shifted, np.zeros(-n_shift)])
            syn_shifted = syn_shifted[:n_data_lens]

        # Compute cross-correlation (using dot product here)
        _var_cc = np.dot(syn_shifted, data[:len(syn_shifted)])

        if _var_cc > max_cc:
            max_cc = _var_cc
            i_shift = n_shift
            best_syn = syn_shifted.copy()

    # Compute misfit and cc at optimal alignment
    misfit = DMisfit(best_syn, data[:len(best_syn)], raw_data, type=type)
    cc = np.corrcoef(best_syn, data[:len(best_syn)])[0][1]

    return best_syn, misfit, cc


