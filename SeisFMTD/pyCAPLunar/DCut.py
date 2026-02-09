# -------------------------------------------------------------------
# Cut phase segments from SGT data.
# Paste the synthetic waveform on the data. Cut the SGT only.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# Modified By: Gang Yang (gangy.yang@mail.utoronto.ca)
# -------------------------------------------------------------------


import numpy as np

def DCut_sgt_pnl(sgt, n_tp, n_length):
  
    #return sgt[n_tp - int(0.4*n_length): n_tp+n_length,...]
    return sgt[n_tp: n_tp+n_length,...]
    

def DCut_sgt_srf(sgt, n_ts, n_length):
  
    #return sgt[n_ts - int(0.3*n_length): n_ts+n_length,...]
    return sgt[n_ts: n_ts+n_length,...]
  
def DCut_data_pnl(data, n_tp, n_length):
  
    #return data[..., n_tp - int(0.4*n_length):n_tp+n_length]
    return data[..., n_tp:n_tp+n_length]
    
def DCut_data_srf(data, n_ts, n_length):
  
    #return data[..., n_ts - int(0.3*n_length):n_ts+n_length]
    return data[..., n_ts:n_ts+n_length]
    

