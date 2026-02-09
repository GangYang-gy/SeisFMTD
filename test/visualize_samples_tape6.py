import sys
sys.path.insert(1, '/scratch/gang97/01_workshop/software/HMC_package/SeisFMTD')

import pickle
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from MTfit.convert.moment_tensor_conversion import MT6_Tape
from pyrocko import moment_tensor as pmt
import os

#------------------ Utility ------------------#
def ENU2NED(mt):
    """Convert moment tensor from ENU to NED convention"""
    new_mt = np.zeros_like(mt)
    new_mt[0] = mt[1]
    new_mt[1] = mt[0]
    new_mt[2] = mt[2]
    new_mt[3] = mt[3]
    new_mt[4] = -1 * mt[5]
    new_mt[5] = -1 * mt[4]
    return np.array(new_mt)

def convert_to_q(mt_enu):
    """Convert ENU moment tensor to Tape parameters q"""
    mt_ned = ENU2NED(mt_enu)
    MT6 = np.array([
        mt_ned[0], mt_ned[1], mt_ned[2],
        np.sqrt(2)*mt_ned[3],
        np.sqrt(2)*mt_ned[4],
        np.sqrt(2)*mt_ned[5]
    ])
    gamma, delta, strike, cosdip, slip = MT6_Tape(MT6)
    m = pmt.values_to_matrix(mt_ned)
    moment = pmt.MomentTensor(m=m).scalar_moment()
    mag = (np.log10(moment) - 9.1)/1.5
    q = np.array([
        np.rad2deg(strike)[0],
        np.rad2deg(np.arccos(cosdip))[0],
        np.rad2deg(slip)[0],
        mag,
        np.rad2deg(delta)[0],
        np.rad2deg(gamma)[0]
    ])
    return q

#------------------ True / Initial model ------------------#
true_strike = 300; true_dip = 70; true_rake = -10; true_mag = 4.3
true_colat = 90; true_lat = 90-true_colat; true_lon = 0
init_strike = true_strike+10; init_dip = true_dip+10; init_rake = true_rake-10
init_mag = true_mag+0.2; init_lat = 90-(true_colat+5); init_lon = true_lon+5

cols_with_misfit = ('strike', 'dip', 'slip', 'mag', 'lat', 'lon', 'misfit')
cols_no_misfit = ('strike', 'dip', 'slip', 'mag', 'lat', 'lon')
Nq = len(cols_no_misfit)

results = []
results_with_misfit = []

#------------------ Load HMC samples ------------------#
sample_pkl = './results/Samples_N1000_.pkl'
with open(sample_pkl, 'rb') as f:
    print(f'Unpacking {sample_pkl} ...')
    samples = pickle.load(f)
    misfits = pickle.load(f)

for i, sample in enumerate(samples):
    if i < 50:  # burn-in skip
        continue
    q = convert_to_q(np.array(sample[:6]))
    results.append(dict(zip(cols_no_misfit, q)))
    results_with_misfit.append(dict(zip(cols_with_misfit, np.append(q, misfits[i]))))

df = pd.DataFrame(results)
df1 = pd.DataFrame(results_with_misfit)

estimates = df.mean().values    # 均值
uncertainties = df.std().values  # 标准差
#------------------ User options ------------------#
show_hmc = True      # 是否显示 HMC samples
show_ukf = False   # 是否显示 UKF trajectory
show_true_init = True  # 是否显示 true solution 和 initial model

#------------------ Plot ------------------#
plt.figure(figsize=(8, 10))
sns.set(style="ticks", font_scale=1.2)

# PairGrid
if show_hmc:
    g = sns.PairGrid(df, diag_sharey=True, corner=True)
    g.map_diag(sns.histplot, bins=12, color="lightblue", edgecolor="black")
    g.map_lower(sns.scatterplot, s=30, color="gray", alpha=0.6, label="HMC Samples")
    g.map_lower(sns.kdeplot, levels=5, color="navy", linewidths=1)

else:
    # 如果不显示 HMC，只创建空 PairGrid 用于绘制 true/initial/UKF
    g = sns.PairGrid(df.iloc[:1], diag_sharey=True, corner=True)  # 只取一行，避免报错
    g.map_diag(lambda *args, **kwargs: None)
    g.map_lower(lambda *args, **kwargs: None)

# True & initial model
if show_true_init:
    true_model = np.array([true_strike, true_dip, true_rake, true_mag, true_lat, true_lon], dtype=object)
    initial_model = np.array([init_strike, init_dip, init_rake, init_mag, init_lat, init_lon], dtype=object)
    for i in range(Nq):
        for j in range(Nq):
            if j < i:
                g.axes[i, j].scatter(x=true_model[j], y=true_model[i], s=200, color='r', marker='*', label="True Solution")
                g.axes[i, j].scatter(x=initial_model[j], y=initial_model[i], s=120, color='green', marker='^', label="Initial Model")

# UKF trajectory
if show_ukf:
    #------------------ Load UKF trajectory & final estimate ------------------#
    trajectory_raw = np.load('./results/ukf_trajectory.npy') # shape (num_iter, 6)
    trajectory = np.array([convert_to_q(mt) for mt in trajectory_raw])

    for i in range(Nq):
        for j in range(Nq):
            if j < i:
                g.axes[i, j].plot(trajectory[:, j], trajectory[:, i], color='orange', linewidth=1.5)
                g.axes[i, j].scatter(trajectory[:, j], trajectory[:, i], s=20, color='orange', label="UKF Trajectory")

# 对角线：显示 estimate ± uncertainty
for i, ax in enumerate(np.diag(g.axes)):
   text = f"{estimates[i]:.2f} ± {uncertainties[i]:.2f}"
   ylim = ax.get_ylim()
   xlim = ax.get_xlim()
   ax.text(0.5 * (xlim[0]+xlim[1]), ylim[1], text, ha='center', va='top', fontsize=18, color='red',fontweight='bold'  )
   
   
# 自动生成图例
unique_handles = {}
for ax_row in g.axes:
    for ax in ax_row:
        if ax is None:
            continue
        handles, labels = ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            if l not in unique_handles:
                unique_handles[l] = h

# -------- 固定 legend 顺序 -------- #
legend_order = ["Initial Model", "True Solution", "HMC Samples", "UKF Trajectory"]
ordered_labels = [l for l in legend_order if l in unique_handles]
ordered_handles = [unique_handles[l] for l in ordered_labels]

# Layout & labels
plt.tight_layout()
for i, col in enumerate(cols_no_misfit):
    g.axes[i, 0].set_ylabel(col, fontsize=30)
    g.axes[-1, i].set_xlabel(col, fontsize=30)

g.fig.align_ylabels(g.axes[:, 0])
g.fig.legend(
    ordered_handles,
    ordered_labels,
    loc="upper left",
    bbox_to_anchor=(0.55, 0.95),
    frameon=True,
    fontsize=30,
    markerscale=2
)
  
if show_ukf:
    plt.savefig('./hmc+ukf.png', dpi=400)
else:
    plt.savefig('./hmc_samples.png', dpi=400)
