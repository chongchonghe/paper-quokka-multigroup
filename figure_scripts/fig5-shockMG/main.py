#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

OUTDIR = "./out"
DATA_EXACT_DIR = "./data/raw"
os.makedirs(OUTDIR, exist_ok=True)

if "--quokka" in sys.argv:
    DATA_DIR = "./data/raw/quokka-result"
else:
    DATA_DIR = "./data/generated"

try:
    import scienceplots
    plt.style.use(["science", "nature"])
except ImportError:
    pass

plt.rcParams['savefig.dpi'] = 300

# def planck(nu, T):
#     x = nu_unit * nu / (k_B * T)
#     return planck_integral(x) / (np.pi**4 / 15.0) * (c / (4.0 * np.pi) * a_r * T**4)


def E_nu(nu, T):

    nu_unit = 6.62606957e-27
    k_B = 1.380649e-16
    c = 2.997925e+10
    a_r = 7.565731e-15

    def planck_integral(x):
        return x**3 / (np.exp(x) - 1)
        
    x = nu_unit * nu / (k_B * T)
    coeff = nu_unit / (k_B * T)
    return coeff * planck_integral(x) / (np.pi**4 / 15.0) * (a_r * T**4)


def plot_init_T():

    # feature = "run-50bin-no-RSLA/n128-t0"
    figname = "radshock_T_v_t0.pdf"

    data_path = DATA_DIR
    log_path = f"out/log.txt"

    data = np.loadtxt(f"{data_path}/radshock_multigroup_temperature.csv", delimiter=",", skiprows=1)
    data_vel = np.loadtxt(f"{data_path}/radshock_multigroup_velocity.csv", delimiter=",", skiprows=1)
    Lx = 0.01575

    f, ax = plt.subplots()
    ax.set_xlim(0, 1)

    # plot temperature
    x = data[:,0] / Lx
    T_rad = data[:,1]
    T_gas = data[:,2]
    ax.plot(x, T_rad, '-', color='C0', label=r"$T_{\rm rad}$")
    ax.plot(x, T_gas, '--', color='C1', label=r"$T_{\rm gas}$")
    # add text "temperature" above the curve at (0.2, 1.0) with a bottom margin of 0.02 relative to the axis
    ax.text(0.2, 1.05, r"$T_{\rm rad}$", color='C0', ha='left', va='bottom')
    ax.text(0.4, 1.02, r"$T_{\rm gas}$", color='C1', ha='left', va='bottom')
    ax.set_xlabel("$x$ (dimensionless)")
    ax.set_ylabel("temperature (dimensionless)")
    
    # plot velocity in the same axis
    ax2 = ax.twinx()
    x = data_vel[:,0] / Lx
    vel = data_vel[:,1]
    ax2.plot(x, vel / 1e7, '-', color='C4', label=r"$v$")
    # ax2.legend()
    ax2.text(0.2, 5.05, r"velocity", color='C4', ha='left', va='top')
    ax2.set_ylabel(r"velocity / $10^7$ (dimensionless)")

    ax.set_title("shock at t = 0.0")

    # annotate at x = 0.1 and x = 0.8603174603 with arrow
    ax.annotate("A", xy=(0.1, 1.), xytext=(0.1, 1.5), color='r', ha='center', va='center',
                arrowprops=dict(facecolor='r', edgecolor='r', arrowstyle='->'))
    ax.annotate("B", xy=(0.8603174603, 3.65), xytext=(0.8603174603, 3.1), color='C2', ha='center', va='center',
                arrowprops=dict(facecolor='C2', edgecolor='C2', arrowstyle='->'))

    plt.savefig(f"{OUTDIR}/{figname}", bbox_inches='tight')
    

def plot_tend_T():

    # feature = "run-50bin-RSLA-slope-minmod-n512"
    # os.makedirs(f"{OUTDIR}/{feature}", exist_ok=True)
    figname = "radshock_T_tend.pdf"
    
    # read data from {DATADIR}/*
    data = np.loadtxt(f"{DATA_DIR}/radshock_singlegroup_temperature.csv", delimiter=",", skiprows=1)
    data_exact = np.loadtxt(f"{DATA_EXACT_DIR}/exact-solution/shock.txt", delimiter=" ", skiprows=1)
    Lx = 0.01575
    TL = 2.18e6

    figwidth = 3
    f, ax = plt.subplots(figsize=(figwidth, 2/3 * figwidth))
    ax.set_xlim(0, Lx)

    skip = data.shape[0] // 40
    x = data[:,0] 
    T_rad = data[:,1] * TL
    T_gas = data[:,2] * TL
    x_strided = data[::skip,0] 
    T_rad_strided = data[::skip,1] * TL
    T_gas_strided = data[::skip,2] * TL
    x_exact = data_exact[:,0] 
    T_gas_exact = data_exact[:,3] * TL
    T_rad_exact = data_exact[:,4] * TL

    ax.plot(x_exact, T_rad_exact, '-', color='k', label=r"$T_{\rm rad}$ exact")
    ax.plot(x_exact, T_gas_exact, '-', color='k', label=r"$T_{\rm gas}$ exact")
    ax.scatter(x_strided, T_rad_strided, color='C0', marker='o', s=8, label=r"$T_{\rm rad}$", zorder=5)
    ax.scatter(x_strided, T_gas_strided, color='C1', marker='o', s=8, label=r"$T_{\rm gas}$", zorder=6)
    ax.set_xlabel("$x$ (cm)")
    ax.set_ylabel("temperature (K)")

    # write text at bottom right corner
    ax.text(0.85, 0.05, r"$T_{\rm gas}$", color='C1', transform=ax.transAxes, ha='left', va='bottom')
    ax.text(0.85, 0.15, r"$T_{\rm rad}$", color='C0', transform=ax.transAxes, ha='left', va='bottom')

    # adding inset
    box_loc = (0.8 * Lx, 3.44 * TL)
    width = 0.12 * Lx
    height = 0.9 * TL
    y_to_x = (ax.get_ylim()[1] - ax.get_ylim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0])
    inset_width = 0.3
    ax_inset = ax.inset_axes([0.16, 0.3, inset_width, inset_width/y_to_x*(height / width)], xlim=(box_loc[0], box_loc[0] + width), ylim=(box_loc[1], box_loc[1] + height), 
                             xticklabels=[], yticklabels=[],
                             )
    # ax_inset.set_xlim(0.8 * Lx, 0.8 * Lx + width)
    # ax_inset.set_ylim(3.44 * TL, 3.44 * TL + height)
    # ax_inset.set_yticks([3.5, 4.0])
    ax_inset.plot(x_exact, T_rad_exact, '-', color='k', alpha=0.7, zorder=20)
    ax_inset.plot(x_exact, T_gas_exact, '-', color='k', alpha=0.7, zorder=21)
    ax_inset.scatter(x, T_rad, color='C0', marker='o', s=8, zorder=10)
    ax_inset.scatter(x, T_gas, color='C1', marker='o', s=8, zorder=11)

    # connect the inset to the rectangle
    ax.indicate_inset_zoom(ax_inset, edgecolor='k', alpha=0.7, zorder=30)
    # ax.plot([0.16 + inset_width, 0.8], [0.3, 3.4], 'k-', lw=1)
    # ax.plot([0.8 + width, 0.8 + width], [3.4, 3.4], 'k-', lw=1)

    # add text "t = 0.0" at central top
    # ax.text(0.5, 0.95, "t = 0.0", color='k', transform=ax.transAxes, ha='center', va='top')
    ax.set_title("shock at t = 1e-9 s")

    # annotate at x = 0.1 and x = 0.8603174603 with arrow
    ax.annotate("A", xy=(0.1 * Lx, 1. * TL), xytext=(0.1 * Lx, 1.5 * TL), color='r', ha='center', va='center',
                arrowprops=dict(facecolor='r', edgecolor='r', arrowstyle='->'))
    ax.annotate("B", xy=(0.8603174603 * Lx, 3.65 * TL), xytext=(0.8603174603 * Lx, 3.1 * TL), color='C2', ha='center', va='center',
                arrowprops=dict(facecolor='C2', edgecolor='C2', arrowstyle='->'))
    
    # plt.savefig(f"{OUTDIR}/{feature}/radshock_multigroup.pdf", bbox_inches='tight')
    plt.savefig(f"{OUTDIR}/{figname}", bbox_inches='tight')

    return


def plot_spec(plottime='end'):

    # feature = "n512"
    # feature = "n128-with-spec"

    if plottime == 'start':
        # feature = "run-50bin-no-RSLA/n128-t0"
        is_F_log = False
        show_exact = True
        show_0_flux = True
        figname = "radshock_spec_t0.pdf"
    elif plottime == 'end':
        # feature = "run-50bin-RSLA-slope-minmod/n128"
        is_F_log = True
        show_exact = False
        show_0_flux = False
        figname = "radshock_spec_tend.pdf"

    data_path = DATA_DIR
    log_path = f"out/log.txt"

    msize = 6
    figwidth = 4
    f, axs = plt.subplots(2, 2, sharex='all', sharey='none', figsize=(figwidth, 2/3 * figwidth))
    axs[0][0].set_ylabel(r"$\log_{10} E_{\nu}$ (erg cm$^{-3}$ Hz$^{-1}$)", fontsize='small')
    if is_F_log:
        axs[1][0].set_ylabel(r"$\log_{10} (-F_{\nu})$ (erg cm$^{-2}$ s$^{-1}$ Hz$^{-1}$)", fontsize='x-small')
    else:
        axs[1][0].set_ylabel(r"$F_{\nu}$ (erg cm$^{-2}$ s$^{-1}$ Hz$^{-1}$)", fontsize='x-small')

    # c = 3.0e10 # speed of light in cm/s
    c_reduce = 6.920000e+08

    T0 = 2.18e6 # reference termperature in K
    
    # loc 1, E_nu
    ax = axs[0][0]
    ax.set(xscale="linear", yscale="linear", xlabel=r"$\log_{10} \nu$ (Hz)")
    fn = f"{data_path}/radshock_multigroup_E_nu_loc1.csv"
    with open(fn, 'r') as fi:
        line = fi.readline()
        T_gas = float(line.split(',')[2].split(' ')[-1]) * T0
        T_rad = float(line.split(',')[3].split(' ')[-1]) * T0
        vel = float(line.split(',')[4].split(' ')[-1]) # cm/s
    data_e_nu_1 = np.loadtxt(fn, skiprows=1, delimiter=",")
    nu_L = data_e_nu_1[:, 0] # N_g
    nu_R = np.zeros(len(nu_L))
    nu_R[:-1] = nu_L[1:]
    nu_R[-1] = np.inf
    dnu = nu_R - nu_L # -1: inf. Units: Hz
    dnu[-1] = nu_L[-1] # = 2 * nu_L[-1] - nu_L[-1]
    nu_center = np.sqrt(nu_L * nu_R) # 0: 0, -1: inf
    nu_center[0] = nu_L[0] / 2.0
    nu_center[-1] = nu_L[-1] * 2.0
    E_r = data_e_nu_1[:, 1] # N_g. Units: erg cm^-3
    dE_dnu = E_r / dnu # N_g. Units: erg cm^-3 Hz^-1
    ax.scatter(np.log10(nu_center), np.log10(dE_dnu), s=msize, marker="o", label="simulation", color='C0')
    # plot BB spectrum
    nu = np.logspace(np.log10(nu_L[0]), np.log10(nu_L[-1]), 100)
    E_nu_exact = E_nu(nu, T_gas)
    ls = '-' if show_exact else '--'
    ax.plot(np.log10(nu), np.log10(E_nu_exact), ls=ls, label="Black body", color='grey', zorder=-1)

    # loc 1, F_nu / c
    ax = axs[1][0]
    ax.set(xscale="linear", yscale="linear", xlabel=r"$\log_{10} \nu$ (Hz)")
    fn = f"{data_path}/radshock_multigroup_F_nu_loc1.csv"
    data_f_nu_1 = np.loadtxt(fn, skiprows=1, delimiter=",")
    nu_L = data_f_nu_1[:, 0] # N_g
    nu_R = np.zeros(len(nu_L))
    nu_R[:-1] = nu_L[1:]
    nu_R[-1] = np.inf
    dnu = nu_R - nu_L # -1: inf. Units: Hz
    dnu[-1] = nu_L[-1] # = 2 * nu_L[-1] - nu_L[-1]
    nu_center = np.sqrt(nu_L * nu_R) # 0: 0, -1: inf
    nu_center[0] = nu_L[0] / 2.0
    nu_center[-1] = nu_L[-1] * 2.0
    F_r = data_f_nu_1[:, 1] # N_g. Units: erg cm^-2 s^-1
    dF_dnu = F_r / dnu # N_g. Units: erg cm^-2 s^-1 Hz^-1
    # ax.scatter(nu_center, dF_dnu, s=msize, marker="o", label="simulation", color='C0')
    if is_F_log:
        ax.scatter(np.log10(nu_center), np.log10(-dF_dnu), s=msize, marker="o", label="simulation", color='C0')
    else:
        ax.scatter(np.log10(nu_center), dF_dnu, s=msize, marker="o", label="simulation", color='C0')
    # plot 4/3 * vel * B_nu
    nu = np.logspace(np.log10(nu_L[0]), np.log10(nu_L[-1]), 100)
    F_BB_nu = 4/3 * vel * E_nu(nu, T_gas)
    if is_F_log:
        ax.plot(np.log10(nu), np.log10(F_BB_nu), '--', label="Black body", color='grey')
    if show_0_flux:
        ax.axhline(y=0, color='grey', ls='-', zorder=-1)
    
    # loc 2, E_nu
    ax = axs[0][1]
    ax.set(xscale="linear", yscale="linear", xlabel=r"$\log_{10} \nu$ (Hz)")
    fn = f"{data_path}/radshock_multigroup_E_nu_loc2.csv"
    with open(fn, 'r') as fi:
        line = fi.readline()
        T_gas = float(line.split(',')[2].split(' ')[-1]) * T0
        T_rad = float(line.split(',')[3].split(' ')[-1]) * T0
        vel = float(line.split(',')[4].split(' ')[-1]) # cm/s
    data_e_nu_2 = np.loadtxt(fn, skiprows=1, delimiter=",")
    nu_L = data_e_nu_2[:, 0] # N_g
    nu_R = np.zeros(len(nu_L))
    nu_R[:-1] = nu_L[1:]
    nu_R[-1] = np.inf
    dnu = nu_R - nu_L # -1: inf
    dnu[-1] = nu_L[-1] # = 2 * nu_L[-1] - nu_L[-1]
    nu_center = np.sqrt(nu_L * nu_R) # 0: 0, -1: inf
    nu_center[0] = nu_L[0] / 2.0
    nu_center[-1] = nu_L[-1] * 2.0
    E_r = data_e_nu_2[:, 1] # N_g
    dE_dnu = E_r / dnu # N_g
    ax.scatter(np.log10(nu_center), np.log10(dE_dnu), s=msize, marker="o", label="simulation", color='C0')
    # plot the exact solution
    nu = np.logspace(np.log10(nu_L[0]), np.log10(nu_L[-1]), 100)
    E_nu_exact = E_nu(nu, T_gas)
    E_nu_exact_at_T_rad = E_nu(nu, T_rad)
    ls = '-' if show_exact else '--'
    ax.plot(np.log10(nu), np.log10(E_nu_exact), ls=ls, label="Black body", color='grey', zorder=-1)
    ax.plot(np.log10(nu), np.log10(E_nu_exact_at_T_rad), ls="--", label="Black body", color='k', zorder=-2)

    # loc 2, F_nu / c
    ax = axs[1][1]
    ax.set(xscale="linear", yscale="linear", xlabel=r"$\log_{10} \nu$ (Hz)")
    fn = f"{data_path}/radshock_multigroup_F_nu_loc2.csv"
    data_f_nu_1 = np.loadtxt(fn, skiprows=1, delimiter=",")
    nu_L = data_f_nu_1[:, 0] # N_g
    nu_R = np.zeros(len(nu_L))
    nu_R[:-1] = nu_L[1:]
    nu_R[-1] = np.inf
    dnu = nu_R - nu_L # -1: inf
    dnu[-1] = nu_L[-1] # = 2 * nu_L[-1] - nu_L[-1]
    nu_center = np.sqrt(nu_L * nu_R) # 0: 0, -1: inf
    nu_center[0] = nu_L[0] / 2.0
    nu_center[-1] = nu_L[-1] * 2.0
    F_r = data_f_nu_1[:, 1] # N_g
    dF_dnu = F_r / dnu # N_g
    # ax.scatter(nu_center, dF_dnu, s=msize, marker="o", label="simulation", color='C0')
    if is_F_log:
        ax.scatter(np.log10(nu_center), np.log10(-dF_dnu), s=msize, marker="o", label="simulation", color='C0')
    else:
        ax.scatter(np.log10(nu_center), dF_dnu, s=msize, marker="o", label="simulation", color='C0')
    # plot 4/3 * vel * B_nu
    nu = np.logspace(np.log10(nu_L[0]), np.log10(nu_L[-1]), 100)
    F_BB_nu = 4/3 * vel * E_nu(nu, T_gas)
    if is_F_log:
        ax.plot(np.log10(nu), np.log10(F_BB_nu), '--', label="Black body", color='grey')
    if show_0_flux:
        ax.axhline(y=0, color='grey', ls='-', zorder=-1)

    # plt.savefig(f"out/{feature}/E_nu_loc1-full.pdf")

    axs[0][0].set_xlim([15, 19])
    # axs[0][0].set_ylim([1e-12, 1e-5])
    axs[0][0].set_ylim([-14, -4])
    axs[0][1].set_ylim([-14, -4])
    # axs[1][0].set_ylim([-5, 4])
    axs[1][0].set_ylim([-3, 5])
    axs[1][1].set_ylim([-3, 5])
    # plt.legend(loc='upper left', frameon=True, fontsize='small')
    axs[0][0].text(0.05, 0.95, "loc A", color="r", transform=axs[0][0].transAxes, ha='left', va='top')
    axs[1][0].text(0.05, 0.95, "loc A", color="r", transform=axs[1][0].transAxes, ha='left', va='top')
    axs[0][1].text(0.05, 0.95, "loc B", color="C2", transform=axs[0][1].transAxes, ha='left', va='top')
    axs[1][1].text(0.05, 0.95, "loc B", color="C2", transform=axs[1][1].transAxes, ha='left', va='top')
    if show_exact:
        text1 = r"exact $(B_{\nu}(T_{\rm gas}))$"
        text2 = r"exact $(B_{\nu}(T_{\rm gas}))$"
    else:
        text1 = r"$B_{\nu}(T_{\rm gas})$"
        text2 = r"$B_{\nu}(T_{\rm gas})$"
        text2_rad = r"$B_{\nu}(T_{\rm rad})$"
    axs[0][0].text(17.9, -11, text1, color="grey", ha='right', va='top', fontsize='small')
    axs[0][1].text(18.5, -11, text2, color="grey", ha='right', va='top', fontsize='small')
    axs[0][1].text(18.55, -12, text2_rad, color="k", ha='right', va='top', fontsize='small')
    if is_F_log:
        axs[1][0].text(17.76, -1.8, r"$\frac{4}{3} v B_{\nu}(T_{\rm gas})$", color="grey", ha='right', va='top', fontsize='small')
        axs[1][1].text(18.5, -1.8, r"$\frac{4}{3} v B_{\nu}(T_{\rm gas})$", color="grey", ha='right', va='top', fontsize='small')

    # remove xlables for the top row and decrease the space between subplots
    plt.setp(axs[0], xlabel='')
    plt.subplots_adjust(hspace=0.1)
    plt.savefig(f"{OUTDIR}/{figname}", bbox_inches='tight')

    return


def test():
    
    # Sample data - replace these with your actual data arrays
    x = np.linspace(-1, 0.02, 100)  # Example x values from -1.0 cm to 0.02 cm
    T_gas_exact = np.sin(x) + 2.5   # Dummy data for T_gas_exact
    T_rad_exact = np.cos(x) + 2.5   # Dummy data for T_rad_exact
    T_gas = T_gas_exact + np.random.normal(0, 0.1, size=x.size)  # Perturbed data for T_gas
    T_rad = T_rad_exact + np.random.normal(0, 0.1, size=x.size)  # Perturbed data for T_rad

    # Plotting the exact solutions
    plt.plot(x, T_gas_exact, 'b--', label='T_gas_exact', linewidth=2)
    plt.plot(x, T_rad_exact, 'r--', label='T_rad_exact', linewidth=2)

    # Plotting the numerical solutions
    plt.scatter(x, T_gas, color='blue', label='T_gas')
    plt.scatter(x, T_rad, color='orange', label='T_rad')

    # Additional plot settings
    plt.xlabel('x [cm]')
    plt.ylabel('Temperature')
    plt.title('Temperature Distribution')
    plt.legend()
    plt.grid(True)

    # Adding inset to the plot
    ax = plt.gca()
    inset_ax = ax.inset_axes([0.5, 0.5, 0.47, 0.47])  # X, Y, width, height in normalized plot coordinates
    inset_ax.plot(x, T_gas_exact, 'b--', linewidth=2)
    inset_ax.plot(x, T_rad_exact, 'r--', linewidth=2)
    inset_ax.scatter(x, T_gas, color='blue')
    inset_ax.scatter(x, T_rad, color='orange')
    inset_ax.set_xlim(-1, -0.5)  # Adjust these limits to zoom appropriately
    inset_ax.set_ylim(1, 4)      # Adjust these limits based on your data
    inset_ax.grid(True)

    # Show the plot
    plt.show()


if __name__ == "__main__":

    # plot_init_T()
    # plot_spec('start')
    # test()

    plot_tend_T()
    plot_spec()
