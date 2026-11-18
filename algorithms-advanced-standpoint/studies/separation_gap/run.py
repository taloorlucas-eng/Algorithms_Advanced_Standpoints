"""Reproduce the supplied two-step example; no external data or GPU required."""
from pathlib import Path
import json
import time
import platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import s, Kt, Pt, alpha, gamma, kappa, u_sep, objective, argmin_objective


def simulate_costs(actions, C0, C1, samples=200_000, seed=0):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(s.mu0, np.sqrt(s.Sigma0), samples)
    w0, w1 = rng.normal(0, np.sqrt(s.Pnoise), (2, samples))
    z0 = rng.normal(0, np.sqrt(s.Znoise), samples)
    values = []
    for u0 in actions:
        C = C0 + C1 * u0
        y0 = C * x0 + z0
        gain = s.Sigma0 * C / (C*C*s.Sigma0+s.Znoise)
        mu1 = s.A*(s.mu0+gain*(y0-C*s.mu0))+s.B*u0
        u1 = -s.A*s.B*s.QT/(s.B*s.B*s.QT+s.R)*mu1
        x1 = s.A*x0+s.B*u0+w0
        x2 = s.A*x1+s.B*u1+w1
        values.append(s.Q*(x0*x0+x1*x1)+s.R*(u0*u0+u1*u1)+s.QT*x2*x2)
    return np.asarray(values)


def main():
    started = time.perf_counter()
    out = Path(__file__).resolve().parent / 'results'
    out.mkdir(exist_ok=True)
    C1, C0 = s.C1, -s.C1*u_sep
    radius = np.sqrt(abs(C1)*np.sqrt(gamma/alpha)-kappa)/abs(C1)
    exact_minima = u_sep + np.array([-radius, radius])
    numeric_minima = argmin_objective(C0, C1)
    actions = np.r_[u_sep, exact_minima]
    constant = (s.Q*(s.mu0*s.mu0+s.Sigma0)
                + Kt[1]*(s.A*s.A*(s.mu0*s.mu0+s.Sigma0)+s.Pnoise)
                + (Pt[1]+s.QT)*s.Pnoise)
    predicted = constant + objective(actions, C0, C1)[2]
    costs = simulate_costs(actions, C0, C1)
    table = pd.DataFrame({'policy':['Separation','Negative minimum','Positive minimum'],
                          'u0':actions, 'analytical_cost':predicted,
                          'mc_mean':costs.mean(1), 'mc_standard_error':costs.std(1,ddof=1)/np.sqrt(costs.shape[1])})
    table.to_csv(out/'costs.csv',index=False)
    rows = []
    for delta in (1.0, 0.0, -0.5, -1.0, -2.0):
        offset = -s.C1*(u_sep+delta)
        roots = argmin_objective(offset,s.C1)
        gap = float(objective(u_sep,offset,s.C1)[2]-objective(roots[0],offset,s.C1)[2])
        rows.append({'delta':delta, 'C0_over_C1':offset/s.C1,
                     'optimal_actions':json.dumps(roots.tolist()),'expected_gap':gap})
    pd.DataFrame(rows).to_csv(out/'offsets.csv',index=False)

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(1,2,figsize=(10.4,4.2))
    fig.subplots_adjust(left=.08,right=.98,bottom=.17,top=.78,wspace=.30)
    u = np.linspace(-.65,.55,1500)
    axes[0].plot(u,constant+objective(u,C0,C1)[2],color='#253f59',lw=2)
    axes[0].scatter(actions,predicted,c=['#a43d40','#227365','#227365'],s=55,zorder=3)
    axes[0].annotate('Separation action',xy=(u_sep,predicted[0]),xytext=(.04,5.08),
                     arrowprops={'arrowstyle':'-','color':'#a43d40'},fontsize=9)
    axes[0].set(title='Control changes measurement quality',xlabel='First action',ylabel='Analytical expected cost')
    ratios = np.linspace(-1.3,1.3,401)
    gaps = [float(objective(u_sep,r*C1,C1)[2]-objective(argmin_objective(r*C1,C1)[0],r*C1,C1)[2]) for r in ratios]
    axes[1].plot(ratios,gaps,color='#227365',lw=2)
    axes[1].set(title='Gap varies with the observation model',xlabel='$C_0/C_1$',ylabel='$J_{sep}-J_{opt}$')
    for ax in axes: ax.grid(alpha=.18)
    fig.suptitle('Two-step scalar control | Numerical reproduction',fontsize=13,fontweight='bold',y=.96)
    fig.savefig(out/'summary.png',dpi=180)
    plt.close(fig)
    paired = costs[0]-costs[1]
    record = {'seed':0,'samples':costs.shape[1],'python':platform.python_version(),
              'numpy':np.__version__,'model_parameters':{**vars(s),'C0':C0,'C1':C1},
              'C0':C0,'C1':C1,'analytic_minima':exact_minima.tolist(),
              'numerical_minima':numeric_minima.tolist(),
              'analytical_gap':float(predicted[0]-predicted[1]),
              'paired_mc_gap':float(paired.mean()),
              'paired_mc_se':float(paired.std(ddof=1)/np.sqrt(len(paired))),
              'elapsed_seconds':time.perf_counter()-started,
              'scope':'Numerical consistency check for the supplied scalar setting; not a novel theorem.'}
    (out/'run.json').write_text(json.dumps(record,indent=2)+'\n')
    print(table.to_string(index=False))
    print(f"Analytical gap: {record['analytical_gap']:.6f}; paired MC: {record['paired_mc_gap']:.6f} +/- {record['paired_mc_se']:.6f} (one SE)")
    print('Results:',out)


if __name__=='__main__':
    main()
