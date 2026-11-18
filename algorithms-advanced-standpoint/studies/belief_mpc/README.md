# Belief-space model predictive control

## Scope

Implementation study based on Cao et al., *Dual Control of Linear Systems from
Bilinear Observations with Belief Space Model Predictive Control*,
https://arxiv.org/abs/2604.24663. The planner optimizes a deterministic belief
surrogate with zero future innovations; this is not the exact stochastic control
problem. Filtering uses a Joseph-form covariance update.

The original notebook contains one executed planning comparison. Its stored
B-MPC surrogate objective is 37.121743 and its gradient infinity norm is about
1.4e-5, above the stated 1e-7 tolerance. It reports `gradient_tolerance_met=False`.
These are supplied notebook outputs, not a rerun during this audit. The notebook
contains closed-loop and repeated-trial functions, but their invocation is commented
out. Thus the saved planning plot does not establish closed-loop superiority.

## Run modes

Install `requirements-optional.txt` first. From the repository root:

```bash
python studies/belief_mpc/run.py --mode planning
python studies/belief_mpc/run.py --mode smoke
python studies/belief_mpc/run.py --mode sweep
```

`planning` and `smoke` use deliberately reduced H=5, T=8 optimizer settings to
check execution. `sweep` uses T=300, horizons 5/15/30 and ten paired trials; it may
be expensive. It is a selected benchmark, not a claim to reproduce every paper
setting. All modes save configuration and numerical diagnostics under `outputs/`.

## Before presenting a performance claim

1. Run the smoke case in a clean PyTorch environment.
2. Check covariance symmetry/positive semidefiniteness and autograd with numerical
   finite differences; record optimizer stops and restart sensitivity.
3. Run paired closed-loop trials and retain raw costs for all controllers.
4. Report the mean paired cost difference with uncertainty, runtime, horizon and
   termination diagnostics. Do not discard unconverged or unfavorable runs.
5. State every departure from the reference paper. Name a result a reproduction
   only to the extent that its setting and reported quantities actually match.

The audit environment had no PyTorch. The new runner was syntax-checked; no
end-to-end B-MPC execution or performance validation is claimed here.
