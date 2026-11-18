# When the separation principle loses optimality

## Question

If the control changes both the state and the quality of a measurement, can an
ordinary separation-principle action be suboptimal?

The scalar two-step model is

$$x_{t+1}=Ax_t+Bu_t+w_t,\qquad y_0=(C_0+C_1u_0)x_0+z_0.$$

The first action is chosen before observing $y_0$; the second action uses the
resulting conditional state estimate. Initial state and noises are independent
Gaussian variables. The cost includes both stage costs and the terminal cost.
For fixed parameters, the action-dependent part of expected cost is

$$f(u)=\alpha u^2+2\beta\mu_0u+
\frac{\gamma}{(C_0+C_1u)^2+\kappa}.$$

The supplied setting has $A=0.9$, $B=1$, $Q=R=Q_T=1$, prior mean $0.1$,
prior variance $2$, process-noise variance $0.01$, and observation-noise variance
$0.09$. $C_1=2.405$ and $C_0=-C_1u_{sep}$ in the symmetric example.
Variance is distinguished from standard deviation in the simulation.

## Evidence

`run.py` enumerates real stationary roots and evaluates their objective values.
It compares the symmetric case with its closed-form solution, then evaluates
actual two-step sample costs using shared randomness for the candidate policies.
The simulation is a consistency check; the analytic expression supplies the
expected cost. The objective is coercive for this setting and has no real poles.

| Policy | First action | Analytical expected cost |
|---|---:|---:|
| Separation principle | -0.052578 | 4.975032 |
| Alternative minimum, negative branch | -0.248256 | 4.521818 |
| Alternative minimum, positive branch | 0.143100 | 4.521818 |

The difference is 0.453214 for these parameters. It is not a universal performance
gain. The root search uses floating-point arithmetic and is not an exact symbolic
certificate for arbitrary parameter regimes.

## Reproduce

From the repository root: `python studies/separation_gap/run.py`.
Outputs are in `results/`: a figure, cost table, parameter sweep and JSON run record.
The original notebook is retained for traceability. Its four-panel figure silently
omits the fifth requested offset; the standalone runner covers all five offsets.

## Reference and scope

Based on Sattar et al., *Sub-optimality of the Separation Principle for Quadratic
Control from Bilinear Observations*, https://arxiv.org/abs/2504.11555.
The underlying separation-principle counterexample is attributable to that work.
The contribution of this portfolio must be described as implementation/analysis
unless a separate original result is established. See the root contribution record.
