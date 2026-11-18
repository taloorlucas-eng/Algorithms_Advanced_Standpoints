# Validation record

Review date: 18 November 2026. This record concerns the supplied archive and the
separate curated candidate, not any existing online repository.

## Executed successfully

- Original scalar separation-gap notebook code, executed in a fresh process.
- Standalone scalar runner with 200,000 samples, seed 0.
- Five numerical tests: symmetric analytical minima versus polynomial roots;
  dense-grid check at five offsets; paired Monte Carlo gap check with seed 17;
  analytic mixture score versus a finite-difference log-density gradient;
  hand-coded NumPy MLP input/parameter gradients versus finite differences.

The scalar analytical gap was 0.4532140930 (rounded display: 0.453214).
The seed-0 paired Monte Carlo estimate was 0.454812 with one standard error 0.002602.
Read `studies/separation_gap/results/run.json` for full precision and settings.

## Source and reported-result checks

Python source in the curated candidate was syntax-checked. Original input source
included an invalid `i+!` expression in the excluded `backward_process` fragment.
The supplied DMD tables agree with the reported no-winner outcome and qualitative
collapse diagnosis. These tables are provided as historical reported results;
their original computation was not independently reproduced here.

## Not validated

The environment did not contain PyTorch. Full B-MPC optimization/closed-loop
execution, face-model training, data downloading, and NODE training were not run.
The mathematical proofs in the manuscripts were not reviewed. Passing the five
small numerical checks is not certification of those pipelines or of a paper.
File provenance hashes record what was supplied, not who authored it.

## Re-run

From the repository root:

```bash
python studies/separation_gap/run.py
python -m unittest discover -s tests -v
```

The fifth check uses scikit-learn and will be explicitly skipped if it is absent;
install `requirements-optional.txt` to enable it. Do not report a skipped check as
a passed check. Test output and exact package versions accompany this candidate.
