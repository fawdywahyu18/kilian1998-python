# matlab/

MATLAB code originally written by Lutz Kilian (University of Michigan, April 1997), implementing the bias-corrected bootstrap confidence intervals for impulse response functions as described in Kilian (1998).

## Original Kilian Files

| File | Description |
|---|---|
| `olsvarc.m` | OLS estimation of VAR with intercept in companion form |
| `asybc.m` | Analytical bias correction following Pope (1990) |
| `irfvar.m` | Impulse response function via Cholesky decomposition |
| `boot.m` | Bootstrap CI (1000 replications, 90% interval) |
| `vec.m` | Helper: column-major matrix vectorization |
| `point.m` | Main controller: calls `olsvarc`, `asybc`, and `boot` |

## Additional Files

Two files were added outside of Kilian's original code for the purpose of verifying the Python replication in `kilian_var.py`:

| File | Description |
|---|---|
| `generate_test_data.m` | Generates simulated bivariate VAR(2) data and exports to `test_data_var.csv` |
| `point_test.m` | Modified version of Kilian's `point.m` that loads data from CSV and exports IRF point estimates and CI bounds to CSV for comparison with Python output |
