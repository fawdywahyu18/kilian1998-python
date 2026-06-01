# kilian1998-python

Python replication of Kilian (1998) bias-corrected bootstrap confidence intervals for impulse response functions, converted from the original MATLAB code. Verified against MATLAB output using simulated VAR(2) data.

## Reference

Kilian, L. (1998). Small-sample confidence intervals for impulse response functions. *Review of Economics and Statistics*, 80(2), 218–230.

## Contents

```
kilian1998-python/
│
├── README.md
├── kilian_var.py              # Python replication (main)
│
├── matlab/
│   ├── olsvarc.m              # OLS estimation of VAR in companion form
│   ├── asybc.m                # Bias correction (Pope 1990)
│   ├── irfvar.m               # Impulse response function via Cholesky
│   ├── boot.m                 # Bootstrap CI
│   ├── vec.m                  # Helper: matrix vectorization
│   ├── point_test.m           # Modified point.m for verification
│   └── generate_test_data.m   # Generate simulated VAR(2) test data
│
└── data/
    └── test_data_var.csv      # Simulated VAR(2) test data (T=200, q=2)
```

## Verification

Output from `kilian_var.py` was verified against `point_test.m` using the same simulated VAR(2) dataset (`test_data_var.csv`):

- **IRF point estimates**: identical (max abs diff < 1e-4)
- **Bootstrap CI**: near-identical (mean abs diff ~0.004), small differences due to RNG differences between MATLAB and Python

## Usage

**Step 1**: Generate test data in MATLAB
```matlab
generate_test_data.m   % outputs: test_data_var.csv
```

**Step 2**: Run MATLAB replication for benchmark
```matlab
point_test.m   % outputs: irf_point_matlab.csv, ci_lower_matlab.csv, ci_upper_matlab.csv
```

**Step 3**: Run Python replication
```bash
python kilian_var.py
# outputs: irf_point_python.csv, ci_lower_python.csv, ci_upper_python.csv
```

## Requirements

```
numpy
pandas
```
