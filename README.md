# One-Loop 4PCF Covariance Figure Reproduction Code

This repository contains scripts used to reproduce selected numerical figures from the papers:

- **Analytical Template for the 4-Point Correlation Function Covariance Beyond the Gaussian Random Field I: 1-Loop Corrections involving Second-Order Densities**  
  arXiv:2509.05419

- **Analytical Template for the 4-Point Correlation Function Covariance Beyond the Gaussian Random Field II: 1-Loop Corrections involving Third-Order Densities**  
  arXiv:2509.05422

This repository does **not** provide a complete public implementation of the full one-loop 4PCF covariance calculation. Instead, it provides a reproducible numerical workflow for generating selected figures shown in the papers. These include convergence tests, selected radial-integral evaluations, and the scale-dependence plot showing the expected size of the one-loop correction relative to the Gaussian covariance.

## Purpose of the repository

The purpose of this repository is to make selected numerical results from the papers reproducible. In particular, the code allows the user to reproduce:

1. Convergence-test figures for selected radial-integral calculations.
2. Selected radial-integral plots used to validate the numerical pipeline.
3. The one-loop correction scale plot, which estimates the fractional size of the one-loop covariance correction relative to the Gaussian covariance as a function of scale \(R\).

The correction scale plot is intended to show at which scales the one-loop covariance contribution is expected to become a given percentage of the Gaussian covariance contribution.

## Repository structure

```text
One-Loop-4PCF-Covariance/
│
├── Computing_All_Constants/
│   ├── Precomputation_Basic_Constants.py
│   ├── precompute_L_tensor_blocks.py
│   ├── precompute_Modified_J_m_blocks.py
│   └── precompute_Qbar_SS_blocks.py
│
├── Convergence_Computation/
│   └── Compute_Projected_SS_Convergence.py
│
├── Convergence_Test/
│
├── Final_Radial_Integrals/
│   ├── S_SS_tables_Buildup.py
│   └── f_integral.py
│
├── Intermediate_Radial_Integrals/
│   ├── Build_h_tables.py
│   └── Compute_gTable.py
│
├── One_Loop_Correction_Script/
│   └── Script/
│       └── NLO_Correction_Scale_Computation.ipynb
│
├── Radial_Integral_Plotting/
│
├── Figure_outputs/
│
├── 2nd_Order_Cov_Figures/
│
├── intermediate_radial_integrals/
│
├── plotting_script/
│
└── README.md
