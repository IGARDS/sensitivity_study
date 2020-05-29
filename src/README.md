# Rankability and Sensitivity Experiments

There's a lot going on here, so let's break down what each of these files do and how they fit together.

## Sensitivity Testbed

The core of the work in this directory supports a single workflow called the Sensitivity Testbed which attempts to measure the noise sensitivity of ranking methods and correlate that with other measurements about the data. The testbed is comprised of three main pieces:

1. **sensitivity_tests.py:** Contains the framework for measuring sensitivity and other metrics of a dataset. It also contains the code for loading / sampling initial data and perturbations. If you are curious about how a specific metric is calculated or where the data / noise are coming from, this is the place to look. (This framework is also supported by the methods implemented in **utilities.py**.)
2. **rankability_sensitivity_tests.ipynb:** The notebook which utilizes the above framework to collect and store data about the sensitivity and rankability of data.
3. **rankability_sensitivity_tests_real_data.ipynb:** A clone of the above notebook with some parameters changed to load in real data.
4. **sensitivity_analysis.ipynb:** The notebook which analyzes the data collected by the above rankability_sensitivity_tests notebook and looks for correlations between measures of rankability and sensitivity.

## Other Experiments

In addition to the sensitivity testbed, this directory also contains many stand-alone experiments which explore related questions to inform the design and use of the testbed.

1. **PermutationToPMapping.ipynb:** Experiment to determine whether each optimal ranking of P was equally likely to be found by Gurobi after randomly permuting D. **Conclusion: Yes, randomly permuting D results in a uniformly random sample from the P set.**
2. **kendall_tau_speed_test.ipynb:** Experiment to determine which implementation of kendall_tau is most efficient **Conclusion: stats.kendall_tau() with np.argsort is faster than our implementation.**
3. **massey_colley_proximity_to_P.ipynb:** Experiment to determine whether Massey and Colley produce rankings close-to/inside P or if they produce wildly different rankings. **Conclusion: *NOT CONCLUSIVE, TO BE REPEATED WITH FULL P SET***

## Out-dated Files

- **prototype_sensitivity.ipynb** (First prototype for measuring sensitivity of data)
- **noise_level_sensitivity_tests.ipynb** (Early prototype exploring relationship between sensitivity and noise strength)


# Setup
- Gurobi (TODO: explain gurobi setup)
- pyrankability_dev (TODO: explain pyrankability_dev setup)
- other requirements (TODO: list other requirements and the pip command to install them)

# TODO:

These files have not yet been explained in this README:

- Ranking_Algs_P_Set_Tests.ipynb
- method_tests.ipynb
- matrix_visualizations.ipynb
- L2_Most_Sampled_Tests.ipynb