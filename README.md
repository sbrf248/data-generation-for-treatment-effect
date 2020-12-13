# data-generation-for-treatment-effect

An implementation of a data generation process for treatment effect estimation
from [Powers, 2018]

[Powers, 2018] Scott Powers, Junyang Qian, Kenneth Jung, Alejandro Schuler, Nigam H. Shah,
Trevor Hastie, Robert Tibshirani. (2018). "Some methods for heterogeneous treatment effect 
estimation in high-dimensions." Statics in Medicine, 37(11): 1767--1787.

# Usage
```python
from generator import generate_data

# Select scenario, the number of data (patients) and the number of features for generating data.
x, t, y, treatment_response, control_response = generate_data(4, 200, 100)

# True CATE can be obtained by substracting control_response from treatment_response.
true_conditional_treatment_effect = treatment_response - control_response
```