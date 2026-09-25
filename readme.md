MATLAB implementations of adaptive filtering techniques. 
The codebase explores three practical applications of stochastic signal models and adaptive algorithms:
- Reference-Free Noise Cancellation: Utilizes the Normalized Least Mean Squares (NLMS) algorithm to denoise ECG signals without a secondary reference, analyzing the impact of optimal delay ($k_0$) and filter order ($p$) on signal reconstruction.
- System Identification: Approximates an unknown Linear Time-Invariant (LTI) system through Steepest Descent Adaptive Filtering (SDAF) and Recursive Least Squares (RLS) algorithms, evaluating learning curves and the effect of the forgetting factor ($\lambda$) on steady-state accuracy.
- Audio Signal Compression: Applies auto-regressive (AR) modeling via an NLMS filter for predictive coding, testing optimal filter lengths and block-based training distributions ($M$ portions and $N$ training percentage) to maximize compression gain.

Developed for the Introduction to Statistical Signal Processing course
