# frappy

**frappy** (**fr**actal **a**nd **p**hase space analysis for **py**thon) is a Python package implementing a non subjective algorithm for calculating correlation dimensions from data.

This package currently contains functions for:

- Converting a time series into a uniform deviate through a rank transformation
- Embedding time series into a d-dimensional vector space
- Calculate correlation dimension at different embedding dimensions
- Calculate saturated correlation dimension and minimum embedding dimension using curve fitting
- Module for generating surrogate data using
  - Random shuffling
  - Preserving autocorrelation at lag-1 (AR(1) surrogates) 


## Installation
You can install frappy from PyPi using  
&nbsp;&nbsp;&nbsp;&nbsp;**pip install frappy**

## Acknowledgements

This work is an python implementation of the algorithm described in Harikrishnan, K. P., Misra, R., Ambika, G., & Kembhavi, A. K. (2006). A non-subjective approach to the GP algorithm for analysing noisy time series. Physica D: Nonlinear Phenomena, 215(2), 137-145. 
