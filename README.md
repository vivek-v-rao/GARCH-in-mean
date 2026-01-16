# GARCH-in-mean
Fit GARCH-in-mean models to daily asset class returns using the Python arch package.
`python xgarch_m.py' gives the results below. Looking at the archm_t column, which has the t-statistic of
the GARCH-in-mean (GIM) effect, the positive values for SPY EFA EEM IEF LQD mean that higher conditional variances
predict higher returns. Only TLT has a negative GIM effect. It is a but surprising that TLT and IEF have GIM effects
of the opposite sign.
```
prices file: prices.csv
date range: 2003-04-14 to 2026-01-16
log returns? True
return scaling: 100.0

series model  nobs     archm   archm_t        mu     omega    alpha1    gamma1     beta1         nu status
   SPY garch  5727  0.033414  2.534280  0.076521  0.019250  0.136815       NaN  0.854431   5.665416     ok
   SPY   gjr  5727  0.026202  3.016073  0.055418  0.025252  0.000000  0.232839  0.855337   6.074321     ok
   EFA garch  5727  0.027573  2.147784  0.051053  0.020768  0.104253       NaN  0.883640   6.873794     ok
   EFA   gjr  5727  0.020552  1.984703  0.037270  0.024818  0.020149  0.131883  0.890621   7.357146     ok
   EEM garch  5727  0.014915  1.216590  0.059911  0.044573  0.101266       NaN  0.879361   8.972271     ok
   EEM   gjr  5727  0.016707  1.747678  0.029390  0.051334  0.027606  0.119184  0.884893   9.549006     ok
   TLT garch  5727 -0.013954 -0.614157  0.028026  0.004563  0.049002       NaN  0.945718  15.020335     ok
   TLT   gjr  5727 -0.010964 -0.491472  0.028929  0.004409  0.055248 -0.012805  0.946331  15.228610     ok
   IEF garch  5727  0.037842  0.712503  0.007473  0.001048  0.048271       NaN  0.946305  12.155254     ok
   IEF   gjr  5727  0.038075  0.717233  0.007564  0.001049  0.048946 -0.001289  0.946278  12.153513     ok
   LQD garch  5727  0.040082  1.837780  0.021912  0.002505  0.071106       NaN  0.915238   8.427796     ok
   LQD   gjr  5727  0.023150  0.992121  0.021430  0.002572  0.045102  0.039219  0.919205   8.639879     ok
```

