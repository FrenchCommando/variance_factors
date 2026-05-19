calibration of parameters for Bergomi 2-factor using SPX data on a rolling window

export project from `~\theta-options\variance_kalman` (don't copy the names, there is nothing Kalman actually)

assume relevant data is already cached (SPXW 1DTE fwds - logswaps for SPX and SPXW)

pull all the code so that we can compute the parameter calibrations on a rolling window, and related diagnostics

outputs are similar to the original repo: images in png and data in feather (we are not saving the feather in the repo - let's gitignore them)

I did a similar 'pull the scripts in a clean repo' for '~/2piece', so I hope this is going to be seamless too - one difference is that this is more more data-focused, although we decide not to put the data directly in the repo, so it might be harder

The point of the repo is quite different too, we don't want to enable the user to reproduce the results, it's just about giving a simple overview of what a very simple methodology yields in terms of results.


# Code

- keep it python 3.14 with a local venv
- graphs are matplotlib


# Contents

The code should be able to regenerate all the graphs and results without calling code from the original repo (if the data is setup correctly).
Data folder location should be clearly defined. How the data is generated is not documented. A new user is not meant to be able to cleanly access correct inputs - however, if the user already has similar data, it should be easy to convert to the right format.

Relevant results are mainly images/graphs. Parameter values are not saved.


# Extension

currently using SPX expiries with a lot of trimming, we will want to try it on SPXW
