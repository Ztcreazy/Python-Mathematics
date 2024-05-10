"""
robust_siso.py

Demonstrate mixed-sensitivity H-infinity design for a SISO plant.

Based on Example 2.11 from Multivariable Feedback Control, Skogestad and Postlethwaite, 1st Edition.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from control import tf, mixsyn, feedback, step_response


