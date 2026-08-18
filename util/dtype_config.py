"""Single shared floating-point dtype for every model's dense and sparse arrays.

Every model file imports ``FLOAT_DTYPE`` from here instead of hardcoding its
own precision, so switching precision project-wide is a one-line change.
"""

import numpy as np

FLOAT_DTYPE = np.float32
