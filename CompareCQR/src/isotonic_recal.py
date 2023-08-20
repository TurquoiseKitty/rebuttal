# code from  https://github.com/uncertainty-toolbox/uncertainty-toolbox

import numpy as np
from sklearn.isotonic import IsotonicRegression
from numpy.linalg import norm as npnorm



def iso_recal(
    exp_props: np.ndarray,
    obs_props: np.ndarray,
) -> IsotonicRegression:
    """Recalibration algorithm based on isotonic regression.
    Fits and outputs an isotonic recalibration model that maps observed
    probabilities to expected probabilities. This mapping provides
    the necessary adjustments to produce better calibrated outputs.
    Args:
        exp_props: 1D array of expected probabilities (values must span [0, 1]).
        obs_props: 1D array of observed probabilities.
    Returns:
        An sklearn IsotonicRegression recalibration model.
    """
    # Flatten
    exp_props = exp_props.flatten()
    obs_props = obs_props.flatten()

    # quants should be a rising list
    assert npnorm(np.sort(exp_props) - exp_props) < 1E-6

    iso_model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    

    try:
        iso_model = iso_model.fit(obs_props, exp_props)
    except Exception:
        raise RuntimeError("Failed to fit isotonic regression model")

    return iso_model
