"""Gaussian-pupil camera model, adapted from the attributed predecessor lab."""
import numpy as np
from scipy.special import ndtr
from scipy.optimize import least_squares

WAVELENGTH_UM, FOCAL_UM, PITCH_UM, PUPIL_RADIUS_UM = .550, 16000., 4., 1000.
# Pupil amplitude exp(-rho^2 / a^2): focal intensity SD = lambda*f/(2*pi*a).
SIGMA = WAVELENGTH_UM*FOCAL_UM/(2*np.pi*PUPIL_RADIUS_UM*PITCH_UM)
PIXELS = np.arange(-16., 17.)
FLUX, BACKGROUND, READ_SD = 10000., 20., 2.
ORIGIN, VELOCITY = .23, .8  # px; px/reference-clock second (assumed independently known)


def image(center, sigma=SIGMA):
    """One-dimensional marginal of a separable 2-D PSF; exact pixel integration."""
    if sigma <= 0:
        raise ValueError('PSF sigma must be positive')
    return BACKGROUND + FLUX*(ndtr((PIXELS+.5-center)/sigma)
                              - ndtr((PIXELS-.5-center)/sigma))


def derivative(center, sigma=SIGMA):
    lo, hi = (PIXELS-.5-center)/sigma, (PIXELS+.5-center)/sigma
    return FLUX/(sigma*np.sqrt(2*np.pi))*(np.exp(-lo*lo/2)-np.exp(-hi*hi/2))


def fit_center(data):
    data = np.asarray(data, dtype=float)
    if data.shape != PIXELS.shape or not np.all(np.isfinite(data)):
        raise ValueError('Expected one finite value per pixel')
    signal = np.maximum(data-BACKGROUND, 0)
    if signal.sum() <= 0:
        raise ValueError('No positive signal')
    initial = np.clip(PIXELS@signal/signal.sum(), -5.9, 5.9)
    fit = least_squares(lambda x: (image(x[0])-data)/READ_SD, [initial],
        jac=lambda x: derivative(x[0])[:,None]/READ_SD, bounds=(-6.,6.),
        xtol=1e-12, ftol=1e-12, gtol=1e-12)
    if not fit.success or np.any(fit.active_mask):
        raise RuntimeError('Center fit failed or reached ROI bound')
    return fit.x[0]
