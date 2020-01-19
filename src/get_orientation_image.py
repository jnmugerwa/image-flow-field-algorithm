import numpy as np
from scipy import signal
import cv2


def get_orientation_image(image_path, smooth_sigma=3, sum_sigma=3):
    """
    Returns the orientation image of an image.

    Calculates the flow direction (angle of maximal variation) at each
    pixel. These angles can be used to create vectors within a mesh grid,
    creating a visual of the flow direction field of the image.

    Resources:
        "ANALYZING ORIENTED PATTERNS" (Kass, Witkin)
        "Directional Field Computation for Fingerprints Based on the Principal
            Component Analysis of Local Gradients" (Bazen, Gerez)
        "Scribe Notes: Multivariate Gaussians and PCA" (Partridge)
        An ungodly amount of:
            StackOverflow
            numpy, scipy, etc. documentation
            Wikipedia
            Random slidedecks and pdfs from computer vision courses

    Parameters
    ----------
    image_path : string
        The filepath to an image.
    smooth_sigma : int, optional
        Standard deviation of the initial Gaussian filter.
        The default is 3.
    smooth_sigma : int, optional
        Standard deviation of the Gaussian kernel used in the weighed summation.
        The default is 3.

    Returns
    -------
    orient_image : 2-D array
        An image/array, identically-sized to the input image/array. At each
        pixel is the flow-direction of the image at that pixel.

    See Also
    --------
    getOrientation() in OpenCV's PCA tutorial:
        https://docs.opencv.org/3.4/d1/dee/tutorial_introduction_to_pca.html

    Example
    --------
    >>> image_path = r"C:\Users\natha\MULTI-SCALE MECHANICS\Test Images\Dendrites.jpg"
    >>> im = cv2.imread(image_path)
    >>> orientim = get_orientation_image(image_path)
    >>> plot_flow_field(orient_im, im)

    """
    # Make sure to cast entries as float, else truncation errors will arise.
    im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype('float')

    # Smooths image using a Gaussian kernel.
    smooth_im = cv2.GaussianBlur(im, (0, 0), smooth_sigma)

    gradients = np.gradient(smooth_im)

    Gx, Gy = np.asarray(gradients)

    # See Kass Paper. The smoothed versions of these will be used to
    # ...calculate flow angles (parllel to Principal Axis 2) at each pixel.
    J1 = 2 * Gx * Gy
    J2 = np.square(Gx) - np.square(Gy)

    # Get Gaussian Kernel.
    size = np.floor(6 * sum_sigma)
    gauss_kernel = get_gaussian_kernel(size, sum_sigma)

    # Weighed summation using Gaussian kernel.
    J1star = signal.fftconvolve(J1, np.rot90(gauss_kernel, 2), mode='same')
    J2star = signal.fftconvolve(J2, np.rot90(gauss_kernel, 2), mode='same')

    # Solving for the flow direction at each pixel.
    orient_image = np.arctan2(J1star, J2star) / 2

    return orient_image
