def get_gaussian_kernel(size, sigma=3):
    """
    2-D Gaussian kernel - should give similar results as MATLAB's
    fspecial('gaussian',[shape],[sigma]).

    All credit to: "ali_m" on StackOverflow :)

    Parameters
    ----------
    size : int
        Size of kernel.
    sigma : int, optional
        Standard deviation of Gaussian.
        The default is 3.

    Returns
    -------
    out : h, an array
        A size x size Gaussian kernel.

    See Also
    --------
    cv2.getGaussianKernel() : OpenCV's implementation, returning a 1-D kernel.

    Example
    --------
    >>> get_gaussian_kernel(5, 1)
    >>> array([[ 0.002969,  0.013306,  0.021938,  0.013306,  0.002969],
    >>> [ 0.013306,  0.059634,  0.09832 ,  0.059634,  0.013306],
    >>> [ 0.021938,  0.09832 ,  0.162103,  0.09832 ,  0.021938],
    >>> [ 0.013306,  0.059634,  0.09832 ,  0.059634,  0.013306],
    >>> [ 0.002969,  0.013306,  0.021938,  0.013306,  0.002969]])

    """
    m, n = [(ss - 1.) / 2. for ss in (size, size)]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
