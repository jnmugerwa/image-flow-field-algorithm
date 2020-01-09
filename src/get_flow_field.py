def get_flow_field(orient_image, original_image, sampling_interval=30):
    """
    Plots the flow field.

    Parameters
    ----------
    orient_image : array
        Array with flow directions (in radians) at each pixel.
    original_image : array
        The image from which orient_image was derived.
    sampling_interval : int, optional
        We will take a sample(direction from orient_image)
        every "sampling_interval" steps.
        The default is 30; fairly granular.

    Returns
    -------
    Nothing; void.

    See Also
    --------
    ...

    Example
    --------
    See examples in "results" directory.

    """
    assert isinstance(sampling_interval, int), "Please give an integer sampling size."

    if sampling_interval < 10:
        print("You requested a very granular flow field; this may take" +
              " more time than usual")

    num_rows, num_cols = orient_image.shape

    # Subsample the orientation data according to the specified spacing.
    orient_im_samples = orient_image[::sampling_interval, ::sampling_interval]

    x, y = np.meshgrid(range(0, num_cols, sampling_interval), range(0, num_rows, sampling_interval))

    u = np.cos(orient_im_samples)
    v = np.sin(orient_im_samples)

    fig, ax = plt.subplots()

    # OpenCV reads images as BGR instead of RGB. This re-converts to RGB.
    ax.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

    ax.quiver(x, y, u, v, color='r')

    fig.savefig('flow_field_image.png')
