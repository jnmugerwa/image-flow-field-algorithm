# Image-Flow-Field-Algorithm
An important feature of a scene is the orientation of the objects it holds. For example, one theory of cell movement is that cells move along the direction of fibers in their environment. You can see why, given a microscope image of a cell and its surroundings, it would be valuable to know the orientation of the cell and the fibers around it (i.e. "are they collinear?").

This algorithm attempts to extract information about the high-level orientation of the scene by computing pixel-wise orientation values for the image.

Specifically, we:

1. Compute the gradient field of the image
2. Estimate the principal axes of the ellipsoid that will contain the low-dimensional projections of the gradient data.
3. Calculate a pixel-wise angle measurement between the principal axes and the gradients. 
