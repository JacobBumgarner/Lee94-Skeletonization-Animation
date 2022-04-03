## Lee94 Skeletonization Animation
I've always wanted to visualize the stages of volume skeletonization. I've done that here using PyVista!

I decided to create a bit of code that easily allows for visualization of the stages of skeletonization using the Lee et. al 1994 medial axis thinning algorithm.

This skeletonization algorithm is the one used in FIJI, skimage, and VesselVio. I rewrote the algorithm using numpy & numba for [VesselVio](https://github.com/JacobBumgarner/VesselVio), so I've just copied it over here.

https://user-images.githubusercontent.com/70919881/161442222-8bb4f95b-4365-4c7d-8271-8fb315f2dd7a.mp4
