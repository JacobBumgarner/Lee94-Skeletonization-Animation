import os
import pyvista as pv
import numpy as np
from skimage.io import imread, imsave
from scipy.ndimage import label, sum
import skeletonize
from time import sleep

def load_volume(filename, objects=3, skeletonize_volume=False):
    """Given a filename, return either the loaded volume or an array 
    volumes that progress to the skeletonized input volume
    
    Parameters 
    ----------
    filename: str
        Path to the volume to be loaded/skeletonized
    
    objects: int, optional
    
    skeletonize: bool, optional
        If true, creates list of intermediate volumes at each iteration
        of the Lee '94 skeletonization
        
    Returns
    ------- 
    Original volume or list of skeletonized volumes
    
    """
    
    if os.path.splitext(filename)[1] == '.npy':
        volume = np.load('labeled.npy')
        return volume
    
    # Load the volume, get the three largest components
    volume = imread(filename)
    volume = skeletonize.binarize(volume)
    labeled, labels = label(volume)
    sizes = sum(volume, labeled, range(labels+1))
    mask = np.zeros(sizes.shape).astype(np.bool_)
    for i in range(objects, 0, -1):
        mask += sizes == sizes[np.argsort(sizes)[-i]]
    volume = mask[labeled]
    volume = np.pad(volume, 1).astype(np.uint8)
   
    # Return original volume if testing
    if not skeletonize_volume:
        return volume
    # Otherwise return list of skeleton series
    else: 
        volume = skeletonize.skeletonize(volume, return_labeled=True,
                                         verbose=True)
        return volume

def get_basis(camera_position):
    """Given the plotter camera position, create a transformation matrix
    to convert the points relative to the camera back to the plotter coordinates
    
    Parameters
    ----------
    camera_position: list
        2D list or matrix containing the camera position, focal point,
        and camera viewup vector

    Returns
    -------
    np.array
        Camera basis transformation matrix
    """
    pos = np.array([p for p in camera_position]) # Convert position to np.array
    b1 = pos[0] - pos[1] # Get b1 pointing from focal point to camera
    b1 /= np.linalg.norm(b1)
    b3 = pos[2] # Viewup
    b2 = np.cross(b1, b3) # Get orthogonal b2 using b1,b3 cross product
    b2 /= np.linalg.norm(b2)
    b2 -= 2 * b2
    return np.array([b1, b2, b3])

def load_path(plotter, basis, points=30, degrees=90, show_path=False,
              reversed=False):
    """Create an orbital path around a focal point
    
    Parameters
    ----------
    plotter: pyvista.Plotter() object
    
    basis: np.array
        Camera basis transformation matrix created with get_basis()
        
    points: int, optional
        Number of points for the camera to fly along during the movie
    
    degrees: int, optional
        Angle of the camera path to fly along from the starting position
        
    show_path: bool, optional
        Plot spheres at each point along the resulting camera path
        
    Returns
    -------
    np.array
        3D array containing the plotter.camera_position for each point along 
        the created path
    
    """
    # Convert the plotter position to an array
    cam_pos = np.array([p for p in plotter.camera_position])
    
    # Load the radians along the desired path
    degrees = np.linspace(0, degrees/360 * 2 * np.pi, num=points)
    norm_raw_points = np.zeros((points, 3)) # normalized raw points
    for i in range(points):
        norm_raw_points[i] = [np.cos(degrees[i]), np.sin(degrees[i]), 0]
        
    # Unnormalize the path points, convert them back into plotter coordinates
    raw_points = norm_raw_points * np.linalg.norm(cam_pos[1] - cam_pos[0])
    path_points = np.dot(raw_points, basis) + cam_pos[1]
    
    if show_path: # Show spheres at each point
        for i in range(path_points.shape[0]):
            sphere = pv.Sphere(radius=1, center=path_points[i])
            plotter.add_mesh(sphere)
    
    # Create the plotter path with the focal point and viewup
    viewup = np.repeat([cam_pos[2]], points, axis=0)
    focal_point = np.repeat([cam_pos[1]], points, axis=0)
    camera_path = np.stack([path_points, focal_point, viewup], axis=1) 
    
    if reversed:
        camera_path = np.flip(camera_path, axis=0)
    return camera_path

def prep_plotter(position=None, add_widget=False, offscreen=False,
                 background='black'):
    """Create a plotter object
    
    Parameters
    ----------
    position: list, optional
        2D list containing the plotter position, focal point, and viewup
    
    add_widget: bool, optional
        Adds a checkbox widget to print the current plotter position
        Useful when trying to find the right position for the camera
        
    offscreen: bool, optional
        Loads the plotter to run the animation off screen
        
    background: str, optional
        'black', 'gray', 'white', etc. String-based single word colors
        
    Returns 
    -------
    pyvista.Plotter object
    
    """
    p = pv.Plotter(window_size=(2000,2000), off_screen=offscreen)
    
    p.set_background(background)
    
    # Add plotter position widget if needed
    def print_position(state):
        cam_pos = p.camera_position
        for i, pos in enumerate(cam_pos):
            pos = f"({pos[0]:0.2f}, {pos[1]:0.2f}, {pos[2]:0.2f})"
            if i < 2:
                pos += ','
            print (pos)
    if add_widget:
        p.add_checkbox_button_widget(print_position, value=False)
    
    if position:
        p.camera_position = position
    
    return p

def get_volume_pd(volume, return_removed=False, step=0):
    """Returns a pyvista PolyData object from an input numpy volume
    
    Parameters
    ----------
    volume: np.array
    
    Returns:
    pv.PolyData
        PolyData object glyphed to show the voxels of the input volume
    """
    if return_removed:
        points = np.vstack(np.where(volume == step)).T
    else:
        points = np.vstack(np.nonzero(volume)).T
    pd = pv.PolyData(points).glyph(geom=pv.Cube())
    return pd


def add_mesh(plotter, volume, removal_id=None, opacity=1):
    """ Adds a volume or volume difference to the plotter
    
    Parameters
    ----------
    plotter: pyvista.Plotter
    
    volume: 3D np.array
    
    removal_id: int, optional
        The id of the values in the array that are to be removed
    
    opacity: float, optional
        Sets the opacity of the differenced volume, if loaded
        
    decay_color: str, optional
        'red', 'orange', 'green', etc. String-based single word colors
        
    Returns
    -------
    None or vtk.Actor
        Returns the actor of the loaded object if the difference is plotted
    
    """
    if removal_id is None:
        volume_pd = get_volume_pd(volume)
        plotter.add_mesh(volume_pd, show_edges=True,)
                        #  diffuse=0.5, specular=0, ambient=0.5)
        return
    else:
        removal_volume = volume == removal_id
        if not np.any(removal_volume):
            return None
        volume_pd = get_volume_pd(volume == removal_id)
        actor = plotter.add_mesh(volume_pd, show_edges=True,
                                 color=decay_color)
        return actor
    
def add_cycle_text(plotter, cycle, iteration):
    plotter.add_text(f"Cycle: {cycle+1}, Iteration: {iteration+1}",
                     name='cycle', position='upper_edge')
    return
    
def load_opacity_decay(frames, decay_rate):
    """Given a number of frames and a decay rate, return an array with a 
    logarithmic decay of opacity at the specified framerate of decay"""
    
    decay_array = np.ones(frames)
    
    for i in range(int(frames*decay_rate)):
        # decay_array[i] = 1 - np.log(i+1) / np.log(frames*decay_rate)
        decay_array[-(i+1)] = i / frames
    
    return decay_array

def test_animation(filename, plotter_position=None, volume_objects=3,
                   save_npy=False, npy_path='labeled.npy'):
    """A minimal function to visualize the dataset for plotter camera position
    configuration and creation of a .npy of the skeletonized volume
    
    Parameters
    ----------
    filename: str
        File path of the volume to be loaded
        
    save_npy: bool, optional
        Determines whether to save a .npy copy of the skeletonized volume.
        Can speed reduce animation runtime if running multiple animations of 
        the same volume
        
    npy_path: str, optional
        Save path for the the .npy volume
        
    """
    
    
    volume = load_volume(filename, volume_objects,
                         skeletonize_volume=save_npy)
    
    if save_npy:
        np.save(npy_path, volume)
    
    plotter = prep_plotter(plotter_position, add_widget=True)
    add_mesh(plotter, volume)
    plotter.add_axes()
    plotter.show()
    return

def animate_skeletonization(filename, movie_path, volume_objects=3, 
                            plotter_position=None,
                            frame_rate=60, iteration_frames=30, decay_rate=0.4, 
                            orbit=False, orbit_angle=90, 
                            bounce=False, rebuild=False,
                            offscreen=False,
                            background='black', decay_color='red'):
    #region
    """Animates the skeletoniation of a volume
    
    Parameters
    ----------
    filename: str
        skimage.io.imread compatible 3D filetype
    
    movie_path: str
        .mp4 File path to save the movie
    
    volume_objects: int, optional
        The n largest objects to keep in the volume
        
    plotter_position: list, optional
        2D list containing the plotter position, focal point, and viewup
    
    frame_rate: int, optional
        The frame rate of the animation
        
    iteration_frames: int, optional
	    The number of frames that the movie will take

    decay_rate: float, 0-1, optional
        Percentage of the iteration_frames that the decay occurs over

    orbit: bool, optional

    orbit_angle: int or float, optional
        Angle of the orbit that the camera will follow during the animation
        
    bounce: bool, optional
        If True, reruns the animation in reverse. Forces a rebuild
                            
    rebuild: bool, optional
        If True, rebuilds the volume at the end of the animation
        
    offscreen: bool, optional
        If True, renders the animation off screen
        
    background: str, optional
        'black', 'gray', 'white', etc. String-based single word colors
        
    decay_color: str, optinoal
        'red', 'orange', 'green', etc. String-based single word colors
    """
    #endregion
    
    # Create the plotter 
    plotter = prep_plotter(position=plotter_position, offscreen=offscreen,
                           background=background)

    volume = load_volume(filename, objects=volume_objects,
                         skeletonize_volume=True)
    vc = volume.copy()
    
    frames = (volume.max()-6) * iteration_frames
    if orbit:
        basis = get_basis(plotter.camera_position)
        path = load_path(plotter, basis, frames, orbit_angle)
        
    # Add the first volume, start the movie
    plotter.show(auto_close=False)
    plotter.open_movie(movie_path, framerate=frame_rate)
    plotter.clear()
    add_mesh(plotter, volume)
    for _ in range(10):
        plotter.write_frame()
    
    # Prep the opacity array
    opacity_decay = load_opacity_decay(iteration_frames, decay_rate)

    for o in range(1 + bounce):
        if o == 1:
            path = np.flip(path, axis=0)

        frame = 0
        for i in range(int(frames/iteration_frames/6)):
            for j in range(6):
                plotter.clear()
                
                # First add the removal volume
                actor = add_mesh(plotter, volume,
                                removal_id=(i*6+j)+1)
                
                # Then update the new volume and add it
                volume[volume==(i*6+j)+1] = 0
                add_mesh(plotter, volume)
                
                if actor:
                    property = actor.GetProperty()
                
                
                add_cycle_text(plotter, i, j)
                for k in range(iteration_frames):
                    print (f"Frame: {frame+1}/{frames}", end='\r')
                    
                    if actor: # Reduce the opacity of the actor
                        property.SetOpacity(opacity_decay[k])

                    if orbit: # Move the camera if orbiting
                        plotter.camera_position = path[frame]
                    frame += 1
                    plotter.write_frame()
                    
        if rebuild or bounce: # Rebuild the skeleton
            for i in range(vc.max()-6, 0, -1):
                plotter.clear()
                add_mesh(plotter, vc >= i)
                for _ in range(6):
                    plotter.write_frame()
            for _ in range(10): # Write some still frames
                plotter.write_frame()
            plotter.clear()
            volume = vc.copy() # Reload the labeled skeleton from the copy
    plotter.close()
    return
    
if __name__ == "__main__":
    filename = "/Users/jacobbumgarner/Documents/GitHub/Lee94-Skeletonization-Animation/Slice 11.nii"
    # filename = "labeled.npy"
    movie_path = "/Users/jacobbumgarner/Desktop/test.mp4"
    
    save_skeleton_npy = False
    npy_path = "labeled.npy"
    
    plotter_position = [
(287.79, -12.34, 105.38),
(18.50, 50.50, 50.50),
(-0.07, 0.46, 0.88)
# (279.58, -50.43, 146.56),
# (25.50, 50.50, 50.50),
# (0.02, 0.72, 0.70)
                        ]
    
    volume_objects = 1
    frame_rate = 60
    iteration_frames = 45
    decay_rate = 0.5
    orbit = True
    orbit_angle = 360
    bounce = False
    rebuild = True
    offscreen = True
    background = 'black'
    decay_color = 'red'
    
    # test_animation(filename, plotter_position,
    #                volume_objects, save_skeleton_npy, npy_path)
    
    animate_skeletonization(filename, movie_path, volume_objects,
                            plotter_position,
                            frame_rate, iteration_frames, decay_rate, 
                            orbit, orbit_angle, 
                            bounce, rebuild,
                            offscreen,
                            background, decay_color)