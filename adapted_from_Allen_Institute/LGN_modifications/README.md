© 2019 Allen Institute for Brain Science. Models of the Mouse Primary Visual Cortex. 
Files are adapted from: https://www.dropbox.com/sh/w5u31m3hq6u2x5m/AACpYpeWnm6s_qJDpmgrYgP7a?dl=0.

This folder contains the files which were modified from the Allen LGN model
to create the M1 and M2 models for the thesis "Exploring a Model of the Mouse Primary Visual Cortex".
The modifications made to each file are listed below.


**spatialfilter_surr.py:**
```diff
# Replace simple Gaussian filter with Difference-of-Gaussians (DoG):
- on_filter_spatial = ndimage.gaussian_filter(on_filter_spatial, 
-                                             (scaled_sigma_x, scaled_sigma_y), 
-                                             mode='nearest', 
-                                             cval=0)
+ scal_sig = 2.45
+ cent_amp = 1.73
+ scal_amp = 1*cent_amp   
+ filter_cen = ndimage.gaussian_filter(on_filter_spatial, 
+                                      (scaled_sigma_x, scaled_sigma_y), 
+                                      mode='nearest', 
+                                      cval=0)
+ filter_sur = ndimage.gaussian_filter(on_filter_spatial, 
+                                      (scal_sig*scaled_sigma_x, scal_sig*scaled_sigma_y), 
+                                      mode='nearest', 
+                                      cval=0)
+ on_filter_spatial = cent_amp*filter_cen - scal_amp*filter_sur


# Remove second normalisation of already normalised spatial filters 
# to avoid wrong amplitude of DoG filters:
- kernel.normalize()
```


**patch_grating.py:**
```diff
# Apply mask to get circular patches from the full-field grating:
+ circle = (xx - self.col_size / 2) ** 2 + (yy - self.row_size / 2) ** 2
+ mask = (circle <= radius**2)
+ data[~mask] = 0
```


**flashing_spots.py:**
```diff
# Change spatial resolution to 1 pixel per degree:
- physical_spacing = 1. / (float(cpd) * 10)
+ physical_spacing = 1. / (float(0.1) * 10)  


# Set number of flashes and length of each flash:
+ nr_of_flashes = 20
+ seg_length = int((numberFramesNeeded-1)/(nr_of_flashes*2))


# Creating segments of gray and white full-field input:
+ gray_part = np.zeros((seg_length, self.row_size, self.col_size))
+ circle_part = np.ones((seg_length, self.row_size, self.col_size))
+ gray_partb = np.zeros((1, self.row_size, self.col_size))


# Apply mask to get spot of specified size:
+ circle = (xx - self.col_size / 2) ** 2 + (yy - self.row_size / 2) ** 2       
+ mask = (circle <= radius**2)
+ circle_part[~mask] = 0


# Create movie sequence:
+ segment = np.concatenate((circle_part, gray_part), axis=0)
+ data = np.tile(segment,reps =[nr_of_flashes,1,1])
+ data = np.concatenate((data, gray_partb), axis=0)
```


**lgn_functions.py:**
```diff
# Import adapted nwb:
- import isee_engine.nwb as nwb
+ from lgnmodel.nwb_copy import *


# Import DoG spatial filters instead of single Gaussians
- from lgnmodel.spatialfilter import GaussianSpatialFilter, ArrayFilter
+ from lgnmodel.spatialfilter_surr import GaussianSpatialFilter, ArrayFilter


# Import visual stimuli, e.g. patch grating:
- from lgnmodel.movie import GratingMovie, FullFieldFlashMovie
+ from lgnmodel.patch_grating import GratingMovie, FullFieldFlashMovie


# Include "radius" as a parameter for visual stimuli:
- def calculate_firing_rate(LGN, stimulus, output_file_name, duration, 
-                           gray_screen,  cpd, TF, direction, contrast):
+ def calculate_firing_rate(LGN, stimulus, output_file_name, duration, 
+                           gray_screen,  cpd, TF, direction, contrast, radius):
- movie_to_show = GratingMovie(120, 240).create_movie(t_min = 0, 
-                                                     t_max = duration, 
-                                                     gray_screen_dur = gray_screen, 
-                                                     cpd = cpd, 
-                                                     temporal_f = TF, 
-                                                     theta = direction, 
-                                                     contrast = contrast/100.)
+ movie_to_show = GratingMovie(120, 240).create_movie(t_min = 0, 
+                                                     t_max = duration, 
+                                                     gray_screen_dur = gray_screen, 
+                                                     cpd = cpd, 
+                                                     temporal_f = TF, 
+                                                     theta = direction,
+                                                     contrast = contrast/100., 
+                                                     radius = radius)
```


**lgn_functions_normalise_rates.py:**
```diff
# Modifications are in addition to those listed above for "lgn_functions.py"
# Find mean of all firing rates per time:
+ net_mean = np.mean(firing_rates[1:,:],axis=0)   
+ net_mean_plus_one = [x+1 for x in np.array(net_mean)]
    
# Normalise rates, scale to keep spontaneous rate unchanged:
+ for counter, node in enumerate(LGN.nodes()):
+     firing_rates[counter+1,:] = np.divide(np.array(firing_rates[counter+1,:]), net_mean_plus_one)*4.84 
```


**simulate_drifting_gratings.py:**
```diff
# Include "radius" as a parameter for visual stimuli:
- calculate_firing_rate(LGN, stimulus, output_file_name, duration, 
-                       gray_screen, cpd, TF, direction, contrast)
+ calculate_firing_rate(LGN, stimulus, output_file_name, duration, 
+                       gray_screen, cpd, TF, direction, contrast, radius)
```


**nwb_copy.py:**
```diff
# Make nwb script compatible with Python 3:
+ dim_string = str(dimension)
+ print('generating spikes')
+ unit_string = str(unit)
+ dim_string_conv = dim_string.encode('UTF-8')
+ unit_string_conv = unit_string.encode('UTF-8')
+ dataset.attrs.create('dimension', dim_string_conv)
+ dataset.attrs.create('unit', unit_string_conv)
```
