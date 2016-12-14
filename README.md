# VideoLabeller

##Dataset credits:
"Recognizing Human Actions: A Local SVM Approach",
Christian Schuldt, Ivan Laptev and Barbara Caputo; in Proc. ICPR'04, Cambridge, UK

##Important Links:
 - Repository: https://github.com/supersaiyanmode/VideoLabeller
 - `dataset/` folder: https://github.com/supersaiyanmode/VideoLabeller/tree/master/dataset
 - `models/vcnn-74.39-10epochs-64filters-3pool-7conv-128-32`: https://github.com/supersaiyanmode/VideoLabeller/tree/master/models/vcnn-74.39-10epochs-64filters-3pool-7conv-128-32
 - `models/sift3d`: https://github.com/supersaiyanmode/VideoLabeller/tree/master/models/sift3d
 - `cache.txt`: https://drive.google.com/file/d/0B1MXRMDMYoj1aEJRNjJKRDk2dVk/view?usp=sharing

##Changing the Config
The file config.py can be edited to change a few parameters. They are described below:

 - `tessellation_levels`: Number of tessellations to perform. Default: 1
 - `tessellation_radius`: Radius of the sphere. Default: 1
 - `two_peak`: Perform Two-Peak testing while rejecting sample points. Default: True
 - `smooth_flag`: Smooth histogram creation. See: descriptor.py:33

 - `save_model`: Path of *folder* to read model from (if present) or write to. Use `None` to prevent model reading/writing.


##Before running:
 - Check presence of `dataset/` folder. If not present download from the repository (linked above).
 - Check presence of `cache.txt` (optional). Program extract the feature vectors from the video and store it here. This file is used (if present) to read the vectors back instead of re-extracting. If not present download cache.zip (linked above).
 - Check presence of `models/vcnn-74.39-10epochs-64filters-3pool-7conv-128-32`. This uses the trained CNN model that gives about 74% accuracy. If not found please download from the repository.
 - Check presence of `models/sift3d`. This uses the trained SIFT 3D model. If not found please download from the repository.

##Running:

###Running SIFT 3D:

    $ python2 main.py sift

###Running Video CNN:

    $ THEANO_FLAGS=exception_verbosity=high,device=gpu,floatX=float32,optimizer=None python2 main.py
