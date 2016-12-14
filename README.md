# VideoLabeller

##Dataset credits:
"Recognizing Human Actions: A Local SVM Approach",
Christian Schuldt, Ivan Laptev and Barbara Caputo; in Proc. ICPR'04, Cambridge, UK

##Changing the Config
The file config.py can be edited to change a few parameters. They are described below:

 - `tessellation_levels`: Number of tessellations to perform. Default: 1
 - `tessellation_radius`: Radius of the sphere. Default: 1
 - `two_peak`: Perform Two-Peak testing while rejecting sample points. Default: True
 - `smooth_flag`: Smooth histogram creation. See: descriptor.py:33

 - `save_model`: Path of *folder* to read model from (if present) or write to. Use `None` to prevent model reading/writing.


##Running:

###Running SIFT 3D:

    $ python2 main.py sift

###Running Video CNN:

    $ THEANO_FLAGS=exception_verbosity=high,device=gpu,floatX=float32,optimizer=None python2 main.py
