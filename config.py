
class BaseConfig(object):
    tessellation_levels = 1
    tessellation_radius = 1
    
    two_peak = True

    mag_factor = 3
    index_size = 2
    index_sigma = 5.0

    smooth_flag = False


    
    def __init__(self):
        self.n_faces = 20 * (4 ** self.tessellation_levels)

class CurrentConfig(BaseConfig):
    def __init__(self):
        super(CurrentConfig, self).__init__()

_c = CurrentConfig()
for key in dir(_c):
    globals()[key] = getattr(_c, key)
