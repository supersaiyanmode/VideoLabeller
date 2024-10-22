
class BaseConfig(object):
    tessellation_levels = 1
    tessellation_radius = 1
    
    two_peak = True

    mag_factor = 3
    index_size = 2
    index_sigma = 5.0

    smooth_flag = True
    smooth_var = 20

    training_points = 50

    save_model = "models/vcnn-74.39-10epochs-64filters-3pool-7conv-128-32"
    
    def __init__(self):
        self.n_faces = 20 * (4 ** self.tessellation_levels)

class CurrentConfig(BaseConfig):
    def __init__(self):
        super(CurrentConfig, self).__init__()

_c = CurrentConfig()
for key in dir(_c):
    globals()[key] = getattr(_c, key)
