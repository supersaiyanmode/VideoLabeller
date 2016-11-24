import config
from buildHist import build_histogram
from sphere_tri import sphere_tri
from utils import dot

sphere = sphere_tri(radius=config.tessellation_radius,
                                    maxlevel=config.tessellation_levels)

vertices, faces, centers = sphere

def get_descriptor(video, coord, image_scale=1, time_scale=1):
    rad = image_scale * 3
    histogram = build_histogram(video, coord, rad, sphere)

    res = sorted([(x, i) for i, x in enumerate(histogram)], reverse=True)
    
    one_two = dot(centers[res[0][1]], centers[res[1][1]])
    one_three = dot(centers[res[0][1]], centers[res[2][1]])

    if config.two_peak and one_two > 0.9 and one_three > 0.9:
        return None

    print res
    return make_keypoint(video, coord, image_scale, time_scale)

def make_keypoint(video, coord, image_scale, time_scale):
    return "keypoint"


if __name__ == '__main__':
    import data
    d = data.DataSet("dataset")
    v = next(d.get_training())
    get_descriptor(v, (60, 40, 5))
