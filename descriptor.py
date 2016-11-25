import math

import config
from buildHist import build_histogram, get_gradient
from sphere_tri import sphere_tri
from utils import dot, normalize

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

    return make_keypoint(video, coord, image_scale, time_scale)

def make_keypoint(video, coord, image_scale, time_scale):
    max_index_val = 0.2
    changed = False

    vec = key_sample_vec(video, coord, image_scale, time_scale)
    vec = normalize(vec)
    vec = [min(max_index_val, x) for x in vec]
    vec = normalize(vec)

    res = [int(512.0 * x) for x in vec]
    assert not any(x < 0 for x in res)

    return [min(255, x) for x in res]

def key_sample_vec(video, coord, image_scale, time_scale):
    #index = KeySample(key, pix)
    index_size = config.index_size

    irow, icol, islice = map(int, coord)
    xy_spacing = image_scale * config.mag_factor
    t_spacing = time_scale * config.mag_factor

    xy_radius = int(1.414 * xy_spacing * (index_size + 1) / 2.0)
    t_radius = int(1.414 * t_spacing * (index_size + 1) / 2.0)
    
    x = range(index_size)
    index = [[[[0]*config.n_faces for _ in x] for _ in x] for _ in x]
    
    for i in range(-xy_radius, xy_radius + 1):
        for j in range(-xy_radius, xy_radius + 1):
            for s in range(-t_radius, t_radius + 1):
                dist_sq = float(i*i + j*j + s*s)
                
                i_i = int(float(i + xy_radius) / (2 * xy_radius/index_size))
                j_i = int(float(j + xy_radius) / (2 * xy_radius/index_size))
                s_i = int(float(s + t_radius) / (2 * t_radius/index_size))

                i_i = min(index_size - 1, i_i)
                j_i = min(index_size - 1, j_i)
                s_i = min(index_size - 1, s_i)

                assert i_i >= 0 and j_i >= 0 and s_i >= 0

                r = irow + i
                c = icol + j
                t = islice + s

                add_sample(index, video, coord, dist_sq, (r, c, t), (i_i, j_i, s_i))
    return index

def add_sample(index, video, coord, dist_sq, rct, ijs_i):
    i, j, s = ijs_i
    r, c, t = rct

    if r < 0 or r >= video.height or c < 0 or c >= video.width or t < 0 \
                or t >= video.total_frames:
        return

    sigma = config.index_sigma * 0.5 * config.index_size
    weight = math.exp(-dist_sq / (2.0 * sigma * sigma))

    grad, vect = get_gradient(video, r, c, t)
    mag = grad * weight

    #Place in index
    corr = [dot(x, vect) for x in centers]
    r = sorted([(x, ind) for ind, x in enumerate(corr)], reverse=True)

    if not config.smooth_flag:
        index[i][j][s][r[0][1]] += mag
    else:
        raise Exception("Not implemented")

if __name__ == '__main__':
    import data
    d = data.DataSet("dataset")
    v = next(d.get_training())
    get_descriptor(v, (60, 40, 5))
