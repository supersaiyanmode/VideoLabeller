import config
from utils import *

def build_histogram(video, coord, rad, sphere):
    vertices, faces, centers = sphere
    xi, yi, zi = map(int, coord)
    rows, cols, slices = video.shape()

    histogram = [0]*config.n_faces

    for r in range(xi - rad, xi + rad + 1):
        for c in range(yi - rad, yi + rad + 1):
            for s in range(zi - rad, zi + rad + 1):
                if 0 < r < rows - 1 and 0 < c < cols - 1 \
                            and 0 < s < slices - 1:
                    mag, vect = get_gradient(video, r, c, s)
                    corr = [dot(x, vect) for x in centers]
                    r = sorted([(x, i) for i, x in enumerate(corr)], reverse=True)
                    histogram[r[0][1]] += mag
    return histogram


def get_gradient(video, r, c, s):
    rows, cols, slices = video.shape()

    if r < 0: r = 0
    if c < 0: c = 0
    if s < 0: s = 0

    if r >= rows: r = rows - 1
    if c >= cols: c = cols - 1
    if s >= slices: s = slices - 1

    if c == 0:
        grad_x = 2.0 * (int(video[s][r,c+1]) - int(video[s][r,c]))
    elif c == cols - 1:
        grad_x = 2.0 * (int(video[s][r,c]) - int(video[s][r,c-1]))
    else:
        grad_x = int(video[s][r, c+1]) - int(video[s][r, c-1])


    if r == 0:
        grad_y = 2.0 * (int(video[s][r,c]) - int(video[s][r+1,c]))
    elif r == rows - 1:
        grad_y = 2.0 * (int(video[s][r-1,c]) - int(video[s][r,c]))
    else:
        grad_y = int(video[s][r-1,c]) - int(video[s][r+1,c])

    if s == 0:
        grad_z = 2.0 * (int(video[s+1][r,c]) - int(video[s][r,c]))
    elif s == slices - 1:
        grad_z = 2.0 * (int(video[s][r,c]) - int(video[s-1][r,c]))
    else:
        grad_z = int(video[s+1][r,c]) - int(video[s-1][r,c])

    grad = float(grad_x), float(grad_y), float(grad_z)

    mag = math.sqrt(dot(grad, grad))

    if mag < 10**-6:
        return mag, (1, 0, 0)

    return mag, vec_divide(grad, mag)
