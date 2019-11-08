import pyproj
import numpy as np
from fastkml import kml

def get_bounds_and_imsize_from_kml(kml_file, definition = 0.3, margin = 100):
    with open(kml_file, 'r') as content_file:
        content = content_file.read()
    k = kml.KML()
    k.from_string(content)
    f = list(list(k.features())[0].features())[0].geometry.bounds
    
    wgs84 = pyproj.Proj('+proj=utm +zone=21 +datum=WGS84 +south')
    lon, lat = wgs84([f[0], f[2]], [f[1], f[3]])
    
    bounds = [[lon[0] - margin, lon[1] + margin], [lat[0] - margin, lat[1] + margin]]

    height = int(round((bounds[0][1] - bounds[0][0]) / definition))
    width = int(round((bounds[1][1] - bounds[1][0]) / definition))
    
    return bounds, (height, width)


def spherical_to_image_positions(lons, lats, bounds, im_size):
    y = (lons - bounds[0][0]) / (bounds[0][1] - bounds[0][0])
    x = (lats - bounds[1][0]) / (bounds[1][1] - bounds[1][0])
    y *= im_size[0]
    x *= im_size[1]
    return x, y


    
def fill_holes(data):
    mask = np.isnan(data)
    data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])
    return data