import numpy as np

try:
    import gdal
    import pyproj
    from fastkml import kml
except:
    pass


def open_gtiff(path, dtype=None):
    ds = gdal.Open(path)
    if dtype is None:
        im_np = np.array(ds.ReadAsArray())
        return im_np.copy()
    else:
        im_np = np.array(ds.ReadAsArray(), dtype=dtype)
        return im_np.copy()


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
    # import ipdb;ipdb.set_trace()    
    y = (lons - bounds[0][0]) / (bounds[0][1] - bounds[0][0])
    x = (lats - bounds[1][0]) / (bounds[1][1] - bounds[1][0])
    y *= im_size[0]
    x *= im_size[1]
    return x, y


def image_positions_to_spherical(x, y, bounds, im_size):
    # import ipdb;ipdb.set_trace()
    y = y/im_size[0]
    x = x/im_size[1]
    lons = y*(bounds[0][1] - bounds[0][0]) + bounds[0][0]
    lats = x*(bounds[1][1] - bounds[1][0]) + bounds[1][0]
    return lons, lats

    
def fill_holes(data):
    mask = np.isnan(data)
    data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])
    return data


class RPCModel:
    def __init__(self, d):
        """
        Args:
            d (dict): dictionary read from a geotiff file with
                rasterio.open('/path/to/file.tiff', 'r').tags(ns='RPC')
        """
        self.row_offset = float(d['LINE_OFF'])
        self.col_offset = float(d['SAMP_OFF'])
        self.lat_offset = float(d['LAT_OFF'])
        self.lon_offset = float(d['LONG_OFF'])
        self.alt_offset = float(d['HEIGHT_OFF'])
        self.row_scale = float(d['LINE_SCALE'])
        self.col_scale = float(d['SAMP_SCALE'])
        self.lat_scale = float(d['LAT_SCALE'])
        self.lon_scale = float(d['LONG_SCALE'])
        self.alt_scale = float(d['HEIGHT_SCALE'])

        self.row_num = list(map(float, d['LINE_NUM_COEFF'].split()))
        self.row_den = list(map(float, d['LINE_DEN_COEFF'].split()))
        self.col_num = list(map(float, d['SAMP_NUM_COEFF'].split()))
        self.col_den = list(map(float, d['SAMP_DEN_COEFF'].split()))

    def to_list(self):
        return [self.row_offset, self.col_offset, self.lat_offset, self.lon_offset, self.alt_offset, self.row_scale, self.col_scale, 
                self.lat_scale, self.lon_scale, self.alt_scale]+self.row_num+self.row_den+self.col_num+self.col_den

def rpc_to_dict(model):
    d = dict()
    d['LINE_NUM_COEFF'] = ' '.join(['{: .30}'.format(x) for x in model.row_num])
    d['LINE_DEN_COEFF'] = ' '.join(['{: .30}'.format(x) for x in model.row_den])
    d['SAMP_NUM_COEFF'] = ' '.join(['{: .30}'.format(x) for x in model.col_num])
    d['SAMP_DEN_COEFF'] = ' '.join(['{: .30}'.format(x) for x in model.col_den])
    d['LINE_OFF'] = model.row_offset
    d['SAMP_OFF'] = model.col_offset
    d['LAT_OFF'] = model.lat_offset
    d['LONG_OFF'] = model.lon_offset
    d['HEIGHT_OFF'] = model.alt_offset
    d['LINE_SCALE'] = model.row_scale
    d['SAMP_SCALE'] = model.col_scale
    d['LAT_SCALE'] = model.lat_scale
    d['LONG_SCALE'] = model.lon_scale
    d['HEIGHT_SCALE'] = model.alt_scale
    return d 