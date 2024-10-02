from osgeo import gdal, osr
from typing import Tuple, Any

gdal.UseExceptions()
DB_REF2016_ZONES = {
    2: 9932,
    3: 9933,
    4: 9934,
    5: 9935
}
WGS84_EPSG = 4326


def pixel_to_coordinates(path: str, x_pixel: float, y_pixel: float) -> Any:
    """
    Convert pixel coordinates to geographic coordinates.

    This function opens a TIFF image and retrieves its geotransformation 
    coefficients to calculate the geographic coordinates in the projection of the image
    for the given pixel coordinates (x_pixel, y_pixel).

    Parameters:
    ----------
    path : str
        Path to the TIFF image.
    x_pixel : int
        Horizontal pixel coordinate.
    y_pixel : int
        Vertical pixel coordinate.

    Returns:
    -------
    Any
        Geographic coordinates (x_coordinate, y_coordinate) in the base projection of the image.

    Source:
    -------
    GDAL Documentation: https://gdal.org/en/latest/api/gdaldataset_cpp.html#_CPPv4N11GDALDataset15GetGeoTransformEPd
    """
    with gdal.Open(path) as dataset:
        geotransform = dataset.GetGeoTransform()

        # x_coordinate = top_left_x_coordinate + x_pixel * pixel_width y_pixel * rotation/skew_correction
        x_coordinate = geotransform[0] + x_pixel*geotransform[1] + y_pixel*geotransform[2];
        y_coordinate = geotransform[3] + y_pixel*geotransform[5] + x_pixel*geotransform[4];

        return x_coordinate, y_coordinate


def dbref_to_wgs84(x_coordinate: float, y_coordinate: float) -> Tuple[float, float, float]:
    """
    Converts coordinated from DB_REF2016 to WGS84.

    Parameters:
    ----------
    x_coordinate : int
        Horizontal coordinate.
    y_coordinate : int
        Vertical coordinate.

    Returns:
    -------
    Tuple[float, float, float]
        WSG84 coordinates; latitude, longitude, height.
    """
    db_ref_zone = int(str(x_coordinate)[0])
    source_srs = osr.SpatialReference()
    if db_ref_zone not in DB_REF2016_ZONES.keys():
        return (None, None, None,)
    source_srs.ImportFromEPSG(DB_REF2016_ZONES[db_ref_zone])
    
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(WGS84_EPSG)

    transform = osr.CoordinateTransformation(source_srs, target_srs)
    return transform.TransformPoint(x_coordinate, y_coordinate)