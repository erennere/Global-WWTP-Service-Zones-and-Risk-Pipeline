"""Raster assignment grid to polygons (SciPy, OpenCV or rasterio contours).

Part of the service-area stage (``python -m src.create_voronoi``).
"""
import logging
import numpy as np
from rasterio.features import shapes
from skimage.measure import find_contours
import cv2
from shapely import Polygon
from shapely.geometry import shape
import shapely.affinity
import shapely

logger = logging.getLogger(__name__)


def extract_contours_scipy(region_mask_2d, n_points, grid_minx, grid_miny):
    """
    Extract polygon contours from a region mask using scipy.measure.find_contours.
    
    Uses scipy's marching squares algorithm to find contours in a 2D binary mask,
    then converts contours to polygon coordinates in the original coordinate system.
    
    Parameters
    ----------
    region_mask_2d : numpy.ndarray
        Boolean mask indicating region membership.
    n_points : int
        Grid spacing multiplier.
    grid_minx : float
        Minimum x coordinate of the grid origin.
    grid_miny : float
        Minimum y coordinate of the grid origin.

    Returns
    -------
    list[shapely.geometry.Polygon]
        Valid polygons extracted from the mask.
    """
    polygons = []
    contours = find_contours(region_mask_2d, level=0.5, fully_connected='low', positive_orientation='low')
    logger.debug(f"scipy.find_contours extracted {len(contours)} contours")
    
    for contour in contours:
        contour = np.array(contour)
        # Convert from grid indices to actual coordinates
        contour_x = contour[:, 1] * n_points + grid_minx
        contour_y = contour[:, 0] * n_points + grid_miny
        poly_coords = np.stack([contour_x, contour_y], axis=-1).reshape(-1, 2)
        
        # Ensure polygon is closed
        if not np.array_equal(poly_coords[0], poly_coords[-1]):
            poly_coords = np.vstack([poly_coords, poly_coords[0]])
        
        # Create and validate polygon
        poly = Polygon(poly_coords).buffer(0)
        if poly.is_valid and not poly.is_empty:
            polygons.append(poly)
    
    logger.debug(f"scipy contour extraction created {len(polygons)} valid polygons")
    return polygons


def extract_contours_cv2(region_mask_2d, n_points, grid_minx, grid_miny):
    """
    Extract polygon contours from a region mask using OpenCV's contour finding.
    
    Uses cv2.findContours with external contour retrieval to extract boundaries
    from a binary mask, then converts contours to polygon coordinates.
    
    Parameters
    ----------
    region_mask_2d : numpy.ndarray
        Boolean mask indicating region membership.
    n_points : int
        Grid spacing multiplier.
    grid_minx : float
        Minimum x coordinate of the grid origin.
    grid_miny : float
        Minimum y coordinate of the grid origin.

    Returns
    -------
    list[shapely.geometry.Polygon]
        Valid polygons extracted from the mask.
    """
    polygons = []
    mask_uint8 = (region_mask_2d.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logger.debug(f"cv2.findContours extracted {len(contours)} contours")
    
    for contour in contours:
        contour = contour.squeeze()
        # Skip invalid contours
        if contour.ndim != 2 or contour.shape[0] < 3:
            continue
        
        # Convert from grid indices to actual coordinates
        contour_x = contour[:, 0] * n_points + grid_minx
        contour_y = contour[:, 1] * n_points + grid_miny
        poly_coords = np.stack([contour_x, contour_y], axis=-1).reshape(-1, 2)
        
        # Ensure polygon is closed
        if not np.array_equal(poly_coords[0], poly_coords[-1]):
            poly_coords = np.vstack([poly_coords, poly_coords[0]])
        
        # Create and validate polygon
        poly = Polygon(poly_coords).buffer(0)
        if poly.is_valid and not poly.is_empty:
            polygons.append(poly)
    
    logger.debug(f"cv2 contour extraction created {len(polygons)} valid polygons")
    return polygons


def extract_contours_rasterio(region_mask_2d, n_points, grid_minx, grid_miny):
    """
    Extract polygon contours from a region mask using rasterio's feature extraction.
    
    Uses rasterio.features.shapes to convert a raster mask to vector polygons,
    then applies affine transformations to map grid coordinates to actual space.
    
    Parameters
    ----------
    region_mask_2d : numpy.ndarray
        Boolean mask indicating region membership.
    n_points : int
        Grid spacing multiplier.
    grid_minx : float
        Minimum x coordinate of the grid origin.
    grid_miny : float
        Minimum y coordinate of the grid origin.

    Returns
    -------
    list[shapely.geometry.Polygon]
        Valid polygons extracted from the mask.
    """
    polygons = []
    results = shapes(region_mask_2d.astype(np.uint8), mask=region_mask_2d > 0)
    for geom, val in results:
        poly = shape(geom)
        # Pixel (i, j) covers index space [i, i+1] x [j, j+1] while its grid
        # node sits at index i: shift by half a cell so each cell is centred
        # on its node, then scale by the grid step.
        poly = shapely.affinity.scale(poly, xfact=n_points, yfact=n_points, origin=(0, 0))
        poly = shapely.affinity.translate(poly, xoff=grid_minx - n_points / 2.0, yoff=grid_miny - n_points / 2.0)
        poly = poly.buffer(0)
        
        if poly.is_valid and not poly.is_empty:
            polygons.append(poly)
    
    logger.debug(f"rasterio contour extraction created {len(polygons)} valid polygons")
    return polygons
