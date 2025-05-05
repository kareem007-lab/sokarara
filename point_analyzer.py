import numpy as np
import cv2
import matplotlib.pyplot as plt

class PointAnalyzer:
    """Class for analyzing points and calculating coordinates."""
    
    def __init__(self, pixel_to_cm_ratio=1.0):
        """
        Initialize the point analyzer.
        
        Args:
            pixel_to_cm_ratio: Conversion ratio from pixels to centimeters
        """
        self.pixel_to_cm_ratio = pixel_to_cm_ratio
        
    def set_pixel_to_cm_ratio(self, ratio):
        """Set the pixel to centimeter conversion ratio."""
        self.pixel_to_cm_ratio = ratio
    
    def find_farthest_point(self, points):
        """
        Find the point farthest from the origin (0,0).
        
        Args:
            points: List of points (x, y)
            
        Returns:
            The farthest point and its distance
        """
        max_distance = 0
        farthest_point = None
        
        for x, y in points:
            distance = np.sqrt(x**2 + y**2)
            if distance > max_distance:
                max_distance = distance
                farthest_point = (x, y)
                
        return farthest_point, max_distance
    
    def calculate_coordinates_cm(self, points):
        """
        Calculate coordinates in centimeters.
        
        Args:
            points: List of points (x, y) in pixels
            
        Returns:
            List of points in centimeters
        """
        return [(x / self.pixel_to_cm_ratio, y / self.pixel_to_cm_ratio) for x, y in points]
    
    def draw_axes(self, image, origin=(0, 0), max_point=None, padding=50):
        """
        Draw X and Y axes on the image based on the farthest point.
        
        Args:
            image: Input image
            origin: Origin point (x, y)
            max_point: Maximum point to determine axis length
            padding: Additional padding for axes
            
        Returns:
            Image with drawn axes
        """
        result = image.copy()
        h, w = image.shape[:2]
        
        # If max_point is not provided, use image dimensions
        if max_point is None:
            max_x, max_y = w - padding, h - padding
        else:
            max_x, max_y = max_point[0] + padding, max_point[1] + padding
        
        # Draw X-axis
        cv2.arrowedLine(result, 
                       (origin[0], origin[1]), 
                       (max_x, origin[1]), 
                       (255, 0, 0), 2, tipLength=0.03)
        
        # Draw Y-axis
        cv2.arrowedLine(result, 
                       (origin[0], origin[1]), 
                       (origin[0], max_y), 
                       (0, 0, 255), 2, tipLength=0.03)
        
        # Label the axes
        cv2.putText(result, "X", (max_x - 20, origin[1] + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.putText(result, "Y", (origin[0] + 20, max_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return result
    
    def create_coordinate_table(self, points, pixel_coords=True):
        """
        Create a table of point coordinates.
        
        Args:
            points: List of points (x, y)
            pixel_coords: Whether points are in pixel coordinates
            
        Returns:
            Table data as a list of rows
        """
        table_data = []
        
        for i, (x, y) in enumerate(points, start=1):
            if pixel_coords:
                x_cm = x / self.pixel_to_cm_ratio
                y_cm = y / self.pixel_to_cm_ratio
            else:
                x_cm, y_cm = x, y
                
            table_data.append({
                'Point': i,
                'X (cm)': round(x_cm, 2),
                'Y (cm)': round(y_cm, 2)
            })
            
        return table_data
