import cv2
import numpy as np
from PIL import Image

# On Android, pytesseract is not available, so we'll disable OCR functionality
# and use a simpler approach for numbered holes detection
PYTESSERACT_AVAILABLE = False
try:
    # Try to import pytesseract for desktop environments
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    print("Warning: pytesseract is not installed. OCR functionality will be disabled.")

# Define a platform variable to check if we're running on Android
import os
ON_ANDROID = False
try:
    from android.permissions import request_permissions, Permission
    ON_ANDROID = True
    # Request necessary permissions on Android
    request_permissions([
        Permission.CAMERA,
        Permission.WRITE_EXTERNAL_STORAGE,
        Permission.READ_EXTERNAL_STORAGE
    ])
except ImportError:
    pass  # Not on Android

class CircleDetector:
    """Class for detecting holes (circles) in an image."""

    def __init__(self):
        # Default parameters for hole detection
        self.dp = 1.5
        self.min_dist = 20
        self.param1 = 100
        self.param2 = 30
        self.min_radius = 5
        self.max_radius = 100

        # Additional parameters for image preprocessing
        self.blur_size = 5
        self.threshold_value = 150
        self.use_adaptive_threshold = True  # Changed to True for better handling of varying lighting
        self.debug_mode = False

        # Hole detection specific parameters
        self.detect_dark_holes = True  # True for dark holes, False for bright holes
        self.auto_zoom = False  # Changed to False to avoid excessive zooming by default

        # Advanced detection parameters
        self.multi_scale_detection = True  # Use multiple scales for detection
        self.use_canny = True  # Use Canny edge detection
        self.edge_detection_threshold = 100  # Threshold for edge detection
        self.min_circularity = 0.6  # Reduced for better detection of slightly irregular holes
        self.use_watershed = True  # Use watershed algorithm for overlapping holes
        self.detection_method = "combined"  # Options: "hough", "contour", "combined"

        # Enhanced detection parameters
        self.use_multiple_thresholds = True  # Try multiple threshold values
        self.use_contrast_enhancement = True  # Enhance contrast before detection
        self.max_detection_attempts = 3  # Number of detection attempts with different parameters

        # Numbered holes detection parameters
        self.detect_only_numbered_holes = False  # Whether to detect only holes near numbers
        self.number_proximity_threshold = 100  # Maximum distance between a hole and a number to consider it numbered
        self.ocr_confidence_threshold = 0.3  # Minimum confidence for OCR detection

    def set_parameters(self, dp=None, min_dist=None, param1=None, param2=None,
                      min_radius=None, max_radius=None, blur_size=None,
                      threshold_value=None, use_adaptive_threshold=None, debug_mode=None,
                      detect_dark_holes=None, auto_zoom=None, multi_scale_detection=None,
                      use_canny=None, edge_detection_threshold=None, min_circularity=None,
                      use_watershed=None, detection_method=None, use_multiple_thresholds=None,
                      use_contrast_enhancement=None, max_detection_attempts=None,
                      detect_only_numbered_holes=None, number_proximity_threshold=None,
                      ocr_confidence_threshold=None):
        """Set parameters for hole detection."""
        if dp is not None:
            self.dp = dp
        if min_dist is not None:
            self.min_dist = min_dist
        if param1 is not None:
            self.param1 = param1
        if param2 is not None:
            self.param2 = param2
        if min_radius is not None:
            self.min_radius = min_radius
        if max_radius is not None:
            self.max_radius = max_radius
        if blur_size is not None:
            self.blur_size = blur_size
        if threshold_value is not None:
            self.threshold_value = threshold_value
        if use_adaptive_threshold is not None:
            self.use_adaptive_threshold = use_adaptive_threshold
        if debug_mode is not None:
            self.debug_mode = debug_mode
        if detect_dark_holes is not None:
            self.detect_dark_holes = detect_dark_holes
        if auto_zoom is not None:
            self.auto_zoom = auto_zoom
        if multi_scale_detection is not None:
            self.multi_scale_detection = multi_scale_detection
        if use_canny is not None:
            self.use_canny = use_canny
        if edge_detection_threshold is not None:
            self.edge_detection_threshold = edge_detection_threshold
        if min_circularity is not None:
            self.min_circularity = min_circularity
        if use_watershed is not None:
            self.use_watershed = use_watershed
        if detection_method is not None:
            self.detection_method = detection_method
        if use_multiple_thresholds is not None:
            self.use_multiple_thresholds = use_multiple_thresholds
        if use_contrast_enhancement is not None:
            self.use_contrast_enhancement = use_contrast_enhancement
        if max_detection_attempts is not None:
            self.max_detection_attempts = max_detection_attempts
        if detect_only_numbered_holes is not None:
            self.detect_only_numbered_holes = detect_only_numbered_holes
        if number_proximity_threshold is not None:
            self.number_proximity_threshold = number_proximity_threshold
        if ocr_confidence_threshold is not None:
            self.ocr_confidence_threshold = ocr_confidence_threshold

    def detect_circles(self, image):
        """
        Detect holes (circles) in the given image using advanced techniques.

        Args:
            image: Input image (BGR format)

        Returns:
            List of detected holes (x, y, radius)
        """
        try:
            # Make a copy of the image for debugging
            debug_images = {}
            original = image.copy()
            debug_images['original'] = original

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            debug_images['gray'] = gray

            # Apply contrast enhancement if enabled
            if self.use_contrast_enhancement:
                try:
                    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    gray = clahe.apply(gray)
                    debug_images['contrast_enhanced'] = gray
                except Exception as e:
                    print(f"Error in contrast enhancement: {str(e)}")

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)
            debug_images['blurred'] = blurred

            # Initialize circles list
            all_circles = []

            # Try multiple detection attempts with different parameters
            for attempt in range(self.max_detection_attempts):
                # Adjust parameters for each attempt
                if attempt == 0:
                    # First attempt: use default parameters
                    current_threshold = self.threshold_value
                    current_param1 = self.param1
                    current_param2 = self.param2
                elif attempt == 1:
                    # Second attempt: adjust threshold and parameters
                    current_threshold = self.threshold_value - 30
                    current_param1 = self.param1 * 0.8
                    current_param2 = self.param2 * 0.8
                else:
                    # Third attempt: try different parameters
                    current_threshold = self.threshold_value + 30
                    current_param1 = self.param1 * 1.2
                    current_param2 = self.param2 * 0.7

                try:
                    # Create binary image using appropriate thresholding
                    binary = self._create_binary_image(blurred, threshold_override=current_threshold)
                    if attempt == 0:
                        debug_images['binary'] = binary

                    # Apply morphological operations to clean up the binary image
                    binary_cleaned = self._clean_binary_image(binary)
                    if attempt == 0:
                        debug_images['morphology'] = binary_cleaned

                    # Prepare edge image if needed
                    edge_image = None
                    if self.use_canny:
                        # Use Canny edge detection for better circle detection
                        low_threshold = self.edge_detection_threshold // 2
                        high_threshold = self.edge_detection_threshold
                        edge_image = cv2.Canny(blurred, low_threshold, high_threshold)
                        if attempt == 0:
                            debug_images['edges'] = edge_image

                        # Dilate edges slightly to connect broken edges
                        edge_image = cv2.dilate(edge_image, np.ones((2, 2), np.uint8), iterations=1)
                        if attempt == 0:
                            debug_images['edges_dilated'] = edge_image

                    # Detect circles using the selected method
                    circles = []

                    if self.detection_method == "hough" or self.detection_method == "combined":
                        try:
                            hough_circles = self._detect_circles_hough(
                                binary_cleaned,
                                edge_image,
                                param1_override=current_param1,
                                param2_override=current_param2
                            )
                            if hough_circles:
                                circles.extend(hough_circles)
                        except Exception as e:
                            print(f"Error in Hough circle detection: {str(e)}")

                    if self.detection_method == "contour" or self.detection_method == "combined" or not circles:
                        try:
                            contour_circles = self._detect_circles_contour(binary_cleaned)

                            # Merge with existing circles, avoiding duplicates
                            for center_x, center_y, radius in contour_circles:
                                # Check if this circle is already in our list (avoid duplicates)
                                is_duplicate = False
                                for cx, cy, cr in circles:
                                    # If centers are close, consider it a duplicate
                                    if np.sqrt((cx - center_x)**2 + (cy - center_y)**2) < self.min_dist:
                                        is_duplicate = True
                                        break

                                if not is_duplicate:
                                    circles.append((center_x, center_y, radius))
                        except Exception as e:
                            print(f"Error in contour circle detection: {str(e)}")

                    # If multi-scale detection is enabled, try detecting at different scales
                    if self.multi_scale_detection:
                        # Try with different parameters
                        for scale_factor in [0.7, 0.9, 1.1, 1.3]:
                            try:
                                # Adjust parameters for this scale
                                temp_min_radius = max(1, int(self.min_radius * scale_factor))
                                temp_max_radius = max(temp_min_radius + 1, int(self.max_radius * scale_factor))

                                # Detect with adjusted parameters
                                if self.detection_method == "hough" or self.detection_method == "combined":
                                    hough_circles = self._detect_circles_hough(
                                        binary_cleaned,
                                        edge_image,
                                        min_radius=temp_min_radius,
                                        max_radius=temp_max_radius,
                                        param1_override=current_param1,
                                        param2_override=current_param2
                                    )

                                    if hough_circles:
                                        # Add non-duplicate circles
                                        for center_x, center_y, radius in hough_circles:
                                            is_duplicate = False
                                            for cx, cy, cr in circles:
                                                if np.sqrt((cx - center_x)**2 + (cy - center_y)**2) < self.min_dist:
                                                    is_duplicate = True
                                                    break

                                            if not is_duplicate:
                                                circles.append((center_x, center_y, radius))
                            except Exception as e:
                                print(f"Error in multi-scale detection with scale {scale_factor}: {str(e)}")

                    # If watershed is enabled and we have circles, use it to separate overlapping circles
                    if self.use_watershed and circles:
                        try:
                            watershed_circles = self._separate_overlapping_circles(binary_cleaned, circles)

                            # Replace circles with watershed results if we got more circles
                            if len(watershed_circles) > len(circles):
                                circles = watershed_circles
                        except Exception as e:
                            print(f"Error in watershed separation: {str(e)}")

                    # Add circles from this attempt to the overall list, avoiding duplicates
                    for center_x, center_y, radius in circles:
                        is_duplicate = False
                        for cx, cy, cr in all_circles:
                            if np.sqrt((cx - center_x)**2 + (cy - center_y)**2) < self.min_dist:
                                is_duplicate = True
                                break

                        if not is_duplicate:
                            all_circles.append((center_x, center_y, radius))

                except Exception as e:
                    print(f"Error in detection attempt {attempt}: {str(e)}")

                # If we've found a good number of circles, we can stop
                if len(all_circles) >= 5:
                    break

            # If detect_only_numbered_holes is enabled, filter circles to keep only those near numbers
            if self.detect_only_numbered_holes:
                try:
                    # Detect numbers in the image
                    detected_numbers = self._detect_numbers(original)

                    # Filter circles to keep only those near numbers
                    all_circles = self._filter_circles_near_numbers(all_circles, detected_numbers)

                    print(f"Detected {len(detected_numbers)} numbers and {len(all_circles)} circles near them")
                except Exception as e:
                    print(f"Error filtering circles near numbers: {str(e)}")

            # If debug mode is on, save debug images
            if self.debug_mode:
                try:
                    # Create a debug image showing all processing steps
                    debug_result = original.copy()
                    for i, (x, y, r) in enumerate(all_circles):
                        cv2.circle(debug_result, (x, y), r, (0, 255, 0), 2)
                        cv2.circle(debug_result, (x, y), 2, (0, 0, 255), 3)
                        cv2.putText(debug_result, str(i+1), (x-10, y-r-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                    debug_images['result'] = debug_result

                    # Save debug images
                    for name, img in debug_images.items():
                        if len(img.shape) == 2:  # Grayscale
                            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                        else:
                            img_color = img

                        # Save to a location that works on Android
                        if ON_ANDROID:
                            save_path = os.path.join(os.getenv('EXTERNAL_STORAGE', '/sdcard'), f"debug_{name}.jpg")
                        else:
                            save_path = f"debug_{name}.jpg"
                        cv2.imwrite(save_path, img_color)
                except Exception as e:
                    print(f"Error saving debug images: {str(e)}")

            return all_circles

        except Exception as e:
            print(f"Error in detect_circles: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def _create_binary_image(self, gray_image, threshold_override=None):
        """
        Create binary image using appropriate thresholding.

        Args:
            gray_image: Grayscale input image
            threshold_override: Optional threshold value to override the default

        Returns:
            Binary image
        """
        # Use override threshold if provided, otherwise use the default
        threshold = threshold_override if threshold_override is not None else self.threshold_value

        if self.use_adaptive_threshold:
            # Adaptive thresholding can handle varying lighting conditions better
            if self.detect_dark_holes:
                binary = cv2.adaptiveThreshold(
                    gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 11, 2
                )
            else:
                binary = cv2.adaptiveThreshold(
                    gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2
                )
        else:
            # Simple binary thresholding
            if self.detect_dark_holes:
                _, binary = cv2.threshold(
                    gray_image, threshold, 255, cv2.THRESH_BINARY_INV
                )
            else:
                _, binary = cv2.threshold(
                    gray_image, threshold, 255, cv2.THRESH_BINARY
                )

        # If using multiple thresholds, try Otsu's method as well
        if self.use_multiple_thresholds and threshold_override is None:
            if self.detect_dark_holes:
                _, otsu_binary = cv2.threshold(
                    gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )
            else:
                _, otsu_binary = cv2.threshold(
                    gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )

            # Combine the two binary images (logical OR)
            binary = cv2.bitwise_or(binary, otsu_binary)

        return binary

    def _clean_binary_image(self, binary_image):
        """Apply morphological operations to clean up the binary image."""
        # Create a kernel for morphological operations
        kernel = np.ones((3, 3), np.uint8)

        # Apply opening to remove small noise
        binary = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=1)

        # Apply closing to fill small holes in the foreground
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        return binary

    def _detect_circles_hough(self, binary_image, edge_image=None, min_radius=None, max_radius=None,
                           param1_override=None, param2_override=None):
        """
        Detect circles using Hough Circle Transform.

        Args:
            binary_image: Binary image for detection
            edge_image: Optional edge image for detection
            min_radius: Optional minimum radius override
            max_radius: Optional maximum radius override
            param1_override: Optional param1 override for HoughCircles
            param2_override: Optional param2 override for HoughCircles

        Returns:
            List of detected circles (x, y, radius)
        """
        try:
            # Use provided parameters or default class values
            min_r = min_radius if min_radius is not None else self.min_radius
            max_r = max_radius if max_radius is not None else self.max_radius
            param1 = param1_override if param1_override is not None else self.param1
            param2 = param2_override if param2_override is not None else self.param2

            # Ensure min_radius is at least 1 and max_radius is greater than min_radius
            min_r = max(1, min_r)
            max_r = max(min_r + 1, max_r)

            # Choose the image to use for detection
            detection_image = edge_image if edge_image is not None else binary_image

            # Ensure the image is properly formatted for HoughCircles
            if detection_image.dtype != np.uint8:
                detection_image = detection_image.astype(np.uint8)

            # Try HOUGH_GRADIENT_ALT first (available in OpenCV 4.x)
            try:
                hough_circles = cv2.HoughCircles(
                    detection_image,
                    cv2.HOUGH_GRADIENT_ALT,
                    dp=self.dp,
                    minDist=self.min_dist,
                    param1=param1,
                    param2=param2 / 100.0,  # HOUGH_GRADIENT_ALT uses param2 as a threshold ratio (0-1)
                    minRadius=min_r,
                    maxRadius=max_r
                )
            except Exception as e:
                print(f"HOUGH_GRADIENT_ALT failed, falling back to HOUGH_GRADIENT: {str(e)}")
                # Fall back to HOUGH_GRADIENT if HOUGH_GRADIENT_ALT is not available
                hough_circles = cv2.HoughCircles(
                    detection_image,
                    cv2.HOUGH_GRADIENT,
                    dp=self.dp,
                    minDist=self.min_dist,
                    param1=param1,
                    param2=param2,
                    minRadius=min_r,
                    maxRadius=max_r
                )

            if hough_circles is not None:
                hough_circles = np.round(hough_circles[0, :]).astype(int)
                return [(x, y, r) for x, y, r in hough_circles]
            else:
                # If no circles found, try with more relaxed parameters
                try:
                    relaxed_param2 = param2 * 0.7  # More relaxed threshold
                    hough_circles = cv2.HoughCircles(
                        detection_image,
                        cv2.HOUGH_GRADIENT,
                        dp=self.dp,
                        minDist=self.min_dist,
                        param1=param1,
                        param2=relaxed_param2,
                        minRadius=min_r,
                        maxRadius=max_r
                    )

                    if hough_circles is not None:
                        hough_circles = np.round(hough_circles[0, :]).astype(int)
                        return [(x, y, r) for x, y, r in hough_circles]
                except Exception as e:
                    print(f"Error in relaxed Hough circle detection: {str(e)}")

                return []
        except Exception as e:
            print(f"Error in _detect_circles_hough: {str(e)}")
            return []

    def _detect_circles_contour(self, binary_image):
        """Detect circles by finding contours and fitting circles."""
        circles = []

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Filter small contours
            if cv2.contourArea(contour) < 20:
                continue

            # Check if contour is approximately circular
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)

            # Circularity = 4*pi*area/perimeter^2, close to 1 for circles
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

            if circularity > self.min_circularity:  # Higher threshold for circularity for holes
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)

                if self.min_radius <= radius <= self.max_radius:
                    circles.append((center[0], center[1], radius))

        return circles

    def _detect_numbers(self, image):
        """
        Detect numbers in the image.

        On desktop with pytesseract: Uses OCR to detect numbers.
        On Android or without pytesseract: Uses a simpler approach based on contour properties.

        Args:
            image: Input image (BGR format)

        Returns:
            List of detected numbers with their positions [(number, x, y, width, height)]
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply threshold to get binary image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Initialize list to store detected numbers
            detected_numbers = []

            if PYTESSERACT_AVAILABLE and not ON_ANDROID:
                # Desktop approach with OCR
                # Process each contour
                for contour in contours:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)

                    # Filter out very small contours
                    if w < 10 or h < 10:
                        continue

                    # Extract the region of interest
                    roi = gray[y:y+h, x:x+w]

                    # Use pytesseract to recognize text
                    config = '--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789'
                    text = pytesseract.image_to_string(roi, config=config).strip()

                    # If a number is detected
                    if text and text.isdigit():
                        # Calculate center of the bounding box
                        center_x = x + w // 2
                        center_y = y + h // 2
                        detected_numbers.append((int(text), center_x, center_y, w, h))
            else:
                # Android approach without OCR
                # We'll use contour properties to identify potential number regions
                for i, contour in enumerate(contours):
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)

                    # Filter out very small or very large contours
                    if w < 10 or h < 10 or w > 100 or h > 100:
                        continue

                    # Check aspect ratio (numbers are usually taller than wide or square)
                    aspect_ratio = float(w) / h
                    if aspect_ratio > 1.5:  # Too wide to be a number
                        continue

                    # Check area (numbers usually have a certain size range)
                    area = cv2.contourArea(contour)
                    if area < 50 or area > 5000:
                        continue

                    # Calculate center of the bounding box
                    center_x = x + w // 2
                    center_y = y + h // 2

                    # Since we can't recognize the actual number, we'll just use the contour index
                    # This is a simplification, but it allows us to identify regions that might contain numbers
                    detected_numbers.append((i+1, center_x, center_y, w, h))

            if self.debug_mode:
                # Create a debug image showing detected numbers
                debug_image = image.copy()
                for number, x, y, w, h in detected_numbers:
                    cv2.rectangle(debug_image, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 255, 0), 2)
                    cv2.putText(debug_image, str(number), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Save to a location that works on Android
                if ON_ANDROID:
                    save_path = os.path.join(os.getenv('EXTERNAL_STORAGE', '/sdcard'), "debug_numbers.jpg")
                else:
                    save_path = "debug_numbers.jpg"
                cv2.imwrite(save_path, debug_image)

            return detected_numbers

        except Exception as e:
            print(f"Error in _detect_numbers: {str(e)}")
            return []

    def _filter_circles_near_numbers(self, circles, numbers):
        """
        Filter circles to keep only those near numbers.

        Args:
            circles: List of detected circles (x, y, radius)
            numbers: List of detected numbers [(number, x, y, width, height)]

        Returns:
            List of filtered circles (x, y, radius)
        """
        if not circles or not numbers:
            return circles

        filtered_circles = []

        for circle_x, circle_y, radius in circles:
            # Check if this circle is close to any number
            is_near_number = False
            for _, number_x, number_y, _, _ in numbers:
                # Calculate distance between circle center and number center
                distance = np.sqrt((circle_x - number_x)**2 + (circle_y - number_y)**2)

                # If distance is less than threshold, consider it a numbered hole
                if distance < self.number_proximity_threshold:
                    is_near_number = True
                    break

            if is_near_number:
                filtered_circles.append((circle_x, circle_y, radius))

        return filtered_circles

    def _separate_overlapping_circles(self, binary_image, initial_circles):
        """Use watershed algorithm to separate overlapping circles."""
        if not initial_circles:
            return []

        # Create a markers image
        markers = np.zeros(binary_image.shape, dtype=np.int32)

        # Mark the background as 1
        markers[binary_image == 0] = 1

        # Mark each circle center with a unique ID
        for i, (x, y, _) in enumerate(initial_circles, start=2):
            markers[y, x] = i

        # Apply watershed
        # We need a color image for watershed
        color_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        cv2.watershed(color_image, markers)

        # Find contours in the watershed result
        watershed_result = np.uint8(markers)
        watershed_result[watershed_result == 1] = 0  # Set background to 0
        watershed_result[watershed_result > 1] = 255  # Set foreground to 255

        # Find contours in the watershed result
        contours, _ = cv2.findContours(watershed_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Fit circles to the contours
        circles = []
        for contour in contours:
            # Filter small contours
            if cv2.contourArea(contour) < 20:
                continue

            # Fit a circle to the contour
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            if self.min_radius <= radius <= self.max_radius:
                circles.append((center[0], center[1], radius))

        return circles

    def draw_circles(self, image, circles, start_index=1):
        """
        Draw detected circles on the image with numbering.

        Args:
            image: Input image
            circles: List of circles (x, y, radius)
            start_index: Starting index for numbering

        Returns:
            Image with drawn circles and numbers
        """
        result = image.copy()

        for i, (x, y, r) in enumerate(circles, start=start_index):
            # Draw the circle
            cv2.circle(result, (x, y), r, (0, 255, 0), 2)

            # Draw the center point
            cv2.circle(result, (x, y), 2, (0, 0, 255), 3)

            # Add the circle number
            cv2.putText(result, str(i), (x - 10, y - r - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        return result
