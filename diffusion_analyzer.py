import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import cv2

class DiffusionAnalyzer:
    """Class for analyzing diffusion patterns in hole positions."""

    def __init__(self):
        """Initialize the diffusion analyzer."""
        self.reference_point = None  # Reference point for diffusion analysis
        self.points = []  # List of points (x, y) for analysis
        self.distances = []  # Distances from reference point
        self.angles = []  # Angles from reference point

        # Analysis categories
        self.categories = {
            'distance': {
                'name': 'تحليل المسافات',
                'description': 'تحليل توزيع المسافات من نقطة الأصل إلى الثقوب'
            },
            'angle': {
                'name': 'تحليل الزوايا',
                'description': 'تحليل توزيع الزوايا للثقوب حول نقطة الأصل'
            },
            'spatial': {
                'name': 'التوزيع المكاني',
                'description': 'تحليل التوزيع المكاني للثقوب في المستوى'
            },
            'uniformity': {
                'name': 'تحليل التجانس',
                'description': 'قياس مدى تجانس توزيع الثقوب'
            }
        }

        # Analysis results
        self.analysis_results = {}

    def set_reference_point(self, point):
        """Set the reference point for diffusion analysis."""
        self.reference_point = point

    def set_points(self, points):
        """Set the points for diffusion analysis."""
        self.points = points
        self.calculate_distances_and_angles()

    def calculate_distances_and_angles(self):
        """Calculate distances and angles from reference point to all points."""
        if not self.reference_point or not self.points:
            return

        self.distances = []
        self.angles = []

        ref_x, ref_y = self.reference_point

        for x, y in self.points:
            # Calculate distance
            dx = x - ref_x
            dy = y - ref_y
            distance = np.sqrt(dx**2 + dy**2)
            self.distances.append(distance)

            # Calculate angle (in degrees)
            angle = np.degrees(np.arctan2(dy, dx))
            self.angles.append(angle)

        # Perform comprehensive analysis
        self._analyze_diffusion()

    def _analyze_diffusion(self):
        """Perform comprehensive analysis of the diffusion pattern."""
        if not self.distances:
            return

        # Distance analysis
        distance_stats = self._analyze_distances()
        self.analysis_results['distance'] = distance_stats

        # Angle analysis
        angle_stats = self._analyze_angles()
        self.analysis_results['angle'] = angle_stats

        # Spatial distribution analysis
        spatial_stats = self._analyze_spatial_distribution()
        self.analysis_results['spatial'] = spatial_stats

        # Uniformity analysis
        uniformity_stats = self._analyze_uniformity()
        self.analysis_results['uniformity'] = uniformity_stats

        # Overall diffusion quality
        self.analysis_results['overall'] = self._calculate_overall_score()

    def _analyze_distances(self):
        """Analyze the distribution of distances."""
        if not self.distances:
            return {}

        # Basic statistics
        min_dist = min(self.distances)
        max_dist = max(self.distances)
        avg_dist = np.mean(self.distances)
        std_dist = np.std(self.distances)

        # Coefficient of variation (lower is more uniform)
        cv = std_dist / avg_dist if avg_dist > 0 else float('inf')

        # Distance uniformity score (0-1)
        distance_uniformity = max(0, min(1, 1 - cv))

        # Radial distribution
        radial_distribution = self.calculate_radial_distribution()

        # Distance zones
        zones = self._calculate_distance_zones()

        return {
            'min': min_dist,
            'max': max_dist,
            'avg': avg_dist,
            'std': std_dist,
            'cv': cv,
            'uniformity_score': distance_uniformity,
            'radial_distribution': radial_distribution,
            'zones': zones
        }

    def _analyze_angles(self):
        """Analyze the distribution of angles."""
        if not self.angles:
            return {}

        # Convert angles to radians for calculations
        angles_rad = np.radians(self.angles)

        # Calculate angular uniformity
        x_coords = np.cos(angles_rad)
        y_coords = np.sin(angles_rad)

        # Mean resultant length (measure of angular dispersion)
        x_mean = np.mean(x_coords)
        y_mean = np.mean(y_coords)
        r = np.sqrt(x_mean**2 + y_mean**2)

        # Angular uniformity score (0-1)
        angular_uniformity = 1 - r

        # Mean direction
        mean_direction = np.degrees(np.arctan2(y_mean, x_mean))

        # Create histogram of angles
        hist, bin_edges = np.histogram(self.angles, bins=8, range=(-180, 180))

        # Calculate angular zones
        zones = self._calculate_angular_zones()

        return {
            'mean_direction': mean_direction,
            'uniformity_score': angular_uniformity,
            'histogram': hist.tolist(),
            'bin_edges': bin_edges.tolist(),
            'zones': zones
        }

    def _analyze_spatial_distribution(self):
        """Analyze the spatial distribution of points."""
        if not self.points or not self.reference_point:
            return {}

        # Calculate nearest neighbor distances
        nn_distances = self._calculate_nearest_neighbor_distances()

        # Calculate quadrant distribution
        quadrants = self._calculate_quadrant_distribution()

        # Calculate density
        area = np.pi * (max(self.distances) ** 2) if self.distances else 0
        density = len(self.points) / area if area > 0 else 0

        return {
            'nearest_neighbor_avg': np.mean(nn_distances) if nn_distances else 0,
            'nearest_neighbor_std': np.std(nn_distances) if nn_distances else 0,
            'quadrants': quadrants,
            'density': density
        }

    def _analyze_uniformity(self):
        """Analyze the overall uniformity of the diffusion pattern."""
        # Calculate basic uniformity score
        uniformity = self.calculate_uniformity()

        # Interpret uniformity
        if uniformity > 0.75:
            interpretation = "انتشار متجانس بشكل ممتاز"
            grade = "A"
        elif uniformity > 0.5:
            interpretation = "انتشار متجانس بشكل جيد"
            grade = "B"
        elif uniformity > 0.25:
            interpretation = "انتشار متوسط التجانس"
            grade = "C"
        else:
            interpretation = "انتشار غير متجانس"
            grade = "D"

        return {
            'score': uniformity,
            'interpretation': interpretation,
            'grade': grade
        }

    def _calculate_overall_score(self):
        """Calculate an overall diffusion quality score."""
        if not self.analysis_results:
            return {'score': 0, 'grade': 'F', 'interpretation': 'لا يوجد بيانات كافية للتحليل'}

        # Weighted average of different uniformity scores
        distance_score = self.analysis_results.get('distance', {}).get('uniformity_score', 0)
        angle_score = self.analysis_results.get('angle', {}).get('uniformity_score', 0)
        uniformity_score = self.analysis_results.get('uniformity', {}).get('score', 0)

        # Calculate overall score (weighted average)
        overall_score = 0.4 * distance_score + 0.3 * angle_score + 0.3 * uniformity_score

        # Determine grade
        if overall_score > 0.8:
            grade = "A+"
            interpretation = "انتشار مثالي"
        elif overall_score > 0.7:
            grade = "A"
            interpretation = "انتشار ممتاز"
        elif overall_score > 0.6:
            grade = "B+"
            interpretation = "انتشار جيد جداً"
        elif overall_score > 0.5:
            grade = "B"
            interpretation = "انتشار جيد"
        elif overall_score > 0.4:
            grade = "C+"
            interpretation = "انتشار مقبول"
        elif overall_score > 0.3:
            grade = "C"
            interpretation = "انتشار متوسط"
        elif overall_score > 0.2:
            grade = "D"
            interpretation = "انتشار ضعيف"
        else:
            grade = "F"
            interpretation = "انتشار غير متجانس"

        return {
            'score': overall_score,
            'grade': grade,
            'interpretation': interpretation
        }

    def _calculate_distance_zones(self, num_zones=4):
        """Divide distances into zones and analyze distribution."""
        if not self.distances:
            return []

        max_dist = max(self.distances)
        zone_size = max_dist / num_zones

        zones = []
        for i in range(num_zones):
            lower = i * zone_size
            upper = (i + 1) * zone_size

            # Count points in this zone
            count = sum(1 for d in self.distances if lower <= d < upper or (i == num_zones - 1 and d == upper))

            # Calculate percentage
            percentage = (count / len(self.distances)) * 100 if self.distances else 0

            zones.append({
                'zone': i + 1,
                'range': (lower, upper),
                'count': count,
                'percentage': percentage
            })

        return zones

    def _calculate_angular_zones(self, num_zones=8):
        """Divide angles into zones and analyze distribution."""
        if not self.angles:
            return []

        # Define angular zones (in degrees)
        zone_size = 360 / num_zones

        zones = []
        for i in range(num_zones):
            lower = -180 + i * zone_size
            upper = -180 + (i + 1) * zone_size

            # Count points in this zone
            count = sum(1 for a in self.angles if lower <= a < upper or (i == num_zones - 1 and a == upper))

            # Calculate percentage
            percentage = (count / len(self.angles)) * 100 if self.angles else 0

            # Determine direction name
            direction = self._get_direction_name(lower + zone_size/2)

            zones.append({
                'zone': i + 1,
                'range': (lower, upper),
                'direction': direction,
                'count': count,
                'percentage': percentage
            })

        return zones

    def _get_direction_name(self, angle):
        """Convert angle to direction name."""
        # Normalize angle to 0-360
        angle = (angle + 180) % 360

        if 337.5 <= angle or angle < 22.5:
            return "شرق"
        elif 22.5 <= angle < 67.5:
            return "شمال شرق"
        elif 67.5 <= angle < 112.5:
            return "شمال"
        elif 112.5 <= angle < 157.5:
            return "شمال غرب"
        elif 157.5 <= angle < 202.5:
            return "غرب"
        elif 202.5 <= angle < 247.5:
            return "جنوب غرب"
        elif 247.5 <= angle < 292.5:
            return "جنوب"
        else:
            return "جنوب شرق"

    def _calculate_nearest_neighbor_distances(self):
        """Calculate distances to nearest neighbors for each point."""
        if len(self.points) < 2:
            return []

        nn_distances = []

        for i, (x1, y1) in enumerate(self.points):
            min_dist = float('inf')

            for j, (x2, y2) in enumerate(self.points):
                if i != j:
                    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    min_dist = min(min_dist, dist)

            if min_dist != float('inf'):
                nn_distances.append(min_dist)

        return nn_distances

    def _calculate_quadrant_distribution(self):
        """Calculate distribution of points in quadrants."""
        if not self.points or not self.reference_point:
            return {}

        ref_x, ref_y = self.reference_point

        # Initialize quadrant counts
        quadrants = {
            'Q1': 0,  # Top-right
            'Q2': 0,  # Top-left
            'Q3': 0,  # Bottom-left
            'Q4': 0   # Bottom-right
        }

        for x, y in self.points:
            dx = x - ref_x
            dy = y - ref_y

            if dx >= 0 and dy <= 0:
                quadrants['Q1'] += 1
            elif dx < 0 and dy <= 0:
                quadrants['Q2'] += 1
            elif dx < 0 and dy > 0:
                quadrants['Q3'] += 1
            else:
                quadrants['Q4'] += 1

        # Calculate percentages
        total = len(self.points)
        result = quadrants.copy()  # Create a copy to avoid modifying during iteration

        if total > 0:
            for q in ['Q1', 'Q2', 'Q3', 'Q4']:
                result[q + '_pct'] = (quadrants[q] / total) * 100

        return result

    def get_diffusion_stats(self):
        """Get statistical information about the diffusion pattern."""
        if not self.distances or not self.analysis_results:
            return None

        # Combine all analysis results
        stats = {
            'min_distance': self.analysis_results.get('distance', {}).get('min', 0),
            'max_distance': self.analysis_results.get('distance', {}).get('max', 0),
            'avg_distance': self.analysis_results.get('distance', {}).get('avg', 0),
            'std_distance': self.analysis_results.get('distance', {}).get('std', 0),
            'count': len(self.distances),
            'uniformity': self.analysis_results.get('uniformity', {}).get('score', 0),
            'overall_score': self.analysis_results.get('overall', {}).get('score', 0),
            'grade': self.analysis_results.get('overall', {}).get('grade', 'F'),
            'interpretation': self.analysis_results.get('overall', {}).get('interpretation', ''),
            'radial_distribution': self.analysis_results.get('distance', {}).get('radial_distribution', {})
        }

        return stats

    def calculate_uniformity(self):
        """Calculate uniformity of the diffusion pattern (0-1 scale)."""
        if not self.distances or not self.angles:
            return 0

        # Normalize distances
        max_dist = max(self.distances)
        if max_dist == 0:
            return 0

        norm_distances = [d / max_dist for d in self.distances]

        # Calculate angular uniformity (how evenly distributed the points are)
        angles_rad = np.radians(self.angles)
        x_coords = np.cos(angles_rad)
        y_coords = np.sin(angles_rad)

        # Calculate the center of mass
        x_mean = np.mean(x_coords)
        y_mean = np.mean(y_coords)

        # Calculate the distance from center of mass to origin
        # A value close to 0 indicates uniform angular distribution
        com_distance = np.sqrt(x_mean**2 + y_mean**2)

        # Convert to a 0-1 scale where 1 is perfectly uniform
        angular_uniformity = 1 - com_distance

        # Calculate distance uniformity (standard deviation of normalized distances)
        distance_uniformity = 1 - min(1, np.std(norm_distances))

        # Combine both metrics
        uniformity = (angular_uniformity + distance_uniformity) / 2

        return uniformity

    def calculate_radial_distribution(self, bins=10):
        """Calculate the radial distribution of points."""
        if not self.distances:
            return []

        # Create histogram of distances
        hist, bin_edges = np.histogram(self.distances, bins=bins)

        # Normalize by area of each annular ring
        areas = []
        for i in range(len(bin_edges) - 1):
            # Area of annular ring = π(r₂² - r₁²)
            area = np.pi * (bin_edges[i+1]**2 - bin_edges[i]**2)
            areas.append(area)

        # Normalize histogram by areas
        normalized_hist = hist / np.array(areas)

        # Further normalize to get a probability density
        if sum(normalized_hist) > 0:
            normalized_hist = normalized_hist / sum(normalized_hist)

        return {
            'histogram': hist.tolist(),
            'bin_edges': bin_edges.tolist(),
            'normalized': normalized_hist.tolist()
        }

    def create_diffusion_plot(self, figure=None):
        """Create a plot showing the diffusion pattern."""
        if not self.reference_point or not self.points:
            return None

        if figure is None:
            figure = Figure(figsize=(6, 6), dpi=100)

        # Create polar plot
        ax1 = figure.add_subplot(221, projection='polar')
        ax1.set_title('Polar Distribution')

        # Convert angles to radians for polar plot
        angles_rad = np.radians(self.angles)

        # Plot points in polar coordinates
        ax1.scatter(angles_rad, self.distances, alpha=0.7)
        ax1.grid(True)

        # Create distance histogram
        ax2 = figure.add_subplot(222)
        ax2.set_title('Distance Distribution')
        ax2.hist(self.distances, bins=10, alpha=0.7)
        ax2.set_xlabel('Distance')
        ax2.set_ylabel('Frequency')
        ax2.grid(True)

        # Create 2D scatter plot
        ax3 = figure.add_subplot(223)
        ax3.set_title('Spatial Distribution')

        # Calculate coordinates relative to reference point
        ref_x, ref_y = self.reference_point
        rel_x = [x - ref_x for x, y in self.points]
        rel_y = [y - ref_y for x, y in self.points]

        # Plot reference point at origin
        ax3.scatter(0, 0, color='red', s=100, marker='x', label='Reference')

        # Plot all points
        ax3.scatter(rel_x, rel_y, alpha=0.7, label='Holes')

        # Add concentric circles for distance reference
        max_dist = max(self.distances) if self.distances else 1
        circle_radii = np.linspace(0, max_dist, 5)[1:]
        for radius in circle_radii:
            circle = plt.Circle((0, 0), radius, fill=False, linestyle='--', alpha=0.3)
            ax3.add_patch(circle)

        ax3.axis('equal')
        ax3.grid(True)
        ax3.legend()

        # Create angular distribution
        ax4 = figure.add_subplot(224, projection='polar')
        ax4.set_title('Angular Distribution')

        # Create histogram of angles
        hist, bin_edges = np.histogram(self.angles, bins=16, range=(-180, 180))

        # Convert bin edges to radians
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_centers_rad = np.radians(bin_centers)

        # Plot angular histogram
        ax4.bar(bin_centers_rad, hist, width=np.radians(22.5), alpha=0.7)

        figure.tight_layout()
        return figure

    def embed_plot_in_frame(self, parent_frame):
        """Embed the diffusion plot in a tkinter frame."""
        # Create figure
        figure = self.create_diffusion_plot()

        if figure is None:
            return None

        # Create canvas
        canvas = FigureCanvasTkAgg(figure, master=parent_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Update canvas
        canvas.draw()

        return canvas

    def visualize_diffusion_on_image(self, image, circles, origin_point):
        """
        Visualize diffusion pattern on the image.

        Args:
            image: Input image
            circles: List of circles (x, y, r)
            origin_point: Origin point (x, y)

        Returns:
            Image with diffusion visualization
        """
        result = image.copy()

        # Set reference point (origin)
        self.set_reference_point(origin_point)

        # Extract center points
        points = [(x, y) for x, y, _ in circles]
        self.set_points(points)

        # Draw reference point
        cv2.circle(result, origin_point, 5, (0, 0, 255), -1)

        # Draw concentric circles
        if self.distances:
            max_dist = max(self.distances)
            circle_radii = np.linspace(0, max_dist, 5)[1:]

            for radius in circle_radii:
                cv2.circle(result, origin_point, int(radius), (0, 255, 255), 1)

        # Draw lines from origin to each point
        for i, (x, y, r) in enumerate(circles):
            # Calculate color based on distance (red to blue gradient)
            if self.distances:
                normalized_dist = self.distances[i] / max(self.distances)
                color = (
                    int(255 * (1 - normalized_dist)),  # B
                    0,                                 # G
                    int(255 * normalized_dist)         # R
                )
            else:
                color = (0, 255, 0)

            # Draw line from origin to point
            cv2.line(result, origin_point, (x, y), color, 1)

            # Draw circle and number
            cv2.circle(result, (x, y), r, (0, 255, 0), 2)
            cv2.circle(result, (x, y), 2, (0, 0, 255), 3)
            cv2.putText(result, str(i+1), (x-10, y-r-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        return result


class DiffusionResultDialog:
    """Dialog for displaying diffusion analysis results."""

    def __init__(self, parent, diffusion_analyzer):
        """Initialize the dialog."""
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("تحليل الانتشار")
        self.dialog.geometry("800x600")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self.diffusion_analyzer = diffusion_analyzer

        # Create UI
        self.create_ui()

    def create_ui(self):
        """Create the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel for plot
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Embed plot
        self.canvas = self.diffusion_analyzer.embed_plot_in_frame(left_panel)

        # Right panel for statistics
        right_panel = ttk.LabelFrame(main_frame, text="إحصائيات الانتشار")
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        # Get statistics
        stats = self.diffusion_analyzer.get_diffusion_stats()

        if stats:
            # Create statistics display
            self.create_stat_row(right_panel, "عدد النقاط:", f"{stats['count']}")
            self.create_stat_row(right_panel, "أقصى مسافة:", f"{stats['max_distance']:.2f}")
            self.create_stat_row(right_panel, "أدنى مسافة:", f"{stats['min_distance']:.2f}")
            self.create_stat_row(right_panel, "متوسط المسافة:", f"{stats['avg_distance']:.2f}")
            self.create_stat_row(right_panel, "الانحراف المعياري:", f"{stats['std_distance']:.2f}")

            # Calculate uniformity percentage
            uniformity_pct = stats['uniformity'] * 100
            self.create_stat_row(right_panel, "مؤشر التجانس:", f"{uniformity_pct:.1f}%")

            # Create uniformity progress bar
            ttk.Label(right_panel, text="مقياس التجانس:").pack(anchor=tk.W, padx=5, pady=(10, 0))

            progress_frame = ttk.Frame(right_panel)
            progress_frame.pack(fill=tk.X, padx=5, pady=5)

            self.progress_var = tk.DoubleVar(value=stats['uniformity'])
            progress = ttk.Progressbar(
                progress_frame,
                orient=tk.HORIZONTAL,
                length=200,
                mode='determinate',
                variable=self.progress_var
            )
            progress['maximum'] = 1.0
            progress.pack(fill=tk.X)

            # Add interpretation
            interpretation_frame = ttk.LabelFrame(right_panel, text="تفسير النتائج")
            interpretation_frame.pack(fill=tk.X, padx=5, pady=10, expand=True)

            if uniformity_pct > 75:
                interpretation = "انتشار متجانس بشكل ممتاز"
            elif uniformity_pct > 50:
                interpretation = "انتشار متجانس بشكل جيد"
            elif uniformity_pct > 25:
                interpretation = "انتشار متوسط التجانس"
            else:
                interpretation = "انتشار غير متجانس"

            ttk.Label(
                interpretation_frame,
                text=interpretation,
                wraplength=200,
                justify=tk.CENTER
            ).pack(padx=5, pady=5)

        # Close button
        ttk.Button(
            right_panel,
            text="إغلاق",
            command=self.dialog.destroy
        ).pack(pady=20)

    def create_stat_row(self, parent, label_text, value_text):
        """Create a row with label and value."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(frame, text=label_text, width=15).pack(side=tk.LEFT)
        ttk.Label(frame, text=value_text).pack(side=tk.RIGHT)
