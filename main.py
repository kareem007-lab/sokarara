import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os

from circle_detector import CircleDetector
from point_analyzer import PointAnalyzer
from ui_components import CoordinateTable, ImageViewer, ParameterPanel
from diffusion_analyzer import DiffusionAnalyzer, DiffusionResultDialog

class PointCoordinateApp:
    """Main application for detecting and analyzing point coordinates."""

    def __init__(self, root):
        self.root = root
        self.root.title("نظام قراءة إحداثيات الثقوب")
        self.root.geometry("1200x800")

        # Initialize components
        self.circle_detector = CircleDetector()
        self.point_analyzer = PointAnalyzer(pixel_to_cm_ratio=10.0)  # Default: 10 pixels = 1 cm
        self.diffusion_analyzer = DiffusionAnalyzer()

        # Variables
        self.current_image = None
        self.detected_circles = []
        self.origin_point = (50, 50)  # Default origin point

        # Create UI
        self.create_ui()

    def create_ui(self):
        """Create the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Top frame for buttons
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=5)

        # Load image button
        ttk.Button(top_frame, text="تحميل صورة", command=self.load_image).pack(side=tk.LEFT, padx=5)

        # Detect holes button
        ttk.Button(top_frame, text="اكتشاف الثقوب", command=self.detect_points).pack(side=tk.LEFT, padx=5)

        # Set origin button
        ttk.Button(top_frame, text="تعيين نقطة الأصل", command=self.set_origin).pack(side=tk.LEFT, padx=5)

        # Export button
        ttk.Button(top_frame, text="تصدير النتائج", command=self.export_results).pack(side=tk.LEFT, padx=5)

        # Diffusion analysis button
        ttk.Button(top_frame, text="تحليل الانتشار", command=self.analyze_diffusion).pack(side=tk.LEFT, padx=5)

        # Content frame
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Left panel for image
        left_panel = ttk.Frame(content_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Image viewer
        self.image_viewer = ImageViewer(left_panel)
        self.image_viewer.pack(fill=tk.BOTH, expand=True)

        # Right panel for controls and table
        right_panel = ttk.Frame(content_frame, width=300)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)

        # Parameter panel
        self.param_panel = ParameterPanel(right_panel, on_param_change=self.update_parameters)
        self.param_panel.pack(fill=tk.X, pady=10)

        # Coordinate table
        ttk.Label(right_panel, text="جدول الإحداثيات:").pack(anchor=tk.W, padx=5, pady=(10, 0))
        self.coord_table = CoordinateTable(right_panel)
        self.coord_table.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Status bar
        self.status_var = tk.StringVar(value="جاهز")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))

    def load_image(self):
        """Load an image from file."""
        file_path = filedialog.askopenfilename(
            title="اختر صورة",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )

        if not file_path:
            return

        try:
            # Load the image
            self.current_image = cv2.imread(file_path)

            if self.current_image is None:
                messagebox.showerror("خطأ", "فشل في تحميل الصورة")
                return

            # Display the image with auto-fit based on user preference
            auto_fit = self.circle_detector.auto_zoom
            self.image_viewer.display_image(self.current_image, auto_fit=auto_fit)

            # Reset detected circles
            self.detected_circles = []

            # Update status
            self.status_var.set(f"تم تحميل الصورة: {os.path.basename(file_path)}")

        except Exception as e:
            messagebox.showerror("خطأ", f"حدث خطأ أثناء تحميل الصورة: {str(e)}")

    def detect_points(self):
        """Detect holes (circles) in the current image."""
        if self.current_image is None:
            messagebox.showwarning("تحذير", "يرجى تحميل صورة أولاً")
            return

        try:
            # Update status
            self.status_var.set("جاري اكتشاف النقاط...")
            self.root.update()

            # Make a copy of the image for processing
            working_image = self.current_image.copy()

            # Detect circles
            self.detected_circles = self.circle_detector.detect_circles(working_image)

            if not self.detected_circles:
                # If no circles detected, show a message and the binary image to help debug
                if self.circle_detector.debug_mode:
                    messagebox.showinfo(
                        "معلومات",
                        "لم يتم اكتشاف أي نقاط. تم حفظ صور المعالجة للتصحيح."
                    )
                else:
                    messagebox.showinfo("معلومات", "لم يتم اكتشاف أي نقاط")

                self.status_var.set("لم يتم اكتشاف أي نقاط")
                return

            # Extract center points
            points = [(x, y) for x, y, _ in self.detected_circles]

            # Find farthest point
            farthest_point, _ = self.point_analyzer.find_farthest_point(points)

            # Draw circles and axes
            result_image = self.circle_detector.draw_circles(self.current_image, self.detected_circles)
            result_image = self.point_analyzer.draw_axes(result_image, self.origin_point, farthest_point)

            # Update diffusion analyzer with current points
            self.diffusion_analyzer.set_reference_point(self.origin_point)
            self.diffusion_analyzer.set_points([(x, y) for x, y, _ in self.detected_circles])

            # Display the result with auto-fit based on user preference
            auto_fit = self.circle_detector.auto_zoom
            self.image_viewer.display_image(result_image, auto_fit=auto_fit)

            # Update coordinate table
            # Adjust points relative to origin
            adjusted_points = [(x - self.origin_point[0], y - self.origin_point[1]) for x, y, _ in self.detected_circles]
            table_data = self.point_analyzer.create_coordinate_table(adjusted_points, pixel_coords=True)
            self.coord_table.update_table(table_data)

            # Update status
            self.status_var.set(f"تم اكتشاف {len(self.detected_circles)} نقطة")

        except Exception as e:
            messagebox.showerror("خطأ", f"حدث خطأ أثناء اكتشاف النقاط: {str(e)}")
            self.status_var.set("حدث خطأ أثناء اكتشاف النقاط")
            import traceback
            traceback.print_exc()

    def set_origin(self):
        """Set the origin point for coordinate system."""
        if self.current_image is None:
            messagebox.showwarning("تحذير", "يرجى تحميل صورة أولاً")
            return

        # Create a dialog for setting origin
        dialog = tk.Toplevel(self.root)
        dialog.title("تعيين نقطة الأصل")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()

        # Create form
        ttk.Label(dialog, text="إحداثي X:").grid(row=0, column=0, padx=10, pady=10)
        x_var = tk.IntVar(value=self.origin_point[0])
        ttk.Entry(dialog, textvariable=x_var, width=10).grid(row=0, column=1, padx=10, pady=10)

        ttk.Label(dialog, text="إحداثي Y:").grid(row=1, column=0, padx=10, pady=10)
        y_var = tk.IntVar(value=self.origin_point[1])
        ttk.Entry(dialog, textvariable=y_var, width=10).grid(row=1, column=1, padx=10, pady=10)

        # Function to apply the new origin
        def apply_origin():
            try:
                x = x_var.get()
                y = y_var.get()
                self.origin_point = (x, y)

                # If circles are detected, redraw with new origin
                if self.detected_circles:
                    self.detect_points()
                else:
                    # Just draw axes
                    result_image = self.current_image.copy()
                    result_image = self.point_analyzer.draw_axes(result_image, self.origin_point)
                    auto_fit = self.circle_detector.auto_zoom
                    self.image_viewer.display_image(result_image, auto_fit=auto_fit)

                dialog.destroy()
                self.status_var.set(f"تم تعيين نقطة الأصل إلى ({x}, {y})")

            except Exception as e:
                messagebox.showerror("خطأ", f"حدث خطأ: {str(e)}")

        # Add buttons
        button_frame = ttk.Frame(dialog)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)

        ttk.Button(button_frame, text="تطبيق", command=apply_origin).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="إلغاء", command=dialog.destroy).pack(side=tk.LEFT, padx=10)

    def update_parameters(self, params):
        """Update detection parameters."""
        # Update circle detector parameters
        self.circle_detector.set_parameters(
            dp=params['dp'],
            min_dist=params['min_dist'],
            param1=params['param1'],
            param2=params['param2'],
            min_radius=params['min_radius'],
            max_radius=params['max_radius'],
            blur_size=params['blur_size'],
            threshold_value=params['threshold_value'],
            use_adaptive_threshold=params['use_adaptive_threshold'],
            debug_mode=params['debug_mode'],
            detect_dark_holes=params['detect_dark_holes'],
            auto_zoom=params['auto_zoom'],
            use_multiple_thresholds=params.get('use_multiple_thresholds', True),
            use_contrast_enhancement=params.get('use_contrast_enhancement', True),
            max_detection_attempts=params.get('max_detection_attempts', 3),
            detect_only_numbered_holes=params.get('detect_only_numbered_holes', False),
            number_proximity_threshold=params.get('number_proximity_threshold', 100)
        )

        # Update point analyzer parameters
        self.point_analyzer.set_pixel_to_cm_ratio(params['pixel_to_cm_ratio'])

        # Re-detect points if an image is loaded
        if self.current_image is not None:
            # If only auto_zoom changed, just update the display
            if self.detected_circles and params.get('auto_zoom') != getattr(self, '_last_auto_zoom', None):
                self._last_auto_zoom = params['auto_zoom']
                # Redisplay the current result with new auto-zoom setting
                if hasattr(self.image_viewer, 'cv_image') and self.image_viewer.cv_image is not None:
                    self.image_viewer.display_image(self.image_viewer.cv_image, auto_fit=params['auto_zoom'])
            else:
                # Otherwise, re-detect points
                self.detect_points()

        self.status_var.set("تم تحديث المعلمات")

    def export_results(self):
        """Export the results to a CSV file."""
        if not self.detected_circles:
            messagebox.showwarning("تحذير", "لا توجد نقاط لتصديرها")
            return

        file_path = filedialog.asksaveasfilename(
            title="حفظ النتائج",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            # Get table data
            adjusted_points = [(x - self.origin_point[0], y - self.origin_point[1]) for x, y, _ in self.detected_circles]
            table_data = self.point_analyzer.create_coordinate_table(adjusted_points, pixel_coords=True)

            # Write to CSV
            with open(file_path, 'w', encoding='utf-8') as f:
                # Write header
                f.write("النقطة,X (cm),Y (cm)\n")

                # Write data
                for row in table_data:
                    f.write(f"{row['Point']},{row['X (cm)']},{row['Y (cm)']}\n")

            self.status_var.set(f"تم تصدير النتائج إلى: {os.path.basename(file_path)}")
            messagebox.showinfo("تم", f"تم تصدير النتائج بنجاح إلى:\n{file_path}")

        except Exception as e:
            messagebox.showerror("خطأ", f"حدث خطأ أثناء تصدير النتائج: {str(e)}")

    def analyze_diffusion(self):
        """Analyze the diffusion pattern of detected holes."""
        if not self.detected_circles:
            messagebox.showwarning("تحذير", "يرجى اكتشاف الثقوب أولاً")
            return

        try:
            # Create a visualization of the diffusion pattern
            diffusion_image = self.diffusion_analyzer.visualize_diffusion_on_image(
                self.current_image,
                self.detected_circles,
                self.origin_point
            )

            # Display the diffusion visualization
            auto_fit = self.circle_detector.auto_zoom
            self.image_viewer.display_image(diffusion_image, auto_fit=auto_fit)

            # Show diffusion analysis dialog
            DiffusionResultDialog(self.root, self.diffusion_analyzer)

            # Update status
            self.status_var.set("تم تحليل الانتشار")

        except Exception as e:
            messagebox.showerror("خطأ", f"حدث خطأ أثناء تحليل الانتشار: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    try:
        print("Starting application...")
        root = tk.Tk()
        print("Tkinter initialized")
        app = PointCoordinateApp(root)
        print("Application initialized")
        root.mainloop()
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
