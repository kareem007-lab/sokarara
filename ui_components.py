import tkinter as tk
from tkinter import ttk, filedialog, Scale
from PIL import Image, ImageTk
import cv2
import numpy as np

class CoordinateTable(ttk.Frame):
    """Table widget for displaying point coordinates."""

    def __init__(self, parent):
        super().__init__(parent)

        # Create treeview for the table
        self.tree = ttk.Treeview(self, columns=('Point', 'X (cm)', 'Y (cm)'), show='headings')

        # Define column headings
        self.tree.heading('Point', text='Point')
        self.tree.heading('X (cm)', text='X (cm)')
        self.tree.heading('Y (cm)', text='Y (cm)')

        # Define column widths
        self.tree.column('Point', width=50)
        self.tree.column('X (cm)', width=100)
        self.tree.column('Y (cm)', width=100)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        # Pack widgets
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def update_table(self, data):
        """
        Update the table with new data.

        Args:
            data: List of dictionaries with 'Point', 'X (cm)', 'Y (cm)' keys
        """
        # Clear existing data
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Insert new data
        for row in data:
            self.tree.insert('', tk.END, values=(
                row['Point'],
                row['X (cm)'],
                row['Y (cm)']
            ))


class ImageViewer(ttk.Frame):
    """Widget for displaying and interacting with images."""

    def __init__(self, parent):
        super().__init__(parent)

        # Control frame for zoom buttons
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X, side=tk.TOP, padx=5, pady=5)

        # Zoom buttons
        ttk.Button(control_frame, text="تكبير +", command=self.zoom_in).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="تصغير -", command=self.zoom_out).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="ملائمة للشاشة", command=self.fit_to_screen).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="الحجم الأصلي", command=self.original_size).pack(side=tk.LEFT, padx=5)

        # Create canvas for displaying the image
        self.canvas = tk.Canvas(self, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Add scrollbars
        self.h_scrollbar = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.v_scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)

        self.canvas.configure(xscrollcommand=self.h_scrollbar.set, yscrollcommand=self.v_scrollbar.set)

        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Variables
        self.image = None  # Original PIL image
        self.photo = None  # PhotoImage for display
        self.image_id = None  # Canvas image ID
        self.cv_image = None  # Original OpenCV image
        self.scale_factor = 1.0  # Current scale factor

        # Bind mouse wheel for zooming
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)

    def display_image(self, cv_image, auto_fit=True):
        """
        Display an OpenCV image on the canvas.

        Args:
            cv_image: OpenCV image (BGR format)
            auto_fit: Whether to automatically fit the image to the screen
        """
        # Store the original OpenCV image
        self.cv_image = cv_image.copy()

        # Convert from BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        self.image = Image.fromarray(rgb_image)

        # Reset scale factor
        self.scale_factor = 1.0

        # Apply auto-fit if requested
        if auto_fit:
            self.fit_to_screen()
        else:
            self.update_display()

    def update_display(self):
        """Update the displayed image with the current scale factor."""
        if self.image is None:
            return

        try:
            # Calculate new dimensions
            width, height = self.image.size
            new_width = int(width * self.scale_factor)
            new_height = int(height * self.scale_factor)

            # Ensure minimum size
            new_width = max(new_width, 50)
            new_height = max(new_height, 50)

            # Resize the image
            if self.scale_factor != 1.0:
                resized_image = self.image.resize((new_width, new_height), Image.LANCZOS)
            else:
                resized_image = self.image

            # Convert to PhotoImage
            self.photo = ImageTk.PhotoImage(resized_image)

            # Get canvas size
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            # If canvas size is not yet available, use a default size
            if canvas_width <= 1:
                canvas_width = 800
            if canvas_height <= 1:
                canvas_height = 600

            # Calculate center position
            x_center = max(0, (canvas_width - new_width) // 2)
            y_center = max(0, (canvas_height - new_height) // 2)

            # Update canvas
            if self.image_id:
                self.canvas.delete(self.image_id)

            self.image_id = self.canvas.create_image(x_center, y_center, anchor=tk.NW, image=self.photo)

            # Update canvas scroll region to include the entire image
            # Add padding to ensure scrollbars work correctly
            padding = 20
            self.canvas.configure(scrollregion=(
                x_center - padding,
                y_center - padding,
                x_center + new_width + padding,
                y_center + new_height + padding
            ))
        except Exception as e:
            print(f"Error updating display: {str(e)}")

    def zoom_in(self):
        """Zoom in the image."""
        self.scale_factor *= 1.2
        self.update_display()

    def zoom_out(self):
        """Zoom out the image."""
        self.scale_factor /= 1.2
        if self.scale_factor < 0.1:
            self.scale_factor = 0.1
        self.update_display()

    def fit_to_screen(self):
        """Fit the image to the screen while ensuring the entire image is visible."""
        if self.image is None:
            return

        try:
            # Get canvas size
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            # If canvas size is not yet available, use a default size
            if canvas_width <= 1:
                canvas_width = 800
            if canvas_height <= 1:
                canvas_height = 600

            # Get image size
            img_width, img_height = self.image.size

            # Calculate scale factors
            width_scale = canvas_width / img_width
            height_scale = canvas_height / img_height

            # Use the smaller scale factor to fit the entire image
            # Reduced from 0.95 to 0.9 to ensure more margin around the image
            self.scale_factor = min(width_scale, height_scale) * 0.9

            # Ensure we don't zoom in too much on small images
            if self.scale_factor > 1.0:
                # For small images, limit the maximum zoom to avoid excessive enlargement
                self.scale_factor = min(self.scale_factor, 1.5)

            # Ensure we don't zoom out too much on large images
            if self.scale_factor < 0.1:
                self.scale_factor = 0.1

            # Update display will center the image
            self.update_display()
        except Exception as e:
            print(f"Error fitting image to screen: {str(e)}")

    def original_size(self):
        """Reset to original image size."""
        self.scale_factor = 1.0
        self.update_display()

    def on_mousewheel(self, event):
        """Handle mouse wheel events for zooming."""
        # Zoom in/out based on mouse wheel direction
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def center_image(self):
        """Center the image in the canvas."""
        if self.image is None or self.image_id is None:
            return

        # Get canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # If canvas size is not yet available, use a default size
        if canvas_width <= 1:
            canvas_width = 800
        if canvas_height <= 1:
            canvas_height = 600

        # Get current image size (after scaling)
        img_width = int(self.image.width * self.scale_factor)
        img_height = int(self.image.height * self.scale_factor)

        # Calculate center position
        x_center = max(0, (canvas_width - img_width) // 2)
        y_center = max(0, (canvas_height - img_height) // 2)

        # Move the image to the center
        try:
            self.canvas.coords(self.image_id, x_center, y_center)

            # Update scroll region
            padding = 20
            self.canvas.configure(scrollregion=(
                x_center - padding,
                y_center - padding,
                x_center + img_width + padding,
                y_center + img_height + padding
            ))
        except Exception as e:
            print(f"Error centering image: {str(e)}")


class ParameterPanel(ttk.Frame):
    """Panel for adjusting detection parameters."""

    def __init__(self, parent, on_param_change=None):
        super().__init__(parent)

        self.on_param_change = on_param_change

        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Basic parameters tab
        basic_tab = ttk.Frame(self.notebook)
        self.notebook.add(basic_tab, text="أساسي")

        # Advanced parameters tab
        advanced_tab = ttk.Frame(self.notebook)
        self.notebook.add(advanced_tab, text="متقدم")

        # Create basic parameter sliders
        self.create_slider(basic_tab, "DP", 1.0, 3.0, 0.1, 1.5)
        self.create_slider(basic_tab, "Min Distance", 5, 100, 1, 20)
        self.create_slider(basic_tab, "Min Radius", 1, 50, 1, 5)
        self.create_slider(basic_tab, "Max Radius", 5, 100, 1, 100)

        # Create hole detection options in basic tab
        hole_frame = ttk.LabelFrame(basic_tab, text="خيارات اكتشاف الثقوب")
        hole_frame.pack(fill=tk.X, padx=5, pady=5, expand=True)

        self.detect_dark_holes_var = tk.BooleanVar(value=True)
        dark_holes_cb = ttk.Checkbutton(
            hole_frame,
            text="اكتشاف الثقوب الداكنة (بدلاً من الفاتحة)",
            variable=self.detect_dark_holes_var
        )
        dark_holes_cb.pack(fill=tk.X, padx=5, pady=5)

        self.auto_zoom_var = tk.BooleanVar(value=True)
        auto_zoom_cb = ttk.Checkbutton(
            hole_frame,
            text="ملائمة الصورة تلقائياً للشاشة",
            variable=self.auto_zoom_var
        )
        auto_zoom_cb.pack(fill=tk.X, padx=5, pady=5)

        # Create advanced parameter sliders
        self.create_slider(advanced_tab, "Param1", 10, 300, 1, 100)
        self.create_slider(advanced_tab, "Param2", 5, 100, 1, 30)
        self.create_slider(advanced_tab, "Blur Size", 3, 15, 2, 5)
        self.create_slider(advanced_tab, "Threshold", 50, 250, 1, 150)

        # Create checkboxes for boolean parameters
        self.adaptive_threshold_var = tk.BooleanVar(value=False)
        adaptive_cb = ttk.Checkbutton(
            advanced_tab,
            text="استخدام العتبة التكيفية",
            variable=self.adaptive_threshold_var
        )
        adaptive_cb.pack(fill=tk.X, padx=5, pady=5)

        self.debug_mode_var = tk.BooleanVar(value=False)
        debug_cb = ttk.Checkbutton(
            advanced_tab,
            text="وضع التصحيح (حفظ صور المعالجة)",
            variable=self.debug_mode_var
        )
        debug_cb.pack(fill=tk.X, padx=5, pady=5)

        # Add enhanced detection options
        enhanced_frame = ttk.LabelFrame(advanced_tab, text="خيارات الكشف المتقدمة")
        enhanced_frame.pack(fill=tk.X, padx=5, pady=5, expand=True)

        self.use_multiple_thresholds_var = tk.BooleanVar(value=True)
        multi_thresh_cb = ttk.Checkbutton(
            enhanced_frame,
            text="استخدام عتبات متعددة للكشف",
            variable=self.use_multiple_thresholds_var
        )
        multi_thresh_cb.pack(fill=tk.X, padx=5, pady=5)

        self.use_contrast_enhancement_var = tk.BooleanVar(value=True)
        contrast_cb = ttk.Checkbutton(
            enhanced_frame,
            text="تحسين التباين قبل الكشف",
            variable=self.use_contrast_enhancement_var
        )
        contrast_cb.pack(fill=tk.X, padx=5, pady=5)

        # Add numbered holes detection options
        numbered_frame = ttk.LabelFrame(advanced_tab, text="خيارات الثقوب المرقمة")
        numbered_frame.pack(fill=tk.X, padx=5, pady=5, expand=True)

        self.detect_only_numbered_holes_var = tk.BooleanVar(value=False)
        numbered_holes_cb = ttk.Checkbutton(
            numbered_frame,
            text="اكتشاف الثقوب المرقمة فقط",
            variable=self.detect_only_numbered_holes_var
        )
        numbered_holes_cb.pack(fill=tk.X, padx=5, pady=5)

        # Add proximity threshold slider
        proximity_frame = ttk.Frame(numbered_frame)
        proximity_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(proximity_frame, text="حد القرب من الرقم:").pack(side=tk.LEFT)

        self.number_proximity_threshold_var = tk.IntVar(value=100)
        proximity_slider = Scale(proximity_frame, from_=20, to=200, resolution=5,
                              orient=tk.HORIZONTAL, variable=self.number_proximity_threshold_var)
        proximity_slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        # Create pixel to cm ratio input
        ratio_frame = ttk.Frame(self)
        ratio_frame.pack(fill=tk.X, padx=5, pady=10)

        ttk.Label(ratio_frame, text="نسبة البكسل إلى السنتيمتر:").pack(side=tk.LEFT)
        self.ratio_var = tk.DoubleVar(value=10.0)
        ratio_entry = ttk.Entry(ratio_frame, textvariable=self.ratio_var, width=10)
        ratio_entry.pack(side=tk.RIGHT, padx=5)

        # Apply button
        ttk.Button(self, text="تطبيق المعلمات", command=self.apply_parameters).pack(pady=10)

    def create_slider(self, parent, name, from_, to, resolution, default):
        """Create a labeled slider."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(frame, text=name + ":").pack(side=tk.LEFT)

        var = tk.DoubleVar(value=default)
        setattr(self, name.lower().replace(" ", "_") + "_var", var)

        slider = Scale(frame, from_=from_, to=to, resolution=resolution,
                      orient=tk.HORIZONTAL, variable=var)
        slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)

    def apply_parameters(self):
        """Apply the current parameter values."""
        if self.on_param_change:
            params = {
                'dp': self.dp_var.get(),
                'min_dist': self.min_distance_var.get(),
                'param1': self.param1_var.get(),
                'param2': self.param2_var.get(),
                'min_radius': self.min_radius_var.get(),
                'max_radius': self.max_radius_var.get(),
                'blur_size': int(self.blur_size_var.get()),
                'threshold_value': int(self.threshold_var.get()),
                'use_adaptive_threshold': self.adaptive_threshold_var.get(),
                'debug_mode': self.debug_mode_var.get(),
                'detect_dark_holes': self.detect_dark_holes_var.get(),
                'auto_zoom': self.auto_zoom_var.get(),
                'pixel_to_cm_ratio': self.ratio_var.get(),
                'use_multiple_thresholds': getattr(self, 'use_multiple_thresholds_var', tk.BooleanVar(value=True)).get(),
                'use_contrast_enhancement': getattr(self, 'use_contrast_enhancement_var', tk.BooleanVar(value=True)).get(),
                'max_detection_attempts': 3,  # Fixed value for now
                'detect_only_numbered_holes': getattr(self, 'detect_only_numbered_holes_var', tk.BooleanVar(value=False)).get(),
                'number_proximity_threshold': getattr(self, 'number_proximity_threshold_var', tk.IntVar(value=100)).get()
            }
            self.on_param_change(params)
