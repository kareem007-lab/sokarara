#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
تطبيق سكارارا للأندرويد
تطبيق لاكتشاف الثقوب وتحليل الانتشار
"""

import os
import cv2
import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.spinner import Spinner
from kivy.uix.checkbox import CheckBox
from kivy.uix.slider import Slider
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.properties import ObjectProperty, BooleanProperty, NumericProperty, StringProperty
from kivy.core.window import Window
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Import our modules
from circle_detector import CircleDetector
from point_analyzer import PointAnalyzer
from diffusion_analyzer import DiffusionAnalyzer

# Set window size for desktop testing
Window.size = (800, 600)

class ImageViewer(Image):
    """Widget for displaying and manipulating images."""
    
    def __init__(self, **kwargs):
        super(ImageViewer, self).__init__(**kwargs)
        self.texture = None
        self.cv_image = None
        self.original_image = None
        self.allow_stretch = True
        self.keep_ratio = True
    
    def display_image(self, cv_image):
        """Display an OpenCV image."""
        if cv_image is None:
            return
        
        self.cv_image = cv_image
        self.original_image = cv_image.copy()
        
        # Convert the image to texture
        buf = cv2.flip(cv_image, 0)
        buf = buf.tostring()
        image_texture = Texture.create(size=(cv_image.shape[1], cv_image.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        
        # Display the image
        self.texture = image_texture

class ParameterPanel(TabbedPanel):
    """Panel for adjusting detection parameters."""
    
    on_param_change = ObjectProperty(None)
    
    # Basic parameters
    dp = NumericProperty(1.5)
    min_dist = NumericProperty(20)
    min_radius = NumericProperty(5)
    max_radius = NumericProperty(100)
    
    # Advanced parameters
    param1 = NumericProperty(100)
    param2 = NumericProperty(30)
    blur_size = NumericProperty(5)
    threshold_value = NumericProperty(150)
    
    # Boolean parameters
    detect_dark_holes = BooleanProperty(True)
    auto_zoom = BooleanProperty(False)
    use_adaptive_threshold = BooleanProperty(True)
    debug_mode = BooleanProperty(False)
    use_multiple_thresholds = BooleanProperty(True)
    use_contrast_enhancement = BooleanProperty(True)
    detect_only_numbered_holes = BooleanProperty(False)
    
    # Other parameters
    number_proximity_threshold = NumericProperty(100)
    pixel_to_cm_ratio = NumericProperty(10.0)
    
    def apply_parameters(self):
        """Apply the current parameter values."""
        if self.on_param_change:
            params = {
                'dp': self.dp,
                'min_dist': self.min_dist,
                'param1': self.param1,
                'param2': self.param2,
                'min_radius': self.min_radius,
                'max_radius': self.max_radius,
                'blur_size': int(self.blur_size),
                'threshold_value': int(self.threshold_value),
                'use_adaptive_threshold': self.use_adaptive_threshold,
                'debug_mode': self.debug_mode,
                'detect_dark_holes': self.detect_dark_holes,
                'auto_zoom': self.auto_zoom,
                'pixel_to_cm_ratio': self.pixel_to_cm_ratio,
                'use_multiple_thresholds': self.use_multiple_thresholds,
                'use_contrast_enhancement': self.use_contrast_enhancement,
                'max_detection_attempts': 3,
                'detect_only_numbered_holes': self.detect_only_numbered_holes,
                'number_proximity_threshold': self.number_proximity_threshold
            }
            self.on_param_change(params)

class SokararaApp(App):
    """Main application class."""
    
    def build(self):
        """Build the application UI."""
        # Create the main layout
        self.main_layout = BoxLayout(orientation='vertical')
        
        # Create the toolbar
        toolbar = BoxLayout(size_hint=(1, 0.1))
        
        # Add buttons to toolbar
        load_btn = Button(text='تحميل صورة')
        load_btn.bind(on_press=self.load_image)
        toolbar.add_widget(load_btn)
        
        detect_btn = Button(text='اكتشاف الثقوب')
        detect_btn.bind(on_press=self.detect_points)
        toolbar.add_widget(detect_btn)
        
        analyze_btn = Button(text='تحليل الانتشار')
        analyze_btn.bind(on_press=self.analyze_diffusion)
        toolbar.add_widget(analyze_btn)
        
        set_origin_btn = Button(text='تعيين نقطة الأصل')
        set_origin_btn.bind(on_press=self.set_origin)
        toolbar.add_widget(set_origin_btn)
        
        self.main_layout.add_widget(toolbar)
        
        # Create content area
        content = BoxLayout(orientation='horizontal')
        
        # Create image viewer
        self.image_viewer = ImageViewer(size_hint=(0.7, 1))
        content.add_widget(self.image_viewer)
        
        # Create parameter panel
        self.parameter_panel = ParameterPanel(size_hint=(0.3, 1), do_default_tab=False)
        self.parameter_panel.on_param_change = self.update_parameters
        content.add_widget(self.parameter_panel)
        
        self.main_layout.add_widget(content)
        
        # Create status bar
        self.status_bar = Label(text='جاهز', size_hint=(1, 0.05))
        self.main_layout.add_widget(self.status_bar)
        
        # Initialize components
        self.circle_detector = CircleDetector()
        self.point_analyzer = PointAnalyzer()
        self.diffusion_analyzer = DiffusionAnalyzer()
        
        # Initialize variables
        self.current_image = None
        self.detected_circles = []
        self.origin_point = (0, 0)
        
        return self.main_layout
    
    def load_image(self, instance):
        """Load an image from file."""
        # Create file chooser popup
        content = BoxLayout(orientation='vertical')
        file_chooser = FileChooserListView(path=os.path.expanduser('~'))
        content.add_widget(file_chooser)
        
        # Add buttons
        btn_layout = BoxLayout(size_hint=(1, 0.1))
        btn_cancel = Button(text='إلغاء')
        btn_load = Button(text='تحميل')
        btn_layout.add_widget(btn_cancel)
        btn_layout.add_widget(btn_load)
        content.add_widget(btn_layout)
        
        # Create popup
        popup = Popup(title='اختر صورة', content=content, size_hint=(0.9, 0.9))
        
        # Bind buttons
        btn_cancel.bind(on_press=popup.dismiss)
        btn_load.bind(on_press=lambda x: self._load_selected_image(file_chooser.selection, popup))
        
        popup.open()
    
    def _load_selected_image(self, selection, popup):
        """Load the selected image."""
        if not selection:
            return
        
        try:
            # Load the image
            file_path = selection[0]
            self.current_image = cv2.imread(file_path)
            
            if self.current_image is None:
                self.status_bar.text = "فشل في تحميل الصورة"
                return
            
            # Display the image
            self.image_viewer.display_image(self.current_image)
            
            # Update status
            self.status_bar.text = f"تم تحميل الصورة: {os.path.basename(file_path)}"
            
            # Reset detection results
            self.detected_circles = []
            self.origin_point = (self.current_image.shape[1] // 2, self.current_image.shape[0] // 2)
            
            # Close popup
            popup.dismiss()
            
        except Exception as e:
            self.status_bar.text = f"خطأ: {str(e)}"
    
    def detect_points(self, instance=None):
        """Detect points in the current image."""
        if self.current_image is None:
            self.status_bar.text = "يرجى تحميل صورة أولاً"
            return
        
        try:
            # Detect circles
            self.detected_circles = self.circle_detector.detect_circles(self.current_image)
            
            if not self.detected_circles:
                self.status_bar.text = "لم يتم العثور على ثقوب"
                return
            
            # Draw circles on the image
            result_image = self.current_image.copy()
            for i, (x, y, r) in enumerate(self.detected_circles):
                cv2.circle(result_image, (x, y), r, (0, 255, 0), 2)
                cv2.circle(result_image, (x, y), 2, (0, 0, 255), 3)
                cv2.putText(result_image, str(i+1), (x-10, y-r-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Draw origin point
            cv2.circle(result_image, self.origin_point, 5, (0, 0, 255), -1)
            cv2.putText(result_image, "O", (self.origin_point[0]+10, self.origin_point[1]+10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display result
            self.image_viewer.display_image(result_image)
            
            # Set points in analyzers
            self.point_analyzer.set_origin(self.origin_point)
            self.point_analyzer.set_points(self.detected_circles)
            self.diffusion_analyzer.set_reference_point(self.origin_point)
            self.diffusion_analyzer.set_points([(x, y) for x, y, _ in self.detected_circles])
            
            # Update status
            self.status_bar.text = f"تم اكتشاف {len(self.detected_circles)} ثقب"
            
        except Exception as e:
            self.status_bar.text = f"خطأ: {str(e)}"
            import traceback
            traceback.print_exc()
    
    def set_origin(self, instance):
        """Set the origin point for coordinate calculations."""
        if self.current_image is None:
            self.status_bar.text = "يرجى تحميل صورة أولاً"
            return
        
        # Create popup for setting origin
        content = BoxLayout(orientation='vertical')
        
        # Add instructions
        content.add_widget(Label(text="أدخل إحداثيات نقطة الأصل:"))
        
        # Add coordinate inputs
        coord_layout = BoxLayout()
        coord_layout.add_widget(Label(text="X:"))
        x_input = Spinner(text=str(self.origin_point[0]), values=[str(i) for i in range(0, self.current_image.shape[1], 10)])
        coord_layout.add_widget(x_input)
        
        coord_layout.add_widget(Label(text="Y:"))
        y_input = Spinner(text=str(self.origin_point[1]), values=[str(i) for i in range(0, self.current_image.shape[0], 10)])
        coord_layout.add_widget(y_input)
        
        content.add_widget(coord_layout)
        
        # Add buttons
        btn_layout = BoxLayout(size_hint=(1, 0.2))
        btn_cancel = Button(text='إلغاء')
        btn_apply = Button(text='تطبيق')
        btn_layout.add_widget(btn_cancel)
        btn_layout.add_widget(btn_apply)
        content.add_widget(btn_layout)
        
        # Create popup
        popup = Popup(title='تعيين نقطة الأصل', content=content, size_hint=(0.8, 0.4))
        
        # Define apply function
        def apply_origin(instance):
            try:
                x = int(x_input.text)
                y = int(y_input.text)
                self.origin_point = (x, y)
                
                # Redraw if we have detected circles
                if self.detected_circles:
                    self.detect_points()
                else:
                    # Just draw the origin point
                    result_image = self.current_image.copy()
                    cv2.circle(result_image, self.origin_point, 5, (0, 0, 255), -1)
                    cv2.putText(result_image, "O", (self.origin_point[0]+10, self.origin_point[1]+10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    self.image_viewer.display_image(result_image)
                
                self.status_bar.text = f"تم تعيين نقطة الأصل إلى ({x}, {y})"
                popup.dismiss()
            except ValueError:
                self.status_bar.text = "خطأ: يرجى إدخال أرقام صحيحة"
        
        # Bind buttons
        btn_cancel.bind(on_press=popup.dismiss)
        btn_apply.bind(on_press=apply_origin)
        
        popup.open()
    
    def update_parameters(self, params):
        """Update detection parameters."""
        # Update circle detector parameters
        self.circle_detector.set_parameters(**params)
        
        # Update point analyzer parameters
        self.point_analyzer.set_pixel_to_cm_ratio(params['pixel_to_cm_ratio'])
        
        # Re-detect points if an image is loaded
        if self.current_image is not None and self.detected_circles:
            self.detect_points()
        
        self.status_bar.text = "تم تحديث المعلمات"
    
    def analyze_diffusion(self, instance=None):
        """Analyze the diffusion pattern of detected holes."""
        if not self.detected_circles:
            self.status_bar.text = "يرجى اكتشاف الثقوب أولاً"
            return
        
        try:
            # Create a visualization of the diffusion pattern
            diffusion_image = self.diffusion_analyzer.visualize_diffusion_on_image(
                self.current_image,
                self.detected_circles,
                self.origin_point
            )
            
            # Display the diffusion visualization
            self.image_viewer.display_image(diffusion_image)
            
            # Show diffusion analysis popup
            self._show_diffusion_results()
            
            # Update status
            self.status_bar.text = "تم تحليل الانتشار"
            
        except Exception as e:
            self.status_bar.text = f"خطأ: {str(e)}"
            import traceback
            traceback.print_exc()
    
    def _show_diffusion_results(self):
        """Show diffusion analysis results in a popup."""
        # Create content layout
        content = BoxLayout(orientation='vertical')
        
        # Get statistics
        stats = self.diffusion_analyzer.get_diffusion_stats()
        
        if not stats:
            content.add_widget(Label(text="لا توجد إحصائيات متاحة"))
        else:
            # Create scrollable stats view
            scroll = ScrollView()
            stats_layout = GridLayout(cols=2, spacing=10, size_hint_y=None)
            stats_layout.bind(minimum_height=stats_layout.setter('height'))
            
            # Add statistics
            self._add_stat_row(stats_layout, "عدد النقاط:", f"{stats['count']}")
            self._add_stat_row(stats_layout, "أقصى مسافة:", f"{stats['max_distance']:.2f}")
            self._add_stat_row(stats_layout, "أدنى مسافة:", f"{stats['min_distance']:.2f}")
            self._add_stat_row(stats_layout, "متوسط المسافة:", f"{stats['avg_distance']:.2f}")
            self._add_stat_row(stats_layout, "الانحراف المعياري:", f"{stats['std_distance']:.2f}")
            
            # Calculate uniformity percentage
            uniformity_pct = stats['uniformity'] * 100
            self._add_stat_row(stats_layout, "مؤشر التجانس:", f"{uniformity_pct:.1f}%")
            
            # Add interpretation
            if uniformity_pct > 75:
                interpretation = "انتشار متجانس بشكل ممتاز"
            elif uniformity_pct > 50:
                interpretation = "انتشار متجانس بشكل جيد"
            elif uniformity_pct > 25:
                interpretation = "انتشار متوسط التجانس"
            else:
                interpretation = "انتشار غير متجانس"
            
            self._add_stat_row(stats_layout, "التفسير:", interpretation)
            
            scroll.add_widget(stats_layout)
            content.add_widget(scroll)
        
        # Create figure for diffusion plot
        fig = self.diffusion_analyzer.create_diffusion_plot()
        if fig:
            content.add_widget(FigureCanvasKivyAgg(fig))
        
        # Add close button
        btn_close = Button(text='إغلاق', size_hint=(1, 0.1))
        content.add_widget(btn_close)
        
        # Create popup
        popup = Popup(title='نتائج تحليل الانتشار', content=content, size_hint=(0.9, 0.9))
        
        # Bind close button
        btn_close.bind(on_press=popup.dismiss)
        
        popup.open()
    
    def _add_stat_row(self, layout, label_text, value_text):
        """Add a row with label and value to the stats layout."""
        layout.add_widget(Label(text=label_text, size_hint_y=None, height=30))
        layout.add_widget(Label(text=value_text, size_hint_y=None, height=30))

if __name__ == '__main__':
    try:
        SokararaApp().run()
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
