import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF
import numpy as np
import time
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# ======================
# ROBUST CONFIGURATION
# ======================
class Config:
    # Device optimization
    DEVICE = 'mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    USE_FP16 = False  # Disable for MPS stability
    USE_COMPILE = False  # Disable for MPS stability
    
    # Paths
    MODEL_PATH = '/Users/vik0t/hackatons/human-segment/segmentation/RobustVideoMatting/rvm_mobilenetv3.pth'
    BG_PATH = "segmentation/image.png"
    
    # Processing parameters
    OUTPUT_WIDTH, OUTPUT_HEIGHT = 1280, 720
    DOWNSAMPLE_RATIO = 0.25
    QUALITY_THRESHOLDS = {
        'dark': 40,
        'low_light': 80,  
        'normal': 120
    }
    
    # Performance tuning
    ADAPTIVE_DOWNSAMPLE = True
    TEMPORAL_CONSISTENCY = True

config = Config()

print(f"Using device: {config.DEVICE}")

# ======================
# ADVANCED LOW-LIGHT ENHANCEMENT
# ======================
class AdvancedLowLightEnhancer:
    def __init__(self):
        self.methods = {
            'fusion': self.fusion_enhance,
            'adaptive': self.adaptive_enhance,
            'fast': self.fast_enhance
        }

    def analyze_light_conditions(self, frame_bgr):
        """Comprehensive light condition analysis"""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        metrics = {
            'brightness': gray.mean(),
            'contrast': gray.std(),
            'dynamic_range': gray.max() - gray.min(),
            'dark_ratio': np.sum(gray < 50) / gray.size,
            'noise_estimate': self.estimate_noise(gray)
        }
        
        return metrics

    def estimate_noise(self, gray_image):
        """Estimate noise level"""
        try:
            dx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            dy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            return np.std(dx) + np.std(dy)
        except:
            return 0.0

    def adaptive_enhance(self, frame_bgr, metrics):
        """Adaptive enhancement based on light analysis"""
        brightness = metrics['brightness']
        contrast = metrics['contrast']
        
        enhanced = frame_bgr.copy()
        
        if brightness < config.QUALITY_THRESHOLDS['dark']:
            enhanced = self.aggressive_enhance(enhanced)
        elif brightness < config.QUALITY_THRESHOLDS['low_light']:
            enhanced = self.balanced_enhance(enhanced)
        else:
            enhanced = self.subtle_enhance(enhanced)
            
        if contrast < 25:
            enhanced = self.enhance_contrast(enhanced)
            
        return enhanced

    def aggressive_enhance(self, frame_bgr):
        """Aggressive enhancement for very dark conditions"""
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        gamma = 2.0
        l = self.adjust_gamma(l, gamma)
        
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        return enhanced

    def balanced_enhance(self, frame_bgr):
        """Balanced enhancement for low light"""
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        gamma = 1.5
        l = self.adjust_gamma(l, gamma)
        
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def subtle_enhance(self, frame_bgr):
        """Subtle enhancement for normal light"""
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def adjust_gamma(self, image, gamma=1.0):
        """Gamma correction"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def enhance_contrast(self, image):
        """Contrast enhancement"""
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    def fusion_enhance(self, frame_bgr, metrics):
        """Fusion enhancement method"""
        return self.adaptive_enhance(frame_bgr, metrics)

    def fast_enhance(self, frame_bgr, metrics):
        """Fast enhancement for real-time processing"""
        gamma = 1.8 if metrics['brightness'] < 60 else 1.2
        return self.adjust_gamma(frame_bgr, gamma)

# ======================
# OPTIMIZED SEGMENTATION PIPELINE
# ======================
class OptimizedSegmentationPipeline:
    def __init__(self, device=config.DEVICE):
        self.device = device
        self.model = None
        self.rec_states = [None] * 4
        self.quality_cache = {}
        self.frame_count = 0
        self.last_downsample_ratio = config.DOWNSAMPLE_RATIO
        
        self.load_model()
        
    def load_model(self):
        """Load and optimize the segmentation model"""
        try:
            from RobustVideoMatting.model import MattingNetwork
            self.model = MattingNetwork('mobilenetv3').eval().to(self.device)
            
            state_dict = torch.load(config.MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(state_dict)
                
            print("✓ Segmentation model loaded and optimized")
            
        except Exception as e:
            print(f"✗ Failed to load segmentation model: {e}")
            raise

    def adaptive_downsample_ratio(self, metrics):
        """Adaptive downsample ratio with state management"""
        base_ratio = config.DOWNSAMPLE_RATIO
        
        if not config.ADAPTIVE_DOWNSAMPLE:
            return base_ratio
            
        brightness = metrics['brightness']
        noise = metrics['noise_estimate']
        
        # Calculate new ratio
        if brightness < config.QUALITY_THRESHOLDS['dark']:
            new_ratio = max(0.1, base_ratio * 0.8)
        elif noise > 30:
            new_ratio = min(0.5, base_ratio * 1.5)
        else:
            new_ratio = base_ratio
            
        # Reset states if ratio changes significantly
        if (self.rec_states[0] is not None and 
            abs(new_ratio - self.last_downsample_ratio) > 0.05):
            self.rec_states = [None] * 4
            print(f"✓ Reset recurrent states due to ratio change: {self.last_downsample_ratio:.3f} -> {new_ratio:.3f}")
            
        self.last_downsample_ratio = new_ratio
        return new_ratio

    def reset_states(self):
        """Reset recurrent states"""
        self.rec_states = [None] * 4
        self.last_downsample_ratio = config.DOWNSAMPLE_RATIO

    def segment_frame(self, frame_tensor, metrics):
        """Segment frame with robust state management"""
        downsample_ratio = self.adaptive_downsample_ratio(metrics)
        
        with torch.no_grad():
            # Perform segmentation
            fgr, pha, *rec = self.model(
                frame_tensor, 
                *self.rec_states, 
                downsample_ratio=downsample_ratio
            )
            
            # Update recurrent states with temporal consistency
            if config.TEMPORAL_CONSISTENCY:
                self.rec_states = rec
                
        return pha, rec

# ======================
# DUAL-STREAM PROCESSOR
# ======================
class DualStreamProcessor:
    def __init__(self):
        self.device = config.DEVICE
        self.enhancer = AdvancedLowLightEnhancer()
        self.segmenter = OptimizedSegmentationPipeline(self.device)
        
        # Performance monitoring
        self.fps = 0.0
        self.fps_alpha = 0.1
        self.processing_times = []
        
        # Load background
        self.bg_tensor = self.load_background()
        
    def load_background(self):
        """Load and prepare background tensor"""
        bg_pil = Image.open(config.BG_PATH).convert("RGB")
        bg_resized = bg_pil.resize((config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT), Image.LANCZOS)
        bg_tensor = TF.to_tensor(bg_resized).to(self.device).unsqueeze(0)
        return bg_tensor

    def np_to_tensor(self, np_rgb):
        """Convert numpy RGB to tensor"""
        tensor = torch.from_numpy(np_rgb).to(self.device)
        tensor = tensor.permute(2, 0, 1).float().div_(255.0).unsqueeze_(0)
        return tensor

    def tensor_to_numpy(self, tensor):
        """Convert tensor to numpy BGR for display"""
        np_rgb = (tensor.squeeze(0).permute(1, 2, 0) * 255).byte().cpu().numpy()
        return cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)

    def process_frame(self, frame_bgr):
        """Main processing pipeline with dual-stream logic"""
        start_time = time.time()
        
        # Analyze frame quality
        metrics = self.enhancer.analyze_light_conditions(frame_bgr)
        
        # Choose processing path based on quality
        if metrics['brightness'] < config.QUALITY_THRESHOLDS['low_light']:
            # Enhanced path for low light
            enhanced_bgr = self.enhancer.fusion_enhance(frame_bgr, metrics)
            processing_path = "enhanced"
        else:
            # Fast path for normal light
            enhanced_bgr = self.enhancer.fast_enhance(frame_bgr, metrics)
            processing_path = "fast"
        
        # Convert to RGB and resize
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        enh_resized_rgb = cv2.resize(enhanced_rgb, (config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT))
        
        # Convert to tensor and segment
        src_tensor = self.np_to_tensor(enh_resized_rgb)
        pha, _ = self.segmenter.segment_frame(src_tensor, metrics)
        
        # Composite with original frame for better color preservation
        orig_resized_rgb = cv2.resize(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), 
            (config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT)
        )
        orig_tensor = self.np_to_tensor(orig_resized_rgb)
        
        # Upsample alpha if needed
        if pha.shape[2:] != (config.OUTPUT_HEIGHT, config.OUTPUT_WIDTH):
            pha_up = F.interpolate(
                pha, 
                size=(config.OUTPUT_HEIGHT, config.OUTPUT_WIDTH), 
                mode='bilinear', 
                align_corners=False
            )
        else:
            pha_up = pha
            
        # Composite
        com = orig_tensor * pha_up + self.bg_tensor * (1 - pha_up)
        output_bgr = self.tensor_to_numpy(com)
        
        processing_time = time.time() - start_time
        
        return {
            'output_bgr': output_bgr,
            'metrics': metrics,
            'processing_path': processing_path,
            'processing_time': processing_time,
            'alpha_mask': pha_up,
            'enhanced_bgr': enhanced_bgr
        }

# ======================
# ENHANCED MAIN APPLICATION
# ======================
class EnhancedHumanSegmentationApp:
    def __init__(self):
        self.processor = DualStreamProcessor()
        self.cap = None
        self.view_mode = 0  # 0 = final, 1 = debug, 2 = metrics
        self.performance_stats = {
            'frame_count': 0,
            'avg_fps': 0.0,
            'avg_processing_time': 0.0,
            'min_fps': float('inf'),
            'max_fps': 0.0
        }
        
    def setup_camera(self):
        """Setup camera with optimized parameters"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")
            
        # Optimize camera settings
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("✓ Camera initialized with optimized settings")

    def draw_metrics_overlay(self, image, result, fps):
        """Draw comprehensive metrics overlay"""
        metrics = result['metrics']
        
        y_offset = 30
        line_height = 25
        
        info_lines = [
            f"FPS: {fps:.1f}",
            f"Mode: {'Final' if self.view_mode == 0 else 'Debug'}",
            f"Path: {result['processing_path']}",
            f"Brightness: {metrics['brightness']:.1f}",
            f"Contrast: {metrics['contrast']:.1f}",
            f"Noise: {metrics['noise_estimate']:.1f}",
            f"Dark Ratio: {metrics['dark_ratio']:.2f}",
            f"Process: {result['processing_time']*1000:.1f}ms"
        ]
        
        for i, line in enumerate(info_lines):
            y_pos = y_offset + i * line_height
            color = (0, 255, 0) if fps > 15 else (0, 165, 255) if fps > 8 else (0, 0, 255)
            cv2.putText(image, line, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def create_debug_display(self, original, result):
        """Create comprehensive debug display"""
        enhanced_bgr = result.get('enhanced_bgr', original)
        alpha_mask = result['alpha_mask']
        
        # Convert alpha mask for display
        alpha_disp = (alpha_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
        alpha_disp = cv2.cvtColor(alpha_disp, cv2.COLOR_GRAY2BGR)
        
        # Resize all to same size for grid
        grid_size = (360, 640)
        orig_small = cv2.resize(original, grid_size)
        enh_small = cv2.resize(enhanced_bgr, grid_size)
        alpha_small = cv2.resize(alpha_disp, grid_size)
        output_small = cv2.resize(result['output_bgr'], grid_size)
        
        # Create 2x2 grid
        top_row = np.hstack((orig_small, enh_small))
        bottom_row = np.hstack((alpha_small, output_small))
        grid = np.vstack((top_row, bottom_row))
        
        # Add labels
        labels = ["Original", "Enhanced", "Alpha Mask", "Final Output"]
        positions = [(10, 30), (grid_size[0] + 10, 30), 
                    (10, grid_size[1] + 30), (grid_size[0] + 10, grid_size[1] + 30)]
        
        for label, pos in zip(labels, positions):
            cv2.putText(grid, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return grid

    def create_metrics_display(self, result):
        """Create dedicated metrics display"""
        metrics = result['metrics']
        display = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(display, "PERFORMANCE METRICS", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # Performance metrics
        perf_lines = [
            f"Average FPS: {self.performance_stats['avg_fps']:.1f}",
            f"Min FPS: {self.performance_stats['min_fps']:.1f}",
            f"Max FPS: {self.performance_stats['max_fps']:.1f}",
            f"Frame Count: {self.performance_stats['frame_count']}",
            f"Avg Process Time: {self.performance_stats['avg_processing_time']*1000:.1f}ms",
            f"Current Process Time: {result['processing_time']*1000:.1f}ms"
        ]
        
        # Quality metrics
        quality_lines = [
            f"Brightness: {metrics['brightness']:.1f}",
            f"Contrast: {metrics['contrast']:.1f}",
            f"Dynamic Range: {metrics['dynamic_range']:.1f}",
            f"Noise Level: {metrics['noise_estimate']:.1f}",
            f"Dark Ratio: {metrics['dark_ratio']:.3f}",
            f"Processing Path: {result['processing_path']}"
        ]
        
        # Draw performance metrics
        for i, line in enumerate(perf_lines):
            y_pos = 120 + i * 40
            color = (0, 200, 255) if i < 3 else (0, 255, 200)
            cv2.putText(display, line, (50, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw quality metrics
        for i, line in enumerate(quality_lines):
            y_pos = 120 + i * 40
            cv2.putText(display, line, (500, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 2)
        
        # Draw visual indicators
        self.draw_visual_indicators(display, metrics, result)
        
        return display

    def draw_visual_indicators(self, display, metrics, result):
        """Draw visual indicators for metrics"""
        # Brightness indicator
        brightness_bar = int((metrics['brightness'] / 255) * 400)
        cv2.rectangle(display, (50, 450), (450, 480), (100, 100, 100), -1)
        cv2.rectangle(display, (50, 450), (50 + brightness_bar, 480), (0, 255, 0), -1)
        cv2.putText(display, "Brightness", (50, 440), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Processing time indicator
        process_time = min(result['processing_time'] * 1000, 100)  # Cap at 100ms
        time_bar = int((process_time / 100) * 400)
        cv2.rectangle(display, (50, 520), (450, 550), (100, 100, 100), -1)
        cv2.rectangle(display, (50, 520), (50 + time_bar, 550), 
                     (0, 255, 0) if process_time < 33 else (0, 165, 255) if process_time < 66 else (0, 0, 255), -1)
        cv2.putText(display, f"Process Time: {process_time:.1f}ms", (50, 510), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def run(self):
        """Main application loop"""
        self.setup_camera()
        
        print("\n=== Enhanced Human Segmentation System ===")
        print("Controls:")
        print("  'q' - Quit")
        print("  't' - Toggle view mode (0=Final, 1=Debug, 2=Metrics)")
        print("  'b' - Reload background")
        print("  'r' - Reset recurrent states")
        print("  'p' - Print performance report")
        print()
        
        try:
            while True:
                start_time = time.time()
                
                # Capture frame
                ret, frame_bgr = self.cap.read()
                if not ret:
                    break
                
                # Process frame
                result = self.processor.process_frame(frame_bgr)
                
                # Update performance stats
                self.update_performance_stats(result['processing_time'])
                
                # Display based on view mode
                current_fps = 1.0 / result['processing_time'] if result['processing_time'] > 0 else 0
                
                if self.view_mode == 0:
                    # Final output
                    output = result['output_bgr']
                    self.draw_metrics_overlay(output, result, self.performance_stats['avg_fps'])
                    cv2.imshow('Enhanced Human Segmentation', output)
                    cv2.destroyWindow('Debug View')
                    cv2.destroyWindow('Metrics View')
                    
                elif self.view_mode == 1:
                    # Debug view
                    debug_display = self.create_debug_display(frame_bgr, result)
                    self.draw_metrics_overlay(debug_display, result, self.performance_stats['avg_fps'])
                    cv2.imshow('Debug View', debug_display)
                    cv2.destroyWindow('Enhanced Human Segmentation')
                    cv2.destroyWindow('Metrics View')
                    
                else:
                    # Metrics view
                    metrics_display = self.create_metrics_display(result)
                    cv2.imshow('Metrics View', metrics_display)
                    cv2.destroyWindow('Enhanced Human Segmentation')
                    cv2.destroyWindow('Debug View')
                
                # Handle controls
                key = self.handle_controls()
                if key == 'quit':
                    break
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def update_performance_stats(self, processing_time):
        """Update comprehensive performance statistics"""
        self.performance_stats['frame_count'] += 1
        current_fps = 1.0 / processing_time if processing_time > 0 else 0
        
        # Exponential moving averages
        alpha = 0.1
        self.performance_stats['avg_fps'] = (
            alpha * current_fps + 
            (1 - alpha) * self.performance_stats['avg_fps']
        )
        self.performance_stats['avg_processing_time'] = (
            alpha * processing_time + 
            (1 - alpha) * self.performance_stats['avg_processing_time']
        )
        
        # Min/Max tracking
        self.performance_stats['min_fps'] = min(self.performance_stats['min_fps'], current_fps)
        self.performance_stats['max_fps'] = max(self.performance_stats['max_fps'], current_fps)

    def handle_controls(self):
        """Handle keyboard controls"""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            return 'quit'
        elif key == ord('t'):
            self.view_mode = (self.view_mode + 1) % 3
            mode_names = ['Final Output', 'Debug View', 'Metrics View']
            print(f"View mode: {mode_names[self.view_mode]}")
            time.sleep(0.2)
        elif key == ord('b'):
            try:
                self.processor.bg_tensor = self.processor.load_background()
                print("✓ Background reloaded")
            except Exception as e:
                print(f"✗ Failed to reload background: {e}")
        elif key == ord('r'):
            self.processor.segmenter.reset_states()
            print("✓ Recurrent states reset")
        elif key == ord('p'):
            self.print_performance_report()
            
        return 'continue'

    def print_performance_report(self):
        """Print detailed performance report"""
        print("\n" + "="*50)
        print("PERFORMANCE REPORT")
        print("="*50)
        print(f"Total Frames Processed: {self.performance_stats['frame_count']}")
        print(f"Average FPS: {self.performance_stats['avg_fps']:.1f}")
        print(f"Min FPS: {self.performance_stats['min_fps']:.1f}")
        print(f"Max FPS: {self.performance_stats['max_fps']:.1f}")
        print(f"Average Processing Time: {self.performance_stats['avg_processing_time']*1000:.1f}ms")
        print("="*50)

    def cleanup(self):
        """Cleanup resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        self.print_performance_report()

# ======================
# MAIN EXECUTION
# ======================
if __name__ == "__main__":
    app = EnhancedHumanSegmentationApp()
    app.run()