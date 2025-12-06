import cv2
import numpy as np
import time

class HandProximityAlertSystem:
    def __init__(self, camera_index=0):
        """
        Initialize the hand proximity alert system.
        
        Args:
            camera_index: Index of the camera to use (default: 0)
        """
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise Exception("Could not open camera")
        
        # Get frame dimensions
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Processing dimensions (for speed)
        self.processing_width = 640
        self.processing_height = 480
        
        # HSV skin color ranges (adjust based on lighting and skin tone)
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Optional: Additional range for some skin tones
        self.lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
        self.upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
        
        # Virtual object/boundary parameters
        self.virtual_object_x = int(self.processing_width * 0.8)  # Right boundary
        self.virtual_object_width = 10  # Thickness of the boundary
        
        # Distance thresholds (in pixels)
        self.warning_threshold = 100
        self.danger_threshold = 50
        
        # State colors
        self.state_colors = {
            'SAFE': (0, 255, 0),      # Green
            'WARNING': (0, 255, 255), # Yellow
            'DANGER': (0, 0, 255)     # Red
        }
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Morphological kernels
        self.kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Optional background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
        
        # Smoothing parameters
        self.distance_history = []
        self.history_length = 5
        
    def preprocess_frame(self, frame):
        """Resize and preprocess the frame."""
        # Resize for faster processing
        frame = cv2.resize(frame, (self.processing_width, self.processing_height))
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        
        return blurred
    
    def skin_segmentation(self, frame):
        """Segment skin-colored regions using HSV color space."""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create skin masks
        mask1 = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        mask2 = cv2.inRange(hsv, self.lower_skin2, self.upper_skin2)
        
        # Combine masks
        skin_mask = cv2.bitwise_or(mask1, mask2)
        
        bg_mask = self.bg_subtractor.apply(frame)
        combined_mask = cv2.bitwise_and(skin_mask, bg_mask)
        
        # Apply morphological operations to clean up the mask
        mask = cv2.erode(combined_mask, self.kernel_erode, iterations=1)
        mask = cv2.dilate(mask, self.kernel_dilate, iterations=2)
        mask = cv2.erode(mask, self.kernel_erode, iterations=1)
        
        return mask
    
    def find_hand_contour(self, mask):
        """Find the largest contour (assumed to be the hand)."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None, None
        
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Filter out small contours (noise)
        if cv2.contourArea(largest_contour) < 1000:
            return None, None, None
        
        # Calculate contour properties
        contour_area = cv2.contourArea(largest_contour)
        
        # Get convex hull and defects for hand analysis
        hull = cv2.convexHull(largest_contour, returnPoints=False)
        
        # Get extreme points
        ext_left = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
        ext_right = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])
        ext_top = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
        ext_bottom = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])
        
        extreme_points = {
            'left': ext_left,
            'right': ext_right,
            'top': ext_top,
            'bottom': ext_bottom
        }
        
        # centroid
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            palm_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        else:
            # Use center of extreme points as fallback
            palm_center = ((ext_left[0] + ext_right[0]) // 2, 
                          (ext_top[1] + ext_bottom[1]) // 2)
        
        return largest_contour, extreme_points, palm_center
    
    def calculate_distance_to_boundary(self, contour, palm_center):
        """Calculate minimum distance from hand to virtual boundary."""
        # Method 1: Distance from palm center to boundary
        distance_to_boundary = abs(palm_center[0] - self.virtual_object_x)
        
        # Method 2: Minimum distance from any contour point to boundary
        if contour is not None:
            contour_points = contour.reshape(-1, 2)
            min_contour_distance = np.min(np.abs(contour_points[:, 0] - self.virtual_object_x))
            # Use the smaller of the two distances
            distance_to_boundary = min(distance_to_boundary, min_contour_distance)
        
        return distance_to_boundary
    
    def classify_state(self, distance):
        """Classify proximity state based on distance thresholds."""
        # Apply smoothing to avoid flickering
        self.distance_history.append(distance)
        if len(self.distance_history) > self.history_length:
            self.distance_history.pop(0)
        
        smoothed_distance = np.mean(self.distance_history)
        
        # Classify based on thresholds
        if smoothed_distance <= self.danger_threshold:
            return 'DANGER', smoothed_distance
        elif smoothed_distance <= self.warning_threshold:
            return 'WARNING', smoothed_distance
        else:
            return 'SAFE', smoothed_distance
    
    def draw_visual_feedback(self, frame, contour, extreme_points, palm_center, state, distance):
        """Draw all visual feedback elements on the frame."""
        # Draw the virtual boundary
        cv2.line(frame, 
                (self.virtual_object_x, 0), 
                (self.virtual_object_x, self.processing_height),
                (255, 255, 255), 2)
        cv2.line(frame, 
                (self.virtual_object_x - self.virtual_object_width, 0), 
                (self.virtual_object_x - self.virtual_object_width, self.processing_height),
                (100, 100, 255), 2)
        
        # Draw hand contour if found
        if contour is not None:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            
            # Draw extreme points
            for point_name, point in extreme_points.items():
                cv2.circle(frame, point, 8, (0, 0, 255), -1)
                cv2.putText(frame, point_name, 
                          (point[0] + 10, point[1]), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw palm center
            cv2.circle(frame, palm_center, 10, (255, 0, 0), -1)
            
            # Draw distance line from palm to boundary
            cv2.line(frame, palm_center, 
                    (self.virtual_object_x, palm_center[1]), 
                    (255, 255, 0), 2)
            
            # Display distance
            cv2.putText(frame, f"Distance: {distance:.1f}px", 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw state indicator with appropriate color
        state_color = self.state_colors[state]
        
        # Create state display
        if state == 'DANGER':
            # Large warning for DANGER state
            cv2.putText(frame, "DANGER! DANGER!", 
                       (self.processing_width // 4, self.processing_height // 2),
                       cv2.FONT_HERSHEY_TRIPLEX, 2, state_color, 4)
            cv2.putText(frame, "TOO CLOSE!", 
                       (self.processing_width // 4, self.processing_height // 2 + 60),
                       cv2.FONT_HERSHEY_TRIPLEX, 1.5, state_color, 3)
        else:
            # Normal state display
            cv2.putText(frame, f"STATE: {state}", 
                       (10, 60), cv2.FONT_HERSHEY_TRIPLEX, 1.5, state_color, 3)
        
        # Draw threshold zones
        cv2.putText(frame, f"Warning Zone (< {self.warning_threshold}px)", 
                   (self.virtual_object_x - self.warning_threshold - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.state_colors['WARNING'], 2)
        cv2.putText(frame, f"Danger Zone (< {self.danger_threshold}px)", 
                   (self.virtual_object_x - self.danger_threshold - 150, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.state_colors['DANGER'], 2)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", 
                   (10, self.processing_height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def process_frame(self, frame):
        """Process a single frame and return annotated result."""
        # Preprocess
        processed_frame = self.preprocess_frame(frame)
        
        # Skin segmentation
        mask = self.skin_segmentation(processed_frame)
        
        # Find hand contour
        contour, extreme_points, palm_center = self.find_hand_contour(mask)
        
        # Calculate distance and classify state
        if contour is not None and palm_center is not None:
            distance = self.calculate_distance_to_boundary(contour, palm_center)
            state, smoothed_distance = self.classify_state(distance)
        else:
            distance = float('inf')
            state = 'SAFE'
            smoothed_distance = float('inf')
        
        # Draw visual feedback
        annotated_frame = self.draw_visual_feedback(
            processed_frame.copy(), contour, extreme_points, 
            palm_center, state, smoothed_distance
        )
        
        # Display mask for debugging
        mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Combine original and processed frames
        combined = np.hstack([annotated_frame, mask_display])
        
        return combined, state, smoothed_distance
    
    def run(self):
        """Main loop to run the hand proximity alert system."""
        print("Starting Hand Proximity Alert System")
        print("Place your hand in front of the camera")
        print("Move your hand toward the right boundary to trigger alerts")
        print("Press 'q' to quit, 'r' to reset, '+'/- to adjust thresholds")
        
        while True:
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Process frame
            processed_frame, state, distance = self.process_frame(frame)
            
            # Calculate FPS
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                elapsed_time = time.time() - self.start_time
                self.fps = self.frame_count / elapsed_time
            
            # Display result
            cv2.imshow('Hand Proximity Alert System', processed_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset background subtractor
                self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, 
                                                                       varThreshold=25, 
                                                                       detectShadows=False)
                print("Background model reset")
            elif key == ord('+'):
                # Increase thresholds
                self.warning_threshold += 10
                self.danger_threshold += 10
                print(f"Thresholds increased: Warning={self.warning_threshold}, Danger={self.danger_threshold}")
            elif key == ord('-'):
                # Decrease thresholds
                self.warning_threshold = max(20, self.warning_threshold - 10)
                self.danger_threshold = max(10, self.danger_threshold - 10)
                print(f"Thresholds decreased: Warning={self.warning_threshold}, Danger={self.danger_threshold}")
            elif key == ord('c'):
                # Calibrate skin color
                self.calibrate_skin_color(frame)
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
    
    def calibrate_skin_color(self, frame):
        """Simple calibration for skin color range."""
        print("\n=== Skin Color Calibration ===")
        print("1. Place your hand in the center of the frame")
        print("2. Press any key when ready...")
        
        # Show calibration frame
        calibration_frame = cv2.resize(frame, (self.processing_width, self.processing_height))
        cv2.putText(calibration_frame, "CALIBRATION: Place hand in center", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Calibration', calibration_frame)
        cv2.waitKey(0)
        
        # Convert to HSV and sample skin color
        hsv = cv2.cvtColor(calibration_frame, cv2.COLOR_BGR2HSV)
        
        # Sample center region
        center_x, center_y = self.processing_width // 2, self.processing_height // 2
        sample_region = hsv[center_y-50:center_y+50, center_x-50:center_x+50]
        
        if sample_region.size > 0:
            # Calculate mean and standard deviation
            mean_hue = np.mean(sample_region[:,:,0])
            std_hue = np.std(sample_region[:,:,0])
            
            # Update skin color ranges
            self.lower_skin[0] = max(0, int(mean_hue - 2 * std_hue))
            self.upper_skin[0] = min(20, int(mean_hue + 2 * std_hue))
            
            print(f"Updated skin color range: H={self.lower_skin[0]}-{self.upper_skin[0]}")
        
        cv2.destroyWindow('Calibration')

# Main execution
if __name__ == "__main__":
    try:
        system = HandProximityAlertSystem(camera_index=0)
        system.run()
    except Exception as e:
        print(f"Error: {e}")