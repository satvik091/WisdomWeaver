from deepface import DeepFace
import cv2
import numpy as np
import threading
import time
from collections import deque
import queue
import logging

class AdvancedEmotionDetector:
    def __init__(self, fallback_emotion="neutral", verbose=False):
        # Load face detection models
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Threading components
        self.emotion_queue = queue.Queue(maxsize=2)
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=2)
        
        # State variables
        self.current_emotion = "Initializing..."
        self.emotion_confidence = {}
        self.is_processing = False
        self.stop_threads = False
        
        # Performance settings
        self.skip_frames = 2
        self.frame_count = 0
        
        # Emotion smoothing (reduces flickering)
        self.emotion_history = deque(maxlen=5)
        
        # Customizable fallback and verbose settings
        self.fallback_emotion = fallback_emotion
        self.verbose = verbose
        logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._emotion_processing_loop, daemon=True)
        self.processing_thread.start()
        
        print("Advanced Emotion Detector initialized!")

    def _emotion_processing_loop(self):
        """Background thread for emotion processing"""
        while not self.stop_threads:
            try:
                if not self.frame_queue.empty():
                    face_roi = self.frame_queue.get(timeout=0.1)
                    
                    # Process emotion
                    emotion, confidence = self._analyze_emotion_internal(face_roi)
                    
                    if emotion:
                        # Add to history for smoothing
                        self.emotion_history.append(emotion)
                        
                        # Get most common emotion from recent history
                        smoothed_emotion = max(set(self.emotion_history), 
                                             key=self.emotion_history.count)
                        
                        # Update results
                        if not self.result_queue.full():
                            self.result_queue.put((smoothed_emotion, confidence))
                
                time.sleep(0.01)  # Small delay to prevent CPU overload
            except queue.Empty:
                continue
            except Exception as e:
                if self.verbose:
                    logging.warning(f"Processing error: {e}")

    def _analyze_emotion_internal(self, face_roi):
        """Internal emotion analysis method with improved neutral handling"""
        try:
            # Ensure minimum size for better accuracy
            if face_roi.shape[0] < 100 or face_roi.shape[1] < 100:
                face_roi = cv2.resize(face_roi, (224, 224))
            
            # Enhance image quality
            face_roi = cv2.convertScaleAbs(face_roi, alpha=1.2, beta=10)
            
            analysis = DeepFace.analyze(
                face_roi,
                actions=['emotion'],
                enforce_detection=False,
                silent=True,
                detector_backend='opencv'  # Faster backend
            )
            
            if isinstance(analysis, list):
                emotion_data = analysis[0]['emotion']
                dominant_emotion = analysis[0]['dominant_emotion']
            else:
                emotion_data = analysis['emotion']
                dominant_emotion = analysis['dominant_emotion']
            
            # Smart neutral handling - show 2nd best if neutral confidence < 98%
            if dominant_emotion == 'neutral':
                neutral_confidence = emotion_data['neutral']
                
                if neutral_confidence < 95.0:
                    # Find alternative emotions sorted by confidence
                    sorted_emotions = sorted(emotion_data.items(), key=lambda x: x[1], reverse=True)
                    
                    # Look for the best non-neutral emotion
                    for emotion_name, confidence_score in sorted_emotions[1:]:  # Skip neutral (first one)
                        if emotion_name == 'angry':
                            if confidence_score > 12.0:
                                dominant_emotion = emotion_name
                                if self.verbose:
                                    logging.warning(f"Neutral confidence {neutral_confidence:.1f}% < 95%, using {emotion_name} ({confidence_score:.1f}%)")
                                break
                            else:
                                if self.verbose:
                                    logging.warning(f"Skipping angry ({confidence_score:.1f}%) - below 12% threshold")
                                continue
                        elif emotion_name == 'sad':
                            if confidence_score > 2.5:
                                dominant_emotion = emotion_name
                                if self.verbose:
                                    logging.warning(f"Neutral confidence {neutral_confidence:.1f}% < 95%, using {emotion_name} ({confidence_score:.1f}%)")
                                break
                            else:
                                if self.verbose:
                                    logging.warning(f"Skipping sad ({confidence_score:.1f}%) - below 2.5% threshold")
                                continue
                        else:
                            if confidence_score > 5.0:
                                dominant_emotion = emotion_name
                                if self.verbose:
                                    logging.warning(f"Neutral confidence {neutral_confidence:.1f}% < 95%, using {emotion_name} ({confidence_score:.1f}%)")
                                break
            
            # Special handling for happy - only display if confidence > 70%
            if dominant_emotion == 'happy':
                happy_confidence = emotion_data['happy']
                
                if happy_confidence <= 70.0:
                    # Find alternative emotions sorted by confidence
                    sorted_emotions = sorted(emotion_data.items(), key=lambda x: x[1], reverse=True)
                    
                    # Look for the best non-happy emotion
                    for emotion_name, confidence_score in sorted_emotions:
                        if emotion_name != 'happy' and confidence_score > 5.0:
                            dominant_emotion = emotion_name
                            if self.verbose:
                                logging.warning(f"Happy confidence {happy_confidence:.1f}% <= 70%, using {emotion_name} ({confidence_score:.1f}%)")
                            break
                    else:
                        # If no alternative found, fall back to custom fallback
                        if self.verbose:
                            logging.warning(f"No suitable emotion found, falling back to {self.fallback_emotion}")
                        dominant_emotion = self.fallback_emotion
            
            return dominant_emotion, emotion_data
            
        except Exception as e:
            if self.verbose:
                logging.warning(f"Error in emotion analysis: {e}, falling back to {self.fallback_emotion}")
            return self.fallback_emotion, None

    def detect_faces_optimized(self, frame):
        """Optimized face detection with single best face selection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better detection
        gray = cv2.equalizeHist(gray)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=8,
            minSize=(80, 80),
            maxSize=(400, 400),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) > 0:
            scored_faces = []
            frame_center_x, frame_center_y = frame.shape[1] // 2, frame.shape[0] // 2
            
            for (x, y, w, h) in faces:
                area = w * h
                face_center_x, face_center_y = x + w // 2, y + h // 2
                distance_from_center = ((face_center_x - frame_center_x) ** 2 + 
                                      (face_center_y - frame_center_y) ** 2) ** 0.5
                aspect_ratio = min(w, h) / max(w, h)
                score = area * aspect_ratio * (1 / (1 + distance_from_center * 0.01))
                scored_faces.append((score, (x, y, w, h)))
            
            best_face = max(scored_faces, key=lambda x: x[0])[1]
            return [best_face]
        
        return faces

    def update_emotion_async(self, face_roi):
        """Add face region to processing queue"""
        if not self.frame_queue.full():
            self.frame_queue.put(face_roi.copy())

    def get_current_emotion(self):
        """Get the latest emotion result"""
        try:
            while not self.result_queue.empty():
                emotion, confidence = self.result_queue.get_nowait()
                self.current_emotion = emotion
                self.emotion_confidence = confidence
        except queue.Empty:
            pass
        
        return self.current_emotion, self.emotion_confidence

    def draw_advanced_results(self, frame, faces):
        """Draw enhanced visualization with emotion details"""
        emotion, confidence = self.get_current_emotion()
        
        for i, (x, y, w, h) in enumerate(faces):
            # Draw face rectangle with rounded corners effect
            cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), (0, 255, 0), 3)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
            
            # Draw emotion text with background
            if emotion and emotion != "Initializing...":
                # Background rectangle for text
                text_size = cv2.getTextSize(f"Emotion: {emotion}", 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(frame, (x, y-35), (x + text_size[0] + 10, y-5), (0, 0, 0), -1)
                
                # Emotion text
                cv2.putText(
                    frame,
                    f"Emotion: {emotion}",
                    (x+5, y-15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
                
                # Confidence bar and details
                if confidence:
                    # Get current emotion confidence
                    current_conf = confidence.get(emotion, 0)
                    
                    # Draw confidence bar
                    bar_width = int((w * current_conf) / 100)
                    cv2.rectangle(frame, (x, y+h+5), (x+bar_width, y+h+15), (0, 255, 0), -1)
                    cv2.rectangle(frame, (x, y+h+5), (x+w, y+h+15), (255, 255, 255), 1)
                    
                    # Confidence percentage
                    cv2.putText(
                        frame,
                        f"{current_conf:.1f}%",
                        (x+w-60, y+h+25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )
                    
                    # Show top 3 emotions with scores
                    sorted_emotions = sorted(confidence.items(), key=lambda x: x[1], reverse=True)[:3]
                    y_offset = y + h + 40
                    
                    for idx, (emo, score) in enumerate(sorted_emotions):
                        if score > 1:  # Only show emotions with >1% confidence
                            color = (0, 255, 0) if emo == emotion else (255, 255, 255)
                            cv2.putText(
                                frame,
                                f"{idx+1}. {emo}: {score:.1f}%",
                                (x, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                color,
                                1
                            )
                            y_offset += 15

    def cleanup(self):
        """Clean up resources"""
        self.stop_threads = True
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1)

def main_advanced():
    detector = AdvancedEmotionDetector()
    
    # Initialize video capture with optimal settings
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Optimize camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
    
    print("Advanced emotion detection started!")
    print("Controls:")
    print("- Press 'q' to quit")
    print("- Press 'r' to reset emotion history")
    print("- Press 'd' to toggle debug mode")
    
    fps_counter = deque(maxlen=30)
    last_time = time.time()
    debug_mode = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to grab frame.")
                break
            
            # Mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect faces
            faces = detector.detect_faces_optimized(frame)
            
            # Process emotion asynchronously
            if len(faces) > 0 and detector.frame_count % detector.skip_frames == 0:
                # Use the single best face (already filtered by detect_faces_optimized)
                x, y, w, h = faces[0]  # Only one face now
                
                # Extract face with padding
                padding = 30
                y1 = max(0, y - padding)
                y2 = min(frame.shape[0], y + h + padding)
                x1 = max(0, x - padding)
                x2 = min(frame.shape[1], x + w + padding)
                
                face_roi = frame[y1:y2, x1:x2]
                
                if face_roi.size > 0:
                    detector.update_emotion_async(face_roi)
            
            # Draw results
            if len(faces) > 0:
                detector.draw_advanced_results(frame, faces)
            else:
                cv2.putText(
                    frame,
                    "No face detected - Move closer to camera",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
            
            # Calculate and display FPS
            current_time = time.time()
            fps_counter.append(1.0 / (current_time - last_time))
            last_time = current_time
            avg_fps = sum(fps_counter) / len(fps_counter)
            
            cv2.putText(
                frame,
                f"FPS: {avg_fps:.1f}",
                (frame.shape[1] - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            # Display queue status
            queue_status = f"Queue: {detector.frame_queue.qsize()}"
            cv2.putText(
                frame,
                queue_status,
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            cv2.imshow("Advanced Emotion Detection", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                detector.emotion_history.clear()
                print("Emotion history reset!")
            elif key == ord('d'):
                debug_mode = not debug_mode
                print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
            
            detector.frame_count += 1
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    finally:
        detector.cleanup()
        cap.release()
        cv2.destroyAllWindows()
        print("Advanced emotion detection stopped.")

if __name__ == "__main__":
    detector = AdvancedEmotionDetector(fallback_emotion="calm", verbose=True)
    # Force a failure by passing None or invalid data
    emotion, _ = detector._analyze_emotion_internal(None)  # Direct call to internal method
    print(f"Current emotion: {emotion}")