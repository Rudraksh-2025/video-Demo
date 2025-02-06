import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(0)  # or 'path/to/video.mp4'
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    ret, frame1 = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return
    
    # 1. Select an ROI (bounding box) in the first frame
    #    Use your mouse to draw a rectangle around the object.
    roi = cv2.selectROI("Select ROI", frame1, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")  # close the ROI selection window
    
    x, y, w, h = [int(v) for v in roi]
    
    # Convert the first frame to grayscale
    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    # Prepare HSV image for flow visualization
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255  # set saturation to maximum

    while True:
        ret, frame2 = cap.read()
        if not ret:
            print("No more frames or camera disconnected.")
            break
        
        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # 2. Calculate Dense Optical Flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Convert flow to polar coordinates: magnitude and angle
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Use HSV color model to visualize flow field
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 3. Compute average flow inside the current ROI
        #    This is a simple way to estimate how the object moved.
        flow_x = flow[y:y+h, x:x+w, 0]
        flow_y = flow[y:y+h, x:x+w, 1]
        
        # If the ROI is valid (non-zero width/height, inside the frame)
        if flow_x.size > 0 and flow_y.size > 0:
            mean_flow_x = np.mean(flow_x)
            mean_flow_y = np.mean(flow_y)
        else:
            mean_flow_x, mean_flow_y = 0, 0
        
        # 4. Update the ROI position based on average flow
        x += int(mean_flow_x)
        y += int(mean_flow_y)
        
        # (Optional) clamp the ROI so it doesn't go out of bounds
        #   especially if the object goes near edges
        frame_height, frame_width = frame2.shape[:2]
        x = max(0, min(x, frame_width - w))
        y = max(0, min(y, frame_height - h))
        
        # Draw the updated ROI on the "Original" frame
        cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Show both the original frame (with tracking box) and the flow
        cv2.imshow('Object Tracking (Dense Flow)', frame2)
        cv2.imshow('Dense Optical Flow Field', bgr_flow)
        
        prev_gray = gray  # update previous frame for next iteration
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
