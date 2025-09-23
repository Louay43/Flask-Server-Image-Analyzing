import cv2
import numpy as np
import math
from ImageClearer import image_clearer

def autoTuneContours(mask, expected_circles=11):
    """
    Finds the best area + circularity thresholds for contour detection.
    Loops through ranges and picks the one closest to expected_circles.
    """
    best_params = (50, 1000, 0.5)  # (min_area, max_area, circularity_threshold)
    best_score = float("inf")
    best_positions = []
    best_radiuses = []

    # Try different min/max area thresholds
    for min_area in range(200, 600, 100):       # ignore small dots
        for max_area in range(2000, 6000, 1000):  # allow large blobs
            for circ_thresh in [0.1, 0.2, 0.3]: # circularity thresholds
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                positions = []
                radiuses = []
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < min_area or area > max_area:
                        continue

                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter == 0:
                        continue

                    circularity = 4 * np.pi * (area / (perimeter * perimeter))
                    if circularity < circ_thresh:
                        continue

                    (x, y), r = cv2.minEnclosingCircle(cnt)
                    positions.append((int(x), int(y)))
                    radiuses.append(r)

                # Score this parameter set
                score = abs(len(positions) - expected_circles)

                if score < best_score:
                    best_score = score
                    best_params = (min_area, max_area, circ_thresh)
                    best_positions = positions
                    best_radiuses = radiuses

    average_radius = int(np.mean(best_radiuses)) if best_radiuses else 0
    return best_params, best_positions, average_radius

def getPlayerMask(hsv_field):
    """
    Returns a binary mask with all player colors (red, orange/yellow, white).
    """
    # ---- Red players ----
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    mask_red1 = cv2.inRange(hsv_field, lower_red1, upper_red1)

    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red2 = cv2.inRange(hsv_field, lower_red2, upper_red2)

    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # ---- Orange / Yellow players ----
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([25, 255, 255])
    mask_orange = cv2.inRange(hsv_field, lower_orange, upper_orange)

    # ---- White players ----
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])
    mask_white = cv2.inRange(hsv_field, lower_white, upper_white)

    # Combine all masks
    mask_players = cv2.bitwise_or(mask_red, mask_orange)
    mask_players = cv2.bitwise_or(mask_players, mask_white)

    return mask_players

def detectDuplicates(circles, min_dist_threshold=100):
    """
    Removes duplicate circles (those with centers too close together).
    circles: list of (x, y, r)
    min_dist_threshold: distance in pixels to consider duplicates
    """
    unique_circles = []
    for (x, y, r) in circles:
        too_close = False
        for (ux, uy, ur) in unique_circles:
            dist = math.sqrt((x - ux) ** 2 + (y - uy) ** 2)
            if dist < min_dist_threshold:
                too_close = True
                break
        if not too_close:
            unique_circles.append((x, y, r))
    return unique_circles

def autoTuneRadius(gray, min_r_start=2, max_r_end=50, expected_circles=11):
    """
    Finds the best min/max radius combination for HoughCircles.
    Loops through radius values and picks the one with circle count closest to expected_circles.
    """
    best_combo = (5, 20)
    best_score = float("inf")
    best_circles = None
    radiuses = []
    for min_r in range(min_r_start, max_r_end-5, 2):  # step by 2 to reduce loops
        for max_r in range(min_r+5, max_r_end+1, 2):
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=20,
                param1=100,
                param2=30,
                minRadius=min_r,
                maxRadius=max_r
            )
           
            if circles is not None:
                raw_circles = [(int(i[0]), int(i[1]), int(i[2])) for i in np.uint16(np.around(circles))[0, :]] # type: ignore
                unique_circles = detectDuplicates(raw_circles, min_dist_threshold=40)
                
                count = len(unique_circles)

                # Score = difference from expected
                score = abs(expected_circles - count)

                if score < best_score:
                    best_score = score
                    best_combo = (min_r, max_r)
                    best_circles = unique_circles

    if best_circles is None:
        return best_combo, [], 0
    
    radiuses = [r for (_, _, r) in best_circles]
    average_radius = int(np.mean(radiuses)) if radiuses else 0
    return best_combo, best_circles, average_radius

class PlayerDetectorWrapper:
    def __init__(self, expected_players=30):
        self.expected_players = expected_players

    def detect_with_circles(self, gray):
        best_combo, best_circles, radius = autoTuneRadius(
            gray, expected_circles=self.expected_players
        )
        positions = []
        if best_circles is not None:
            for (x, y, r) in best_circles:
                positions.append((x, y))
        return positions, radius

    def detect_with_contours(self, hsv_field):
        mask_players = getPlayerMask(hsv_field)
        kernel = np.ones((5,5), np.uint8)
        mask_players = cv2.morphologyEx(mask_players, cv2.MORPH_CLOSE, kernel)
        mask_players = cv2.morphologyEx(mask_players, cv2.MORPH_OPEN, kernel)

        _, positions, radius = autoTuneContours(mask_players, expected_circles=self.expected_players)
        return positions, radius

    def score(self, positions):
        return abs(len(positions) - self.expected_players)

    def detect_players(self, img):
        # Preprocess field
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # type: ignore

        # Define green range (tweak if needed)
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])

        # Mask the green field
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Find bounding box of field
        coords = cv2.findNonZero(mask)

        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            field = img[y:y+h, x:x+w] # type: ignore
        else:
            # print("⚠️ Could not detect field automatically, using full image.")
            field = img

        # Resize to a reasonable size (so HoughCircles works better)
        field_resized = cv2.resize(field, (650, 600)) # type: ignore

        gray = cv2.cvtColor(field_resized, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)
        
        hsv_field = cv2.cvtColor(field_resized, cv2.COLOR_BGR2HSV)

        # Run both methods
        pos_circles, radiusCircle = self.detect_with_circles(blur)
        pos_contours, radiusContour = self.detect_with_contours(hsv_field)

        # Score both
        score_circles = self.score(pos_circles)
        score_contours = self.score(pos_contours)

        # Choose better one
        if score_circles <= score_contours:
            # print(f"Circles chosen with score {score_circles}")
            return pos_circles, field_resized, radiusCircle
        else:
            # print(f"Contours chosen with score {score_contours}")
            return pos_contours, field_resized, radiusContour

def circle_detector(path_name: str, clr_image= np.ndarray):
    img = cv2.imread(path_name)   
    wrapper = PlayerDetectorWrapper(expected_players=30)
    positions, field_resized, radius = wrapper.detect_players(img)

    for (x, y) in positions:
        cv2.circle(field_resized, (x, y), radius, (0, 255, 0), 2)
        cv2.circle(field_resized, (x, y), 2, (0, 0, 255), 3)

    cleared_image = clr_image.copy()
    for (x, y) in positions:
        cv2.circle(cleared_image, (x, y), radius, (0, 0, 0), -1)  
        cv2.circle(cleared_image, (x, y), radius, (0, 0, 0), 2)   


    # cv2.imshow("Best Detection", field_resized)
    # cv2.imshow("On Cleared Image", cleared_image)
    # cv2.imwrite("images/line_detection_result.png", cleared_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return positions, radius

# path = 'images/play4.png'

# circle_detector(path, image_clearer(path))