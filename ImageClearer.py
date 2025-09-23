import cv2
import numpy as np

def image_clearer(image_path: str):
    img = cv2.imread(image_path)

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
    img = field_resized.copy()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # type: ignore

    # Detect yellow and white lines in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)


    # Detect black lines (like in play0)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 60])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    # Combine masks
    mask = cv2.bitwise_or(mask_yellow, mask_black)


    # Use this mask instead of plain threshold
    thresh = mask


    gray = thresh.copy()

    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)


    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if length > 200:  # ignore very long lines (likely field lines)
            continue
        angle = np.degrees(np.arctan2(y2-y1, x2-x1))
        if abs(angle) < 10 or abs(angle) > 170:  # almost horizontal
            continue

        cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 2)
        # print(f"Line from ({x1}, {y1}) to ({x2}, {y2})")

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h if h > 0 else 0
        if area > 100 and (aspect_ratio > 2 or aspect_ratio < 0.5):
            cv2.drawContours(img, [cnt], -1, (0,255,0), 2)
            approx = cv2.approxPolyDP(cnt, 0.04*cv2.arcLength(cnt, True), True)
            if len(approx) == 3:
                cv2.drawContours(img, [cnt], -1, (255,0,0), 2)  # arrow tip


    # Ensure background is black
    white_pixels = cv2.countNonZero(thresh)
    black_pixels = thresh.size - white_pixels
    if white_pixels > black_pixels:
        # print("⚠️ Inverting image to keep background black")
        thresh = cv2.bitwise_not(thresh)

    # cv2.imshow("Final Threshold", thresh)
    # cv2.imwrite("images/line_detection_result.png", thresh)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
