import cv2
import numpy as np
from CircleDetector import circle_detector
from ImageClearer import image_clearer
import json


def classify_contour(cnt):
    """Return 'circle', 'x', or 'path'."""
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return "unknown"

    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    x, y, w, h = cv2.boundingRect(cnt)
    aspect = w / float(h)

    # --- Circle detection ---
    if 0.7 < circularity <= 1.2 and 0.8 < aspect < 1.2 and 50 < area < 1500:
        return "circle"

    # --- X detection ---
    if circularity < 0.6 and 0.8 < aspect < 1.3 and 100 < area < 3000:
        hull = cv2.convexHull(cnt, returnPoints=False)
        if hull is not None and len(hull) > 3:
            defects = cv2.convexityDefects(cnt, hull)
            if defects is not None and len(defects) >= 2:
                return "x"

    # --- Path detection ---
    if aspect > 2.0 or aspect < 0.5 or area > 1500:
        return "path"

    return "unknown"

def get_image_just_lines():
    img = img_cleared
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Threshold or edge detection
    edges = cv2.Canny(gray, 80, 200)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    edges = cv2.erode(edges, np.ones((3,3), np.uint8), iterations=1)

    # Find contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours
    min_contour_length = 20  # adjust as needed
    filtered_contours = [cnt for cnt in contours if cv2.arcLength(cnt, False) > min_contour_length]

    # Draw contours
    for cnt in filtered_contours:
        shape = classify_contour(cnt)
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        if shape == "circle":
            cv2.circle(img, (cx, cy), 12, (0, 0, 0), -1)
        elif shape == "x":
            cv2.drawMarker(img, (cx, cy), (0, 0, 0),
                           markerType=cv2.MARKER_TILTED_CROSS,
                           markerSize=20, thickness=10)
    
    positions, radius = circlePositions, circleRadius
    #hide circles
    for (cx, cy) in positions:
        cv2.circle(img, (cx, cy), int(radius * 2), (0, 0, 0), -1)
    return img

def merge_similar_contours(contours, dist_thresh=20, overlap_thresh=0.3):
    """
    Merge contours that represent the same drawn path, avoiding cross-line bridges.
    """
    merged = []
    used = set()

    for i, c1 in enumerate(contours):
        if i in used:
            continue
        x1, y1, w1, h1 = cv2.boundingRect(c1)
        area1 = w1 * h1
        merged_cnt = c1.copy()

        for j, c2 in enumerate(contours):
            if i == j or j in used:
                continue
            x2, y2, w2, h2 = cv2.boundingRect(c2)
            area2 = w2 * h2

            # 1️⃣ bounding box overlap
            ix1, iy1 = max(x1, x2), max(y1, y2)
            ix2, iy2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
            inter_w, inter_h = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter_area = inter_w * inter_h
            overlap = inter_area / float(min(area1, area2))

            # 2️⃣ endpoint distance (to avoid connecting far-away ends)
            p1_start, p1_end = c1[0][0], c1[-1][0]
            p2_start, p2_end = c2[0][0], c2[-1][0]
            dists = [
                np.linalg.norm(p1_start - p2_start),
                np.linalg.norm(p1_start - p2_end),
                np.linalg.norm(p1_end - p2_start),
                np.linalg.norm(p1_end - p2_end),
            ]
            min_dist = min(dists)

            # 3️⃣ Merge only if overlap OR close endpoints
            if overlap > overlap_thresh or min_dist < dist_thresh:
                merged_cnt = np.vstack((merged_cnt, c2))
                used.add(j)

        merged.append(merged_cnt)
        used.add(i)

    return merged

def filter_contours(gray: np.ndarray):
    # detect edges
    edges = cv2.Canny(gray, 80, 200)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    edges = cv2.erode(edges, np.ones((3,3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # filter small noise
    contours = [c for c in contours if cv2.arcLength(c, False) > 60]
    contours = merge_similar_contours(contours)
    return contours

def export_to_play_json_from_contours(homePositions, awayPositions, circleRadius, endpoints, contours, img_shape, play_name="Offensive Play 2", filename="offensive_play.json"):
    h, w = img_shape[:2]

    def norm(x, y):
        return round(x / w, 9), round(1.0 - (y / h), 9)

    def round9(val):
        return round(float(val), 9)

    def build_player_entry(cx, cy):
        return {
            "position": {
                "x": round9(cx / w),
                "y": round9(1.0 - (cy / h)),
                "z": 0.0
            },
            "run": {
                "runType": 0,
                "waypoints": [],
                "maxSpeed": 5.0,
                "endStyleIndex": 0
            }
        }

    play_data = {
        "playName": play_name,
        "homePlayerData": [],
        "awayPlayerData": [],
        "footballPositions": [],
        "playTags": []
    }

    # helper to attach closest contour-based route
    def attach_run(player_entry, cx, cy):
        closest = None
        closest_d = float("inf")
        for i, (s, e) in enumerate(endpoints):
            for ep in (s, e):
                d = np.linalg.norm(np.array(ep) - np.array([cx, cy]))
                if d < closest_d and d < circleRadius * 3:
                    closest, closest_d = i, d

        if closest is not None:
            cnt = contours[closest].reshape(-1, 2)
            s, e = endpoints[closest]
            if np.linalg.norm(cnt[0] - np.array(s)) > np.linalg.norm(cnt[-1] - np.array(s)):
                cnt = np.flip(cnt, 0)
            waypoints = [
                {"x": round9(nx), "y": round9(ny), "z": 0.0}
                for (nx, ny) in [norm(x, y) for (x, y) in cnt]
            ]
            player_entry["run"]["waypoints"] = waypoints

    # home team
    for (cx, cy) in homePositions:
        player = build_player_entry(cx, cy)
        attach_run(player, cx, cy)
        play_data["homePlayerData"].append(player)

    # away team
    for (cx, cy) in awayPositions:
        player = build_player_entry(cx, cy)
        attach_run(player, cx, cy)
        play_data["awayPlayerData"].append(player)

    with open(filename, "w") as f:
        json.dump(play_data, f, indent=2)


def run_detection():
    global homePositions, awayPositions, circlePositions, circleRadius

    img = get_image_just_lines()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours = filter_contours(gray)

    output = np.zeros_like(img)
    endpoints = []  # [(start_point, end_point)]
    positions, radius = circlePositions, circleRadius


    # --- Step 1: find endpoints for each run ---
    for cnt in contours:
        pts = cnt.reshape(-1, 2)
        max_dist = 0
        start_point = end_point = pts[0]
        for p1 in pts[::max(1, len(pts)//50)]:
            for p2 in pts[::max(1, len(pts)//50)]:
                dist = np.linalg.norm(p1 - p2)
                if dist > max_dist:
                    max_dist = dist
                    start_point, end_point = p1, p2
        endpoints.append((tuple(start_point.astype(int)), tuple(end_point.astype(int))))

    endpoint_to_circle = {}  # maps endpoint -> (circle_idx, distance)

    # --- Step 2: assign endpoints to nearest circle if unique ---
    for ci, (cx, cy) in enumerate(positions):
        for ri, (start, end) in enumerate(endpoints):
            for ep in [start, end]:
                dist = np.linalg.norm(np.array([cx, cy]) - np.array(ep))
                if dist < radius * 3:
                    if ep in endpoint_to_circle:
                        prev_ci, prev_dist = endpoint_to_circle[ep]
                        if dist < prev_dist:
                            endpoint_to_circle[ep] = (ci, dist)
                    else:
                        endpoint_to_circle[ep] = (ci, dist)

    circle_colors = {}

    # --- 🔍 Step 3: filter contours not connected to any circle ---
    kept_contours = []
    kept_endpoints = []
    for i, (start_point, end_point) in enumerate(endpoints):
        start_is_connected = start_point in endpoint_to_circle
        end_is_connected = end_point in endpoint_to_circle

        if start_is_connected or end_is_connected:
            kept_contours.append(contours[i])
            kept_endpoints.append((start_point, end_point))
        # else → discard this contour silently

    contours = kept_contours
    endpoints = kept_endpoints

    # --- Step 4: draw kept contours ---
    for i, cnt in enumerate(contours):
        line_color = tuple(np.random.randint(100, 255, size=3).tolist())
        cv2.drawContours(output, [cnt], -1, line_color, 2)

        # ✅ Add evenly spaced red points along the contour
        spacing = 5  # pixels between red points
        pts = cnt.reshape(-1, 2)
        for k in range(0, len(pts), spacing):
            cv2.circle(output, tuple(pts[k]), 3, (0, 0, 255), -1)

        start_point, end_point = endpoints[i]
        start_is_connected = start_point in endpoint_to_circle
        end_is_connected = end_point in endpoint_to_circle

        if start_is_connected and not end_is_connected:
            start_color, end_color = (255, 0, 0), (0, 255, 0)
            circle_idx = endpoint_to_circle[start_point][0]
            circle_colors[circle_idx] = start_color
        elif end_is_connected and not start_is_connected:
            start_color, end_color = (0, 255, 0), (255, 0, 0)
            circle_idx = endpoint_to_circle[end_point][0]
            circle_colors[circle_idx] = end_color
        elif start_is_connected and end_is_connected:
            s_circle, s_dist = endpoint_to_circle[start_point]
            e_circle, e_dist = endpoint_to_circle[end_point]
            if s_dist <= e_dist:
                start_color, end_color = (255, 0, 0), (0, 255, 0)
                circle_colors[s_circle] = start_color
            else:
                start_color, end_color = (0, 255, 0), (255, 0, 0)
                circle_colors[e_circle] = end_color
        else:
            start_color, end_color = (0, 0, 255), (0, 0, 255)

        cv2.circle(output, start_point, 6, start_color, -1)
        cv2.circle(output, end_point, 6, end_color, -1)

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(output, f"Run {i+1}", (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 2)

    # --- Step 5: draw circles (matching start color) ---
    for ci, (cx, cy) in enumerate(positions):
        color = circle_colors.get(ci, (0, 255, 0))
        cv2.circle(output, (cx, cy), int(radius * 1.2), color, -1)

    # cv2.imshow("Separated Routes", output)
    export_to_play_json_from_contours(homePositions, awayPositions, circleRadius, endpoints, contours, output.shape)






# index = 0    
# path = f'images/play{index}.png'
# img_cleared = image_clearer(path)
# homePositions, awayPositions, circlePositions, circleRadius = circle_detector(path, img_cleared)


# run_detection()

# cv2.waitKey(0)
# cv2.destroyAllWindows()