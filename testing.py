import cv2
import numpy as np
from CircleDetector import circle_detector
from ImageClearer import image_clearer
import json
import os


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

    # --- X detection (improved stability) ---
    if circularity < 0.6 and 0.8 < aspect < 1.3 and 100 < area < 3000:

        epsilon = 0.01 * cv2.arcLength(cnt, True)
        cnt_smooth = cv2.approxPolyDP(cnt, epsilon, True)


        hull = cv2.convexHull(cnt_smooth, returnPoints=False)
        if hull is None or len(hull) < 4:
            return "unknown"

        hull = hull.flatten()
        if len(hull) < 3 or not np.all(np.diff(np.sort(hull)) >= 0):
            return "unknown"

        try:
            defects = cv2.convexityDefects(cnt_smooth, hull.reshape(-1, 1))
            if defects is not None and len(defects) >= 2:
                return "x"
        except cv2.error:
            # Do NOT discard all — fallback to shape reasoning
            if 0.5 < aspect < 1.5 and 200 < area < 2000:
                return "x"  # likely an X even if convex check fails
            return "unknown"

    return "unknown"



def get_image_just_lines(path: str, circlePositions, circleRadius):
    """Removes detected shapes and circles, leaving only player paths."""
    img = image_clearer(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 80, 200)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.erode(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.arcLength(cnt, False) > 20]

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
            cv2.drawMarker(img, (cx, cy), (0, 0, 0), cv2.MARKER_TILTED_CROSS, 20, 10)

    for (cx, cy) in circlePositions:
        cv2.circle(img, (cx, cy), int(circleRadius * 2), (0, 0, 0), -1)

    return img


def merge_similar_contours(contours, dist_thresh=20, overlap_thresh=0.3):
    """Merge contours that represent the same drawn path."""
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

            ix1, iy1 = max(x1, x2), max(y1, y2)
            ix2, iy2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
            inter_w, inter_h = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter_area = inter_w * inter_h
            overlap = inter_area / float(min(area1, area2))

            p1_start, p1_end = c1[0][0], c1[-1][0]
            p2_start, p2_end = c2[0][0], c2[-1][0]
            dists = [
                np.linalg.norm(p1_start - p2_start),
                np.linalg.norm(p1_start - p2_end),
                np.linalg.norm(p1_end - p2_start),
                np.linalg.norm(p1_end - p2_end),
            ]
            min_dist = min(dists)

            if overlap > overlap_thresh or min_dist < dist_thresh:
                merged_cnt = np.vstack((merged_cnt, c2))
                used.add(j)

        merged.append(merged_cnt)
        used.add(i)

    return merged


def filter_contours(gray: np.ndarray):
    """Detect and filter path contours."""
    edges = cv2.Canny(gray, 80, 200)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.erode(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.arcLength(c, False) > 60]
    return merge_similar_contours(contours)


def export_to_play_json_from_contours(homePositions, awayPositions, circleRadius, endpoints, contours, img_shape, play_name="Offensive Play 2", filename="offensive_play.json"):
    """Create the standard play JSON file."""
    h, w = img_shape[:2]

    def norm(x, y):
        return round(x / w, 9), round(1.0 - (y / h), 9)

    def round9(val):
        return round(float(val), 9)

    def build_player_entry(cx, cy):
        return {
            "position": {"x": round9(cx / w), "y": round9(1.0 - (cy / h)), "z": 0.0},
            "run": {"runType": 0, "waypoints": [], "maxSpeed": 5.0, "endStyleIndex": 0},
        }

    play_data = {
        "playName": play_name,
        "homePlayerData": [],
        "awayPlayerData": [],
        "footballPositions": [],
        "playTags": [],
    }

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

    for (cx, cy) in homePositions:
        player = build_player_entry(cx, cy)
        attach_run(player, cx, cy)
        play_data["homePlayerData"].append(player)

    for (cx, cy) in awayPositions:
        player = build_player_entry(cx, cy)
        attach_run(player, cx, cy)
        play_data["awayPlayerData"].append(player)

    with open(filename, "w") as f:
        json.dump(play_data, f, indent=2)


def run_detection(path: str):
    """Main analysis pipeline that exports JSON and draws results."""
    img_cleared = image_clearer(path)
    homePositions, awayPositions, circlePositions, circleRadius = circle_detector(path, img_cleared)

    if img_cleared is None or not len(circlePositions):
        print("[Error] Could not detect circles or load image properly.")
        return None, np.zeros((800, 800, 3), dtype=np.uint8)

    img = get_image_just_lines(path, circlePositions, circleRadius)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours = filter_contours(gray)

    output = np.zeros_like(img)
    endpoints = []

    for cnt in contours:
        pts = cnt.reshape(-1, 2)
        max_dist = 0
        start_point = end_point = pts[0]
        for p1 in pts[::max(1, len(pts) // 50)]:
            for p2 in pts[::max(1, len(pts) // 50)]:
                dist = np.linalg.norm(p1 - p2)
                if dist > max_dist:
                    max_dist = dist
                    start_point, end_point = p1, p2
        endpoints.append((tuple(start_point.astype(int)), tuple(end_point.astype(int))))

    # --- Draw everything safely ---
    drawn = img.copy()

    for cnt in contours:
        cv2.drawContours(drawn, [cnt], -1, (255, 0, 255), 2)

    for (start, end) in endpoints:
        cv2.circle(drawn, start, 5, (0, 255, 0), -1)
        cv2.circle(drawn, end, 5, (255, 0, 0), -1)
        cv2.line(drawn, start, end, (0, 255, 255), 1)

    for (cx, cy) in circlePositions:
        cv2.circle(drawn, (int(cx), int(cy)), int(circleRadius), (255, 255, 255), 2)

    export_filename = os.path.splitext(os.path.basename(path))[0] + "_testing.json"

    export_to_play_json_from_contours(
        homePositions,
        awayPositions,
        circleRadius,
        endpoints,
        contours,
        drawn.shape,
        play_name="Offensive Play (testing)",
        filename=export_filename,
    )

    with open(export_filename, "r") as f:
        play_json = json.load(f)

    return play_json, drawn



# --- Local Testing ---
if __name__ == "__main__":
    index = 0
    path = f"C:\\Users\\louay\\Desktop\\Python\\images\\play{index}.png"

    play_json, drawn = run_detection(path)

    cv2.imshow("Testing Visualization", drawn)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
