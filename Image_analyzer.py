import cv2
import numpy as np
from CircleDetector import circle_detector
from ImageClearer import image_clearer
import json

MIN_RUN_LENGTH = 100  # adjust depending on scale (pixels)

def run_length(run_lines, w, h):
    """Compute total length of run lines in pixel space."""
    total = 0.0
    for (x1n, y1n), (x2n, y2n) in run_lines:
        x1, y1 = x1n * w, y1n * h
        x2, y2 = x2n * w, y2n * h
        total += np.hypot(x2 - x1, y2 - y1)
    return total

class Line:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        self.angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

def are_lines_similar(line1, line2, endpoint_thresh=10, angle_thresh=5, length_thresh=10):
    # Compare endpoints (both orders)
    endpoints1 = [(line1.x1, line1.y1), (line1.x2, line1.y2)]
    endpoints2 = [(line2.x1, line2.y1), (line2.x2, line2.y2)]
    dist1 = np.linalg.norm(np.array(endpoints1[0]) - np.array(endpoints2[0])) + np.linalg.norm(np.array(endpoints1[1]) - np.array(endpoints2[1]))
    dist2 = np.linalg.norm(np.array(endpoints1[0]) - np.array(endpoints2[1])) + np.linalg.norm(np.array(endpoints1[1]) - np.array(endpoints2[0]))
    endpoints_close = dist1 < endpoint_thresh*2 or dist2 < endpoint_thresh*2

    # Compare angles (allow for opposite direction)
    angle1 = np.degrees(np.arctan2(line1.y2 - line1.y1, line1.x2 - line1.x1))
    angle2 = np.degrees(np.arctan2(line2.y2 - line2.y1, line2.x2 - line2.x1))
    angle_diff = min(abs(angle1 - angle2), abs(abs(angle1 - angle2) - 180))
    length1 = np.sqrt((line1.x2 - line1.x1)**2 + (line1.y2 - line1.y1)**2)
    length2 = np.sqrt((line2.x2 - line2.x1)**2 + (line2.y2 - line2.y1)**2)
    length_close = abs(length1 - length2) < length_thresh

    return (endpoints_close or (angle_diff < angle_thresh and length_close))

def remove_duplicate_lines(lines):
    unique = []
    for line in lines:
        if not any(are_lines_similar(line, uline) for uline in unique):
            unique.append(line)
    return unique

def is_line_inside(long_line, short_line, angle_thresh=5, endpoint_thresh=10):
    # Calculate angles
    angle_long = np.degrees(np.arctan2(long_line.y2 - long_line.y1, long_line.x2 - long_line.x1))
    angle_short = np.degrees(np.arctan2(short_line.y2 - short_line.y1, short_line.x2 - short_line.x1))
    if abs(angle_long - angle_short) > angle_thresh:
        return False

    # Check if both endpoints of short_line are close to long_line
    def point_to_line_dist(px, py, x1, y1, x2, y2):
        # Distance from point (px, py) to line segment (x1, y1)-(x2, y2)
        line_mag = np.hypot(x2 - x1, y2 - y1)
        if line_mag < 1e-6:
            return np.hypot(px - x1, py - y1)
        u = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_mag ** 2)
        u = max(min(u, 1), 0)
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        return np.hypot(px - ix, py - iy)

    d1 = point_to_line_dist(short_line.x1, short_line.y1, long_line.x1, long_line.y1, long_line.x2, long_line.y2)
    d2 = point_to_line_dist(short_line.x2, short_line.y2, long_line.x1, long_line.y1, long_line.x2, long_line.y2)
    return d1 < endpoint_thresh and d2 < endpoint_thresh

def remove_inner_lines(lines, angle_thresh=5, endpoint_thresh=10):
    keep = []
    for i, line in enumerate(lines):
        is_inner = False
        for j, other in enumerate(lines):
            if i == j:
                continue
            len_line = np.hypot(line.x2 - line.x1, line.y2 - line.y1)
            len_other = np.hypot(other.x2 - other.x1, other.y2 - other.y1)
            if len_other > len_line and is_line_inside(other, line, angle_thresh, endpoint_thresh):
                is_inner = True
                break
        if not is_inner:
            keep.append(line)
    return keep

def merge_parallel_lines(lines, angle_thresh=5, midpoint_thresh=15):
    merged = []
    used = set()
    for i, line1 in enumerate(lines):
        if i in used:
            continue
        angle1 = np.degrees(np.arctan2(line1.y2 - line1.y1, line1.x2 - line1.x1))
        mid1 = ((line1.x1 + line1.x2) / 2, (line1.y1 + line1.y2) / 2)
        found = False
        for j, line2 in enumerate(lines):
            if i == j or j in used:
                continue
            angle2 = np.degrees(np.arctan2(line2.y2 - line2.y1, line2.x2 - line2.x1))
            mid2 = ((line2.x1 + line2.x2) / 2, (line2.y1 + line2.y2) / 2)
            if abs(angle1 - angle2) < angle_thresh and np.linalg.norm(np.array(mid1) - np.array(mid2)) < midpoint_thresh:
                # Keep the longer line
                len1 = np.hypot(line1.x2 - line1.x1, line1.y2 - line1.y1)
                len2 = np.hypot(line2.x2 - line2.x1, line2.y2 - line2.y1)
                merged.append(line1 if len1 >= len2 else line2)
                used.add(i)
                used.add(j)
                found = True
                break
        if not found:
            merged.append(line1)
            used.add(i)
    return merged

def point_to_line_dist(px, py, x1, y1, x2, y2):
    # Distance from point (px, py) to line segment (x1, y1)-(x2, y2)
    line_mag = np.hypot(x2 - x1, y2 - y1)
    if line_mag < 1e-6:
        return np.hypot(px - x1, py - y1)
    u = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_mag ** 2)
    u = max(min(u, 1), 0)
    ix = x1 + u * (x2 - x1)
    iy = y1 + u * (y2 - y1)
    return np.hypot(px - ix, py - iy)

def lines_connect(line1, line2, endpoint_thresh=15):
    # Check endpoints
    endpoints1 = [(line1.x1, line1.y1), (line1.x2, line1.y2)]
    endpoints2 = [(line2.x1, line2.y1), (line2.x2, line2.y2)]
    for p in endpoints2:
        # Check if p is close to either endpoint of line1
        if (np.hypot(p[0] - line1.x1, p[1] - line1.y1) < endpoint_thresh or
            np.hypot(p[0] - line1.x2, p[1] - line1.y2) < endpoint_thresh):
            return True
        # Check if p is close to line1 segment
        if point_to_line_dist(p[0], p[1], line1.x1, line1.y1, line1.x2, line1.y2) < endpoint_thresh:
            return True
    return False

def group_connected_lines_no_branch(lines, endpoint_thresh=20):
    unused = set(range(len(lines)))
    runs = []

    while unused:
        idx = unused.pop()
        run = [lines[idx]]
        # Endpoints available for matching (initially both)
        available_endpoints = { (lines[idx].x1, lines[idx].y1), (lines[idx].x2, lines[idx].y2) }
        added = True
        while added and available_endpoints:
            added = False
            last = run[-1]
            for i in list(unused):
                l = lines[i]
                endpoints = [ (l.x1, l.y1), (l.x2, l.y2) ]
                matched = None
                for ep in endpoints:
                    for av_ep in available_endpoints:
                        if np.hypot(ep[0] - av_ep[0], ep[1] - av_ep[1]) < endpoint_thresh or \
                           point_to_line_dist(ep[0], ep[1], last.x1, last.y1, last.x2, last.y2) < endpoint_thresh:
                            matched = (ep, av_ep)
                            break
                    if matched:
                        break
                if matched:
                    # After a match is found:
                    run.append(l)
                    unused.remove(i)

                    # Consume both matched endpoints
                    available_endpoints.discard(matched[1])   # consume old run endpoint
                    available_endpoints.discard(matched[0])   # consume new line endpoint

                    # Add the other endpoint of the new line (the one not matched)
                    other_ep = endpoints[1] if matched[0] == endpoints[0] else endpoints[0]
                    if other_ep not in available_endpoints:
                        available_endpoints.add(other_ep)
                    added = True
                    break
        runs.append(run)
    return runs

def closest_line_to_circle(circle, runs):
    cx, cy, r = circle
    closest_dist = float("inf")
    closest_line = None
    run_idx = None

    for i, run in enumerate(runs):
        for line in run:
            d = point_to_line_dist(cx, cy, line.x1, line.y1, line.x2, line.y2)
            if d < closest_dist:
                closest_dist = d
                closest_line = line
                run_idx = i
    return closest_line, run_idx, closest_dist

def export_to_play_json(circle_run_data, homePositions=None, awayPositions=None, play_name="Offensive Play 1", filename="play.json"):
    play_data = {
        "playName": play_name,
        "homePlayerData": [],
        "awayPlayerData": [],
        "footballPositions": [],
        "playTags": []
    }

    def round9(val):
        return round(float(val), 9)

    def make_player_entry(cx, cy, waypoints):
        return {
            "position": {"x": round9(cx), "y": round9(cy), "z": 0.0},
            "run": {
                "runType": 0,
                "waypoints": waypoints,
                "maxSpeed": 5.0,
                "endStyleIndex": 0
            }
        }

    for data in circle_run_data:
        cx, cy, r = data["circle"]
        waypoints = []
        if data["run_lines"]:
            waypoints.append({"x": round9(cx), "y": round9(cy), "z": 0.0})
            for (x1, y1), (x2, y2) in data["run_lines"]:
                waypoints.append({"x": round9(x1), "y": round9(y1), "z": 0.0})
                waypoints.append({"x": round9(x2), "y": round9(y2), "z": 0.0})
            # Remove duplicates
            seen = set()
            unique = []
            for wp in waypoints:
                tup = (wp["x"], wp["y"], wp["z"])
                if tup not in seen:
                    seen.add(tup)
                    unique.append(wp)
            waypoints = unique

        player_entry = make_player_entry(cx, cy, waypoints)

        # Assign to home or away based on detected groups
        if homePositions is not None and any(abs(cx - hp[0]) < 0.02 and abs(cy - hp[1]) < 0.02 for hp in homePositions):
            play_data["homePlayerData"].append(player_entry)
        elif awayPositions is not None and any(abs(cx - ap[0]) < 0.02 and abs(cy - ap[1]) < 0.02 for ap in awayPositions):
            play_data["awayPlayerData"].append(player_entry)
        else:
            play_data["homePlayerData"].append(player_entry)

    with open(filename, "w") as f:
        json.dump(play_data, f, indent=2)

    return play_data


def run_detection(image_path: str):
    # Load image
    img = image_clearer(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert so lines are white on black
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Edge detection
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

    # Hough Line Transform (probabilistic, gives segments)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=30, maxLineGap=10)

    all_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            all_lines.append(Line(x1, y1, x2, y2))


    all_lines = remove_duplicate_lines(all_lines)
    all_lines = remove_inner_lines(all_lines)
    all_lines = merge_parallel_lines(all_lines)

    # Usage after filtering lines:
    runs = group_connected_lines_no_branch(all_lines)
    # Each element in runs is a list of Line objects forming a continuous run
    homePositions, awayPositions, positions, radius = circle_detector(image_path, img)


    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    i = 0
    for run in runs:
        # print(f"--------------Run {i} with {len(run)} lines:---------------")
        for line in run:
            cv2.line(img, (line.x1, line.y1), (line.x2, line.y2), colors[i % len(colors)], 5)
            print(f"Line {i}: Line from ({line.x1}, {line.y1}) to ({line.x2}, {line.y2}), length: {line.length:.1f}, angle: {line.angle:.1f}°")
        i += 1
    # cv2.imshow(f"Detected Lines {i}", img)


    # -------------------------------
    # Build assignments for all circle-line pairs
    assignments = []  # (distance, circle_idx, run_idx)
    RADIUS_MULTIPLIER = 2.0  # try 1.5, 2.0, etc.

    assignments = []  # (distance, circle_idx, run_idx)
    for ci, (cx, cy) in enumerate(positions):
        r = radius
        for ri, run in enumerate(runs):
            min_d = float("inf")
            for line in run:
                d = point_to_line_dist(cx, cy, line.x1, line.y1, line.x2, line.y2)
                if d < min_d:
                    min_d = d
            # Only consider this run if distance <= radius * multiplier
            if min_d <= r * RADIUS_MULTIPLIER:
                assignments.append((min_d, ci, ri))


    # Sort by distance so closest matches are assigned first
    assignments.sort(key=lambda x: x[0])

    circle_to_run = {}
    used_runs = set()
    used_circles = set()

    for d, ci, ri in assignments:
        if ci not in used_circles and ri not in used_runs:
            circle_to_run[ci] = ri
            used_circles.add(ci)
            used_runs.add(ri)


    # -------------------------------

    # Draw results
    for ci, (cx, cy) in enumerate(positions):
        if ci in circle_to_run:
            run_idx = circle_to_run[ci]
            color = colors[run_idx % len(colors)]
            cv2.circle(img, (cx, cy), radius, color, 3)
            cv2.putText(img, f"Run {run_idx}", (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            cv2.circle(img, (cx, cy), radius, (128, 128, 128), 2)
            cv2.putText(img, "No Run", (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

    # cv2.imshow("Detected Circles + Runs", img)
    h, w = img.shape[:2]
    circle_run_data = []

    for ci, (cx, cy) in enumerate(positions):
        circle_info = {
            "circle": (cx / w, 1.0 - (cy / h), radius / w),
            "run_lines": []
        }

        if ci in circle_to_run:
            run_idx = circle_to_run[ci]
            run_lines = runs[run_idx]

            # Build normalized line data
            line_infos = [
                ((line.x1 / w, 1.0 - (line.y1 / h)),
                (line.x2 / w, 1.0 - (line.y2 / h)))
                for line in run_lines
            ]

            # Only keep run if long enough
            if run_length(line_infos, w, h) >= MIN_RUN_LENGTH:
                circle_info["run_lines"].extend(line_infos)

        circle_run_data.append(circle_info)


    #now loop throught circle_run_data and put each circle and its lines on a new clear image, with each circle having the same color as its run lines
    # Blank image
    cleared_image = np.zeros((600, 650, 3), dtype=np.uint8)

    # Colors (reuse your palette)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (0, 255, 255), (255, 0, 255)]

    h, w = cleared_image.shape[:2]

    for ci, data in enumerate(circle_run_data):
        # Assign color based on index
        color = colors[ci % len(colors)]

        # Circle (denormalize back to pixels)
        cx = int(data["circle"][0] * w)
        cy = int(data["circle"][1] * h)
        r  = int(data["circle"][2] * w)  # normalized by width earlier

        cv2.circle(cleared_image, (cx, cy), r, color, 2)

        # Lines (denormalize and draw)
        for (x1n, y1n), (x2n, y2n) in data["run_lines"]:
            x1, y1 = int(x1n * w), int(y1n * h)
            x2, y2 = int(x2n * w), int(y2n * h)
            cv2.line(cleared_image, (x1, y1), (x2, y2), color, 2)

    # Flip the image around the x-axis (vertically)
    cleared_image = cv2.flip(cleared_image, 0)
    
    # cv2.imshow("Cleared Circles + Runs", cleared_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    play_json = export_to_play_json(
        circle_run_data,
        homePositions=homePositions,
        awayPositions=awayPositions,
        play_name="Offensive Play 2",
        filename="offensiveSS_play.json"
    )

    # print(json.dumps(play_json, indent=2))
    return play_json 

index = 0
path = f'images/play{index}.png'
run_detection(path)