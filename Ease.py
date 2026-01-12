import time
import numpy as np
import cv2
import cvzone
from ultralytics import YOLO
from sort import Sort

# ---------------- CONFIG ----------------
YOLO_WEIGHTS = "../Yolo-Weights/yolov8m.pt"  # user requested
FRAME_W, FRAME_H = 640, 360               # per-quadrant resolution
CONF_THRESH = 0.35
VEHICLE_CLASSES = {"car", "motorbike", "bus", "truck"}

# Video files (Option A)
caps = {
    "NORTH": cv2.VideoCapture("n1.mp4"),
    "SOUTH": cv2.VideoCapture("s.mp4"),
    "EAST":  cv2.VideoCapture("e1.mp4"),
    "WEST":  cv2.VideoCapture("w1.mp4")
}

# YOLO model
model = YOLO(YOLO_WEIGHTS)

# Per-camera SORT tracker (avoid ID collision across feeds)
trackers = {d: Sort(max_age=15, min_hits=2, iou_threshold=0.3) for d in caps.keys()}

# Per-camera prev center dict and counted IDs used to detect crossings
prev_centers = {d: {} for d in caps.keys()}
counted_ids = {d: set() for d in caps.keys()}

# Counters:
# - analysis_counts: used after initial 10s analysis to form Cycle 1 order
# - cycle_counts: counts gathered during current serving cycle (used for next cycle)
analysis_counts = {d: 0 for d in caps.keys()}
cycle_counts = {d: 0 for d in caps.keys()}

# Enum-like modes
MODE_ANALYSE = "ANALYSING"
MODE_RUNNING = "RUNNING"

mode = MODE_ANALYSE
analysis_duration = 10.0  # seconds for initial analysis
analysis_start_time = time.time()

# Signal control
current_green = None
green_phase_end = None
# fixed green times mapping by rank (A: fixed)
FIXED_TIMES_BY_RANK = [15, 12, 10, 8]  # index 0 -> highest, 1 -> second, ...

# Priority list used in current cycle (order of directions to serve)
priority_order = []
priority_index = 0

# Utility functions
def safe_read(cap):
    """Read and loop a video if it ends; resize to FRAME_W x FRAME_H."""
    success, frame = cap.read()
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, frame = cap.read()
        if not success:
            # black placeholder
            return np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    return cv2.resize(frame, (FRAME_W, FRAME_H))

def detect_vehicles(frame):
    """Run YOLO and return detections [[x1,y1,x2,y2,conf], ...]"""
    dets = np.empty((0, 5))
    results = model(frame, stream=True, verbose=False)
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            cls_name = model.names[cls]
            if cls_name in VEHICLE_CLASSES and conf >= CONF_THRESH:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                dets = np.vstack((dets, [x1, y1, x2, y2, conf]))
    return dets

def process_frame_for_direction(direction, cap, is_green, counting_set):
    """
    Process single camera: detection -> tracking -> crossing detection.
    counting_set: the set used to mark counted IDs for the current counting phase (analysis or cycle)
    Returns: waiting_count, annotated_frame, newly_counted (int # newly counted this call)
    Side effects:
      - updates trackers[direction] internal state
      - updates prev_centers[direction]
      - adds IDs to counting_set when crossing is detected
    """
    frame = safe_read(cap)
    h, w = frame.shape[:2]
    line_y = h // 2
    line_thickness = 6

    # detections
    dets = detect_vehicles(frame)

    # tracker update (per camera)
    trks = trackers[direction].update(dets)

    newly_counted = 0
    # default red line
    cv2.line(frame, (0, line_y), (w, line_y), (0, 0, 255), line_thickness)

    for trk in trks:
        x1, y1, x2, y2, tid = trk
        x1, y1, x2, y2, tid = int(x1), int(y1), int(x2), int(y2), int(tid)
        bw, bh = x2 - x1, y2 - y1
        cx, cy = x1 + bw // 2, y1 + bh // 2

        # draw box/id
        cvzone.cornerRect(frame, (x1, y1, bw, bh), l=7, rt=2, colorR=(0,180,255))
        cvzone.putTextRect(frame, f"ID:{tid}", (max(0, x1), max(30, y1)), scale=0.8, thickness=2, offset=3)
        cv2.circle(frame, (cx, cy), 4, (200, 0, 200), cv2.FILLED)

        prev = prev_centers[direction].get(tid, None)
        if prev is not None:
            _, prev_cy = prev
            # top->down crossing
            if prev_cy < line_y <= cy and tid not in counting_set:
                counting_set.add(tid)
                newly_counted += 1
            # bottom->top crossing
            elif prev_cy > line_y >= cy and tid not in counting_set:
                counting_set.add(tid)
                newly_counted += 1

        prev_centers[direction][tid] = (cx, cy)

    if newly_counted:
        cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 0), line_thickness)

    # label direction + small stats for debug
    sig_color = (0,255,0) if is_green else (0,0,255)
    cv2.putText(frame, direction, (12,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, sig_color, 2)
    waiting_count = len(trks)
    cv2.putText(frame, f"Waiting:{waiting_count}", (12, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    return waiting_count, frame, newly_counted

def make_tile_ui(tile, direction, count, is_green):
    """
    Draws polished, perfectly aligned UI on a tile (FRAME_W x FRAME_H).
    - Direction label top-left
    - Big signal circle top-right
    - Vehicles count centered in bottom box
    """
    h, w = tile.shape[:2]
    # translucent header
    header_h = 70
    overlay = tile.copy()
    cv2.rectangle(overlay, (0,0), (w, header_h), (30,30,30), -1)
    cv2.addWeighted(overlay, 0.6, tile, 0.4, 0, tile)

    # Title left
    cv2.putText(tile, direction, (18,44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 3)

    # Big signal circle
    circle_x = w - 48
    circle_y = 40
    r = 28
    color = (0,255,0) if is_green else (0,0,255)
    cv2.circle(tile, (circle_x, circle_y), r, color, -1)
    cv2.circle(tile, (circle_x, circle_y), r+2, (255,255,255), 2)

    # Bottom count box
    box_h = 80
    box_top = h - box_h - 12
    box_left, box_right = 12, w - 12
    cv2.rectangle(tile, (box_left, box_top), (box_right, h - 12), (0,0,0), -1)
    cv2.rectangle(tile, (box_left, box_top), (box_right, h - 12), (80,80,80), 2)

    text = f"Vehicles: {count}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
    tx = box_left + (box_right - box_left - tw) // 2
    ty = box_top + (box_h + th) // 2 - 6
    cv2.putText(tile, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,215,0), 3)

    return tile

# ---------------- MAIN CONTROL LOOP ----------------
print("[INFO] Starting system. Initial analysis will run for {} seconds.".format(analysis_duration))
analysis_end_time = analysis_start_time + analysis_duration

# We'll use these temporary sets to collect counted IDs separately per phase:
# - analysis_counted_ids (cleared before analysis)
# - cycle_counted_ids (cleared at start of each cycle)
analysis_counted_ids = {d: set() for d in caps.keys()}
cycle_counted_ids = {d: set() for d in caps.keys()}

# We'll also keep a variable to track current phase cycles
in_cycle = False  # True when serving a cycle (after analysis)

# Variables for building per-cycle priority from previous cycle
last_cycle_counts = {d: 0 for d in caps.keys()}  # used to determine next priority after first cycle

# Initialize: mode = ANALYSING
mode = MODE_ANALYSE
print("[INFO] ANALYSIS MODE started for {} seconds...".format(analysis_duration))

while True:
    # read/process each camera
    waiting = {}
    frames = {}
    newly_counts_this_frame = {d: 0 for d in caps.keys()}

    for d, cap in caps.items():
        if mode == MODE_ANALYSE:
            # use analysis_counted_ids set for counting during analysis
            wait_cnt, frame, newly = process_frame_for_direction(d, cap, False, analysis_counted_ids[d])
            analysis_counts[d] = len(analysis_counted_ids[d])
        else:
            # RUNNING mode -> we may be in a green phase for some direction
            is_green = (current_green == d)
            wait_cnt, frame, newly = process_frame_for_direction(d, cap, is_green, cycle_counted_ids[d])
            # accumulate for the cycle_counts (used for the next cycle)
            cycle_counts[d] = len(cycle_counted_ids[d])

        waiting[d] = wait_cnt
        frames[d] = frame
        newly_counts_this_frame[d] = newly

    # Build UI tiles
    north_tile = make_tile_ui(frames["NORTH"].copy(), "NORTH", (analysis_counts["NORTH"] if mode==MODE_ANALYSE else cycle_counts["NORTH"]), current_green == "NORTH")
    south_tile = make_tile_ui(frames["SOUTH"].copy(), "SOUTH", (analysis_counts["SOUTH"] if mode==MODE_ANALYSE else cycle_counts["SOUTH"]), current_green == "SOUTH")
    east_tile  = make_tile_ui(frames["EAST"].copy(),  "EAST",  (analysis_counts["EAST"]  if mode==MODE_ANALYSE else cycle_counts["EAST"]),  current_green == "EAST")
    west_tile  = make_tile_ui(frames["WEST"].copy(),  "WEST",  (analysis_counts["WEST"]  if mode==MODE_ANALYSE else cycle_counts["WEST"]),  current_green == "WEST")

    top_row = cv2.hconcat([north_tile, south_tile])
    bottom_row = cv2.hconcat([east_tile, west_tile])
    combined = cv2.vconcat([top_row, bottom_row])

    # Top info bar
    Hc, Wc = combined.shape[:2]
    overlay = combined.copy()
    cv2.rectangle(overlay, (0,0), (Wc, 86), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.62, combined, 0.38, 0, combined)

    # Display mode and timers
    if mode == MODE_ANALYSE:
        now = time.time()
        remaining = max(0, int(analysis_end_time - now))
        cv2.putText(combined, f"MODE: ANALYSIS (collecting for {analysis_duration}s)  -  {remaining}s left", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        # show current analysis counts
        off = 360
        cv2.putText(combined, f"Counts: N:{analysis_counts['NORTH']}  S:{analysis_counts['SOUTH']}  E:{analysis_counts['EAST']}  W:{analysis_counts['WEST']}", (off, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)

        # Switch to RUNNING when analysis period ends
        if time.time() >= analysis_end_time:
            # build initial priority list from analysis_counts (highest first)
            priority_order = sorted(analysis_counts.keys(), key=lambda x: analysis_counts[x], reverse=True)
            priority_index = 0
            # set up cycle counts to zero and clear cycle_counted_ids
            for d in cycle_counts:
                cycle_counts[d] = 0
                cycle_counted_ids[d].clear()
            # set first green direction
            if priority_order:
                current_green = priority_order[priority_index]
                # green duration based on rank index using FIXED_TIMES_BY_RANK
                green_dur = FIXED_TIMES_BY_RANK[min(priority_index, len(FIXED_TIMES_BY_RANK)-1)]
                green_phase_end = time.time() + green_dur
                print("[INFO] ENTERING RUNNING MODE. Priority (from analysis):", priority_order)
                print("[INFO] First green:", current_green, "for", green_dur, "s")
                mode = MODE_RUNNING
                in_cycle = True
    else:
        # RUNNING mode
        remaining = int(max(0, green_phase_end - time.time())) if green_phase_end else 0
        cv2.putText(combined, f"MODE: RUNNING  -  Serving: {current_green}  ({remaining}s left)", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        # show cycle_counts (counts collected during this cycle)
        cv2.putText(combined, f"Cycle Counts: N:{cycle_counts['NORTH']}  S:{cycle_counts['SOUTH']}  E:{cycle_counts['EAST']}  W:{cycle_counts['WEST']}",
                    (18, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

        # Highlight current green tile indicator circle by re-drawing tile small overlay
        # (already colored via make_tile_ui using current_green)

        # Check if green phase finished
        if green_phase_end and time.time() >= green_phase_end:
            # move to next priority
            priority_index += 1
            if priority_index < len(priority_order):
                current_green = priority_order[priority_index]
                green_dur = FIXED_TIMES_BY_RANK[min(priority_index, len(FIXED_TIMES_BY_RANK)-1)]
                green_phase_end = time.time() + green_dur
                print("[INFO] Next green:", current_green, "for", green_dur, "s")
            else:
                # finished serving all 4 directions -> end of this cycle
                print("[INFO] Completed a full cycle. Cycle counts:", cycle_counts)
                # Prepare priority for next cycle based on cycle_counts (use these counts)
                priority_order = sorted(cycle_counts.keys(), key=lambda x: cycle_counts[x], reverse=True)
                priority_index = 0
                # reset counted id sets and prev_centers for next cycle counting;
                # but we keep trackers running (so IDs may persist across cycles)
                for d in cycle_counted_ids:
                    cycle_counted_ids[d].clear()
                    prev_centers[d].clear()
                    # also reset counted_ids global set so tracking can count again next cycle
                    counted_ids[d].clear()
                # Set current_green to next according to priority_order
                if priority_order:
                    current_green = priority_order[priority_index]
                    green_dur = FIXED_TIMES_BY_RANK[min(priority_index, len(FIXED_TIMES_BY_RANK)-1)]
                    green_phase_end = time.time() + green_dur
                    print("[INFO] Starting next cycle. New priority order (from last cycle):", priority_order)
                    print("[INFO] Next green:", current_green, "for", green_dur, "s")
                else:
                    # fallback
                    priority_order = list(caps.keys())
                    priority_index = 0
                    current_green = priority_order[priority_index]
                    green_phase_end = time.time() + FIXED_TIMES_BY_RANK[0]

    # Show combined window
    cv2.imshow("EaseTraffic - 4grid Fixed-cycle", combined)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('r'):
        # Reset everything and restart analysis
        print("[INFO] Reset requested by user. Restarting analysis.")
        # clear counters and counting sets
        for d in caps.keys():
            analysis_counts[d] = 0
            cycle_counts[d] = 0
            analysis_counted_ids[d].clear()
            cycle_counted_ids[d].clear()
            prev_centers[d].clear()
            counted_ids[d].clear()
        # restart analysis timer
        analysis_start_time = time.time()
        analysis_end_time = analysis_start_time + analysis_duration
        mode = MODE_ANALYSE
        current_green = None
        green_phase_end = None
        priority_order = []
        priority_index = 0

# cleanup
for c in caps.values():
    c.release()
cv2.destroyAllWindows()
