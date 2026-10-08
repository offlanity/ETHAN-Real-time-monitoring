# Real-time Face Recognition & Room Monitor Documentation

This project is a classroom-facing surveillance and monitoring system built around a live webcam feed. It uses computer vision to:

- recognize registered students by face,
- track their movements over time,
- detect suspicious or unusual motion states,
- flag possible visual cues such as dark clothing or face coverings,
- display live data in a local web dashboard.

The system is designed to run locally, with a Python backend and a lightweight browser dashboard.

## Project structure

- [app.py](./app.py) — main application entry point and runtime loop
- [recognizer.py](./recognizer.py) — InsightFace-based identity recognition
- [kinematics.py](./kinematics.py) — movement tracking, speed, displacement, and motion alerts
- [appearance_cues.py](./appearance_cues.py) — heuristic visual cue detection
- [dashboard.py](./dashboard.py) — HTTP dashboard server and state aggregation
- [dashboard.html](./dashboard.html) — browser frontend for monitoring and roster editing
- [utils.py](./utils.py) — drawing utilities and HUD overlay rendering
- [calibration_config.py](./calibration_config.py) — calibration persistence
- [calibration.py](./calibration.py) — manual camera calibration helper
- [calibration.json](./calibration.json) — saved pixels-per-meter scale
- [requirements.txt](./requirements.txt) — Python dependency list
- [dataset/](./dataset) — registered identity images used for face matching
- [tests/](./tests) — unit tests for appearance cues, kinematics, and calibration

## High-level architecture

The system is split into 4 main layers:

1. Inference layer
   - `FaceRecognizer` identifies faces using InsightFace.
2. Motion layer
   - `KinematicsTracker` converts bounding boxes into tracked identities and physics metrics.
3. Visualization layer
   - `utils.py` draws boxes, speed text, labels, and prompts directly onto the camera frame.
4. Monitoring layer
   - `DashboardServer` + `dashboard.html` expose a live room-monitor dashboard for the user.

The program starts in [app.py](./app.py), creates the recognizer, starts the tracker, opens the camera, launches the dashboard server, and then continuously processes frames in a real-time loop.

## Runtime flow

### 1) Startup
In [app.py](./app.py):

- it ensures the dataset folder exists,
- creates a `FaceRecognizer`,
- creates a `KinematicsTracker`,
- opens the camera with OpenCV,
- starts the local dashboard server,
- opens the dashboard in the default browser.

The main runtime loop then runs until the user quits.

### 2) Frame processing
On each frame:

- OpenCV reads the next camera frame.
- `FaceRecognizer.process_frame(frame)` runs InsightFace face detection and embedding comparison.
- each face is assigned either:
  - a known identity name,
  - or `Unknown` if it does not meet the similarity threshold.
- the recognized faces are passed to `KinematicsTracker.update(...)`.
- the motion tracker associates detections across frames, computes speed and displacement, and emits a status like `Stationary`, `Walking`, or `RUNNING (ALERT!)`.
- final annotations are drawn onto the live OpenCV window by helper functions from [utils.py](./utils.py).

### 3) Unknown face registration flow
If an unrecognized face is found, `app.py` stores a snapshot and bounding box for that face. The user can press Enter to register the person:

- the app enters a name-entry mode,
- the user types a name,
- the face crop is extracted from the frame,
- the embedding is added to the registered identity list,
- a saved image is written to the dataset folder.

This flows through `FaceRecognizer.add_person(...)`.

### 4) Dashboard updates
Every frame, the backend publishes:

- live detections,
- recognized names,
- active alerts,
- FPS,
- frame image,
- roster config.

This is managed by `DashboardState` in [dashboard.py](./dashboard.py).

## File-by-file breakdown

### [app.py](./app.py)
This is the main orchestration script.

Key responsibilities:

- sets global app configuration (`DATASET_PATH`, `SIMILARITY_THRESHOLD`, camera source, dashboard host/port),
- initializes the recognizer and tracker,
- opens the camera stream,
- starts the dashboard web server,
- serves frames to the OpenCV window,
- listens for keyboard input.

Important behavior:

- `CAMERA_SOURCE` can be `0` for webcam or a DroidCam URL.
- `SKIP_FRAMES` avoids running expensive face recognition every frame to preserve consistent FPS.
- `typing_mode` allows a user to add new identities while the app is running.

The loop is the backbone of the app:

```python
while True:
    ok, frame = cap.read()
    if not ok:
        break

    if frame_idx % SKIP_FRAMES == 0:
        cached_detections = recognizer.process_frame(frame)

    kinematics_results = tracker.update(cached_detections)
    ... draw overlays ...
    dashboard_state.update(...)
    cv2.imshow(...)
```

This keeps the visual overlay, recognition pipeline, motion tracker, and dashboard all synchronized.

### [recognizer.py](./recognizer.py)
This is the identity recognition engine.

Purpose:

- load labeled images from the dataset,
- detect faces inside each image,
- embed them with InsightFace,
- compare live face embeddings against stored embeddings,
- determine if a face is known or unknown.

Core logic:

- `FaceAnalysis(name="buffalo_l")` builds the face model.
- `_load_dataset()` scans the dataset directory and loads each person image.
- each face image is converted into a 512-dimensional embedding vector.
- embeddings are L2-normalized, then stacked into a matrix.
- `_best_match()` computes cosine similarity using the dot product of normalized vectors.

The match threshold is set by `threshold` (default 0.45), meaning a face is considered known only when the similarity is high enough.

The class stores:

- `self._names` — list of registered names
- `self._embeddings` — matrix of registered embeddings

`add_person(...)` allows runtime registration by extracting a crop from the live frame and saving it as a new file in the dataset folder.

### [kinematics.py](./kinematics.py)
This module provides motion tracking and behavioral analytics.

It creates persistent “tracks” for each person across frames, using Euclidean distance between the face anchor point and the previous position.

Core concepts:

- anchor point = bottom-center of the face bounding box
- `history` = deque of recent position samples
- `track_id` = persistent identifier for a person across frames
- `total_distance` = cumulative path distance traveled
- `displacement` = net change from starting position
- `speed` = instantaneous speed over a short time window

Tracking logic:

- create a registered track when a new detection appears
- match current detections with previous tracks based on nearby positions
- update each track’s position and distance accumulation
- mark tracks as disappeared when no match is found for several frames, then deregister them

Status logic:

```python
if speed >= RUNNING_SPEED_THRESHOLD:
    status = "RUNNING (ALERT!)"
elif speed >= 0.35:
    status = "Walking"
else:
    status = "Stationary"
```

The calibration scale is loaded from [calibration_config.py](./calibration_config.py), which reads `calibration.json` to convert pixels to meters.

### [appearance_cues.py](./appearance_cues.py)
This module is intentionally heuristic and lightweight.

Its goal is not to make a definitive judgment. Instead, it looks for rough visual indicators around a detected face:

- `possible_dark_clothing`
- `possible_face_covering`

The cues are computed from image statistics:

- dark clothing is approximated by low brightness in pixels below the face region,
- possible face covering is estimated by a sharp color/texture contrast between the middle and lower face regions.

This is meant to raise a possible alert for human review, not to make a final classification.

### [utils.py](./utils.py)
This file handles drawing everything onto the live OpenCV frame.

Provided helpers:

- `FPSCounter` — tracks rolling FPS
- `draw_kinematics_face(...)` — renders bounding box, motion trail, HUD, and cues
- `draw_fps(...)` — shows FPS in the corner
- `draw_info_bar(...)` — text overlay with registration count and controls
- `draw_add_prompt(...)` — shows prompt when an unknown face is pending
- `draw_name_input(...)` — draws the text-entry overlay for new identities

This is where the user sees the live system state directly in the OpenCV window.

### [dashboard.py](./dashboard.py)
This file provides the local web dashboard.

It contains:

- `DashboardState` — thread-safe snapshot holder for backend state
- `DashboardServer` — HTTP server exposing the web UI and API

Key endpoints:

- `/` — serves [dashboard.html](./dashboard.html)
- `/api/state` — returns JSON snapshot of current detections, people, events, and frame readiness
- `/stream.mjpg` — streams live video as multipart JPEG frames
- `/api/config` — updates room/section assignment configuration
- `/api/events/clear` — clears recent alert history

`DashboardState` also manages:

- active alert set,
- event log,
- roster membership mapping,
- saved classroom configuration in `dataset/classroom.json`.

When an unknown face appears or a student is assigned to the wrong section, it creates alert entries that are displayed in the web panel with timestamps and messages.

### [dashboard.html](./dashboard.html)
This is the browser UI for the dashboard.

It is a single-page app rendered by the local HTTP server. It includes:

- top bar with live status and clock
- camera feed panel
- active alerts metric
- roster count metric
- people table with per-person section assignments
- recent alerts/event log
- config save actions

The script uses JavaScript to:

- fetch `/api/state` every ~900 ms,
- render current people and detections,
- display attendance or alert info,
- allow saving any roster configuration or room section,
- stream the live camera image from `/stream.mjpg`.

The UI is intentionally local-only and simple; it acts as a monitoring dashboard rather than a heavy web application.

### [calibration_config.py](./calibration_config.py)
This module handles the pixel-to-meter scale used by motion tracking.

Responsibilities:

- default calibration = `200.0` pixels per meter
- `load_pixels_per_meter()` reads `calibration.json`
- `save_pixels_per_meter()` writes a new calibration value
- `validate_pixels_per_meter()` ensures the value is a positive finite number

This value is vital because speed and distance estimates in `kinematics.py` depend on it.

### [calibration.py](./calibration.py)
This is a manual calibration tool for a physical camera.

Procedure:

1. point the camera at a known 1-meter reference,
2. click the start and end points in the camera image,
3. press `s` to save the scale,
4. restart the app to load the new calibration.

The script uses OpenCV mouse callbacks and computes:

```python
pixel_distance = math.hypot(x2 - x1, y2 - y1)
pixels_per_meter = pixel_distance / 1.0
```

This gives the average pixel length per meter for your specific camera setup.

### [requirements.txt](./requirements.txt)
Lists the project dependencies, including:

- OpenCV (`opencv-python`)
- InsightFace (`insightface`)
- NumPy
- other Python runtime pieces needed for the model and detection stack

This project is strongly dependent on the model ecosystem around face detection and embeddings.

### [dataset/](./dataset)
This directory holds the known-person images used by the recognition pipeline.

The expected naming convention is:

```text
FirstName_LastName.jpg
```

At runtime, a file like `Alice_Smith.jpg` becomes the display label `Alice Smith`.

The app also stores new user registrations here as cropped face images with names sanitized into safe file names.

### [tests/](./tests)
The tests verify the important heuristics and calibration rules.

- [tests/test_appearance_cues.py](./tests/test_appearance_cues.py)
  - validates dark-clothing logic,
  - validates face-covering heuristics,
  - checks DashboardState alert behavior.

- [tests/test_kinematics.py](./tests/test_kinematics.py)
  - validates calibration file behavior,
  - checks displacement vs distance calculations,
  - checks tracker behavior with saved scale.

These tests confirm the system's rules are stable even when the surrounding camera setup changes.

## Data flow and system interaction

The project is best understood as a stateful vision loop.

### Step-by-step flow

1. camera frame enters the app
2. face detector runs on the frame
3. each face gets an embedding and similarity score
4. tracker attaches identity and motion history to each face
5. heuristics add appearance cues
6. dashboard state combines detections with room section configuration
7. UI overlays and browser dashboard reflect the newest state
8. user can register new faces or adjust section assignments

### What data structures matter

A detection result typically contains fields such as:

- `bbox`
- `name`
- `score`
- `recognized`
- `appearance_cues`
- `track_id`
- `speed`
- `displacement`
- `total_distance`
- `status`
- `history`

This is the main payload passed from recognition to kinematics to the dashboard.

## How the recognition works mathematically

The recognizer uses InsightFace embeddings.

For each face, the system computes an embedding vector. The model turns a face image into a feature vector. Then it compares the live embedding against the dataset embeddings using cosine similarity.

This is effectively:

```python
query = normalize(embedding)
score = dataset_embeddings @ query
```

Since both vectors are L2-normalized, the dot product gives cosine similarity. If the score exceeds the threshold, the face is considered recognized.

## How motion and alerts work

The tracker follows face centers across frames. It measures:

- how far a person moved between recent frames,
- how much total path they covered,
- how much the net position changed from their start point,
- whether they exceed a running threshold.

If motion is fast enough, the person is marked as `RUNNING (ALERT!)`.

The system also merges signal from appearance heuristics and roster logic to raise alerts like:

- unregistered person
- wrong section
- possible dark clothing
- possible face covering

## Why this architecture works well

This project is intentionally modular:

- recognition and tracking are separate concerns,
- calibration is stored independently from runtime logic,
- dashboard data and frame generation are separated from OpenCV rendering,
- tests focus on the most important computational logic,
- the app stays simple enough for local classroom monitoring.

This makes the code easy to reason about while still handling the main real-time tasks:

- face recognition,
- attendee tracking,
- motion analytics,
- alert generation,
- visual monitoring.

## Practical usage notes

- For a laptop webcam, set `CAMERA_SOURCE = "0"`.
- For an Android phone camera via DroidCam, replace it with the DroidCam URL.
- Use `calibration.py` to calibrate the motion scale before relying on distance/speed output.
- The system expects a good dataset of clear face images for consistent recognition.
- Appearance indicators are heuristic and should be treated as human-review signals, not definitive evidence.

## Summary

This project combines computer vision, a custom motion tracker, and a real-time monitoring dashboard to create a classroom room-monitoring tool.

The main idea is:

- identify who is in frame,
- track where each person moves,
- classify motion and alert states,
- surface all of this in a browser dashboard for live monitoring.

At the center of the application is [app.py](./app.py), which coordinates the recognition pipeline, tracker, UI drawing, and web dashboard into one continuous real-time loop.
