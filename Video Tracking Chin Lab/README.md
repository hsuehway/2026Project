# Orthogonal Camera 3D Particle Tracking

## SP Trajectory Studio

Double-click **Launch SP Transition v2.cmd** for direct future-position models,
purged temporal tests, grouped cross-run validation, fixed continuity cohorts and
per-run forecasting skill. The original acceleration model is still available
through the experiment selector. See [trajectory_ml/README.md](trajectory_ml/README.md)
for v2 methods and validation criteria.

Double-click **Launch SP Trajectory Studio.cmd** to open the separate
single-particle trajectory learning website, or run:

```powershell
.\.venv\Scripts\python.exe launch_sp_trajectory_studio.py
```

It provides SP-only data QC, gap-aware derivatives, nested experiment validation,
Ridge/boosting comparison, past-only trajectory forecasts, and saved reports and
model candidates. See [trajectory_ml/README.md](trajectory_ml/README.md) for the
scientific assumptions, configuration and reproducible command-line workflow.

## Video Analysis Studio

The local Video Analysis Studio joins the background-subtraction workflow used
to create final translational and rotational state-vector overlays. It is an
additive controller: every original Python script, command-line argument,
interactive window, and output format remains available exactly as before.

To launch without keeping a terminal window open, double-click:

```text
Launch Video Analysis Studio.cmd
```

On macOS, double-click:

```text
Launch Video Analysis Studio.command
```

The macOS launcher uses `.venv-macos` when it is available and otherwise uses
`python3`. The first launch may require macOS to approve opening the local
launcher. Keep the launcher window open while using the Studio.

The equivalent command is:

```powershell
.\.venv\Scripts\python.exe -m video_analysis_studio
```

The studio opens one local browser tab and provides:

- separate Home, Setup, Track, Review, and Results pages so the workflow remains
  digestible as more analysis controls are added;
- a recommended-next-action banner and a pre-flight checklist that explain what
  is missing before a long tracking run can start;
- local browser import for videos and background media, with imported raw files
  excluded from normal Git history and a SHA-256 provenance manifest saved in
  `outputs_background/import_manifests/`;
- media-metadata inspection so the recording frame rate can be confirmed rather
  than guessed;
- in-browser front/side particle clicks after a calibration has been selected or
  created;
- direct support for extensionless Matroska recordings such as `New Deets/SP6`;
- automatic MI/SP matching for videos, backgrounds, and calibrations;
- background-subtraction preview settings and captured output paths;
- background-subtraction tracking with the existing seed and correction windows;
- automatic raw/smoothed tracking verification over the original camera video,
  plus optional identity review;
- one-click translational finishing: motion, state vectors, and overlays;
- one-click rotational finishing: motion, rotation estimation, combined reduced
  state vectors, and the final full-state overlay;
- a combined finish that calculates motion once and produces both overlay families;
- an integrated trajectory workspace with MP4 playback, projection plots, CSV
  previews, downloads, and descriptive track-health metrics;
- a Results workspace that can import a position-versus-time CSV and optional
  companion video, run the circular-motion detector, and display a synchronized
  interactive report with video, timeline, XY/XZ/YZ path views, live CSV values,
  episode statistics, diagnostic plots, derived time series, and downloads
  without leaving the Studio;
- the complete SP6 synchronized orbit viewer as an embedded case study, including
  the video, nearest measured rows, interactive 3-D path, temperature trace,
  interval sensitivity, and uncertainty notes;
- an enforced raw-first review sequence: create the raw detector overlay,
  confirm that it follows the visible particle, and only then render the
  smoothed overlay or derive motion;
- per-file run provenance and recent-run history so restored outputs are clearly
  identified as saved results rather than work rerun when the Studio opens;
- persistent job logs, cancellation, local session autosave, and in-tab MP4 playback;
- resume fields for outputs that were already created with the original scripts.

Use **Test a short preview** while tuning subtraction settings, then use
**Render full preview** before building the final overlays.

For a new recording, select or import the video, confirm the detected frame
rate, and choose a background. If there is no run-specific calibration yet,
click **Create calibration & test preview** and complete the guided camera-panel
and plate-point windows. Then load the calibrated camera views in the Studio,
click each particle in the same order in both views, and click **Track + create
overlay**. The Review section can display and download `tracks_3d.csv`,
trajectory plots and overlays, motion tables, and state-vector CSV files without
requiring separate command-line steps.

The Track Health panel is intentionally descriptive: it reports coverage,
missing frames, front/side vertical disagreement, frame-to-frame step size, and
chamber inclusion. It does not declare scientific pass/fail until a documented,
dataset-specific threshold system is established. Always review the trajectory
overlay before accepting or deriving motion from a track.

The Review page explains each artifact. In particular, the raw detector overlay
contains unsmoothed coordinates after the optional maximum-jump filter; it is not
the untouched pre-filter point set. Set maximum jump to `0` when a comparison
that retains every reconstructed point is required. Smoothing is Savitzky–Golay
post-processing, is saved alongside raw coordinates, and never repairs gaps or
identity mistakes.

Studio tracking supplies the confirmed frame rate to the tracker, so its CSV
tables include `time_s` alongside `frame` and the raw `x_mm`, `y_mm`, and `z_mm`
measurements. Command-line tracking remains backward compatible: when `--fps`
is omitted, no time column is invented.

The studio pauses for the original graphical interaction windows when a script
requires calibration points, focus boxes, particle seeds, lost-track corrections,
or identity decisions. Closing or completing that interaction returns control to
the same browser workflow. Generated paths are carried forward automatically.

On the Results page, click **Use current tracked CSV** to analyze the active
tracking output, or import another analyzer CSV. The detector recognizes common
`time_s`/`frame` and `x_mm`/`y_mm` column names; other schemas can be mapped in
the page. A confirmed frame rate is required only when the CSV has frames but no
time column. Multi-particle files require a particle ID. Every run is written to
`outputs_background/orbit_results/` with a source checksum, settings, run
manifest, interval CSV, derived time-series CSV, JSON summary, plots, and HTML
report. When a companion video is supplied, the report synchronizes playback to
the analysis time basis; confirm that the video and CSV share the same time zero.
Matroska inputs are remuxed losslessly to a derived MP4 beside the report so
browser playback does not depend on MKV support; the video frames are not
re-encoded.
The source CSV and companion video are never modified.

Circular-motion analysis uses two explicit stages. First, the bandpassed rolling
RMS amplitude finds candidate motion; this value is not labeled as a geometric
radius. Second, overlapping local fits model a linearly moving center plus the
orbital harmonic and check radial residual, axis ratio, angular coverage, and
winding consistency. The Results page exposes every geometry threshold and an
**RMS candidates only** compatibility mode. Reports distinguish candidate
intervals from validated episodes, show the moving-center circle-fit radius,
and retain each local fit and rejection reason in `orbit_timeseries.csv`.

## Translational Force Studio

The single-particle radius estimator is part of the existing Translational
Force Studio. Launch the one combined workflow from the project root:

```powershell
.\.venv\Scripts\python.exe -m two_particle_mediated_force.translational
```

Each recording keeps its own pressure, calibration, unlimited calibrated
video-snapshot clicks, individual radius estimates, and trial-specific masses.
Two selected individual estimates can be applied to the recording's existing
particle 1 and particle 2 force-model slots; extra clicks remain independent
observations in the radius distribution. The Studio autosaves the combined
setup locally.

For each recording, the Studio can explicitly run COM and force checks from
the canonical, raw, or smoothed position/velocity/acceleration columns. The
selected source is saved in the manifest and labeled in residual outputs.

The Studio also has a **Single-particle inference** mode. It accepts
`particle_state_vectors.csv`, lets you select a particle ID when the file
contains more than one, and uses `700 kg/m^3` datasheet particle density as the
recommended fixed constraint: radius is searched and mass is derived. Soft-prior
and independent mass-radius sensitivity modes remain available. The recommended
primary fit uses known forces only; the optional vertical background is labeled
as a sensitivity check because it can cancel gravity and confound radius. It then evaluates
the per-frame balance
`m a - (gravity + drag + thermophoresis)`. This mode reports the unexplained
residual force for that particle only; it intentionally skips interaction
fitting and the two-particle center-of-mass gate.

For SP recordings, each trial can use **Measured trace from raw video** as its
bottom-temperature input. Choose the original 1920x1080 video containing the
NI USB-TC01 graph and click **Extract 1 Hz measured trace**. The Studio
digitizes one value per logger second, aligns time zero to video frame zero,
shows a preview, and stores a reproducible CSV under
`outputs/mediated_force/translational/temperature_traces`. The force model
linearly interpolates those measured points at trajectory/ODE times; it does
not treat interpolated values as additional measurements and it refuses to
extrapolate beyond the trace. The constant bottom temperature remains a
fallback and sensitivity input.

The raw SP videos in the current cohort are 120 fps. Their active motion and
state-vector artifacts were regenerated at 120 fps in sibling folders ending
in `_fps120_20260904`; the original 60 fps artifacts were preserved. A
video-derived temperature trace will fail validation if an older 60 fps state
vector extends beyond the actual video clock.

The force workflow uses a calibrated 2-D axisymmetric temperature field rather
than a height-only profile. It solves steady variable-conductivity heat flow
between finite-radius plates, uses the plate diameters and gap from each
calibration, and exposes the ambient boundary temperature and field radius in
the Studio.

## Ceramic Glass Bubble Shape Analysis

A separate scale-free still-image analyzer is available for ceramic glass bubble
silhouettes, including conservative handling of touching clusters, manual review,
average normalized silhouettes, and deviation-magnitude distributions:

```powershell
.\.venv\Scripts\python.exe analyze_bubble_shapes.py
```

See [`bubble_shape_analysis/README.md`](bubble_shape_analysis/README.md) for the
guided workflow, batch command, output files, and interpretation notes.

To create a grand average from multiple reviewed dataset outputs:

```powershell
.\.venv\Scripts\python.exe combine_bubble_shape_datasets.py `
  --parent-dir "outputs_bubble_shapes\analysis_runs" `
  --name "all_ceramic_bubble_batches"
```

To project an average profile onto an axisymmetric 3D surface:

```powershell
.\.venv\Scripts\python.exe project_shape_profile_to_3d.py
```

This project reconstructs particle positions from orthogonal camera views using `trackpy`.

- Front camera horizontal position becomes `x_mm`.
- Side camera horizontal position becomes `y_mm`.
- The vertical image coordinate from either camera becomes `z_mm`.
- The chamber is modeled as a circular frustum with a 33.7 mm bottom plate, 25.4 mm top plate, and a configurable physical gap that defaults to 10.0 mm.

## Translational Two-Particle Mediated-Force Analysis

The Revision 5 translational force-identification package is located at
[`two_particle_mediated_force/translational`](two_particle_mediated_force/translational).
It consumes `pair_state_vectors.csv`, requires measured pressure for every
trial, uses Allen-Raabe Cunningham drag by default, compares reciprocal and
proportional-nonreciprocal interactions, and can fit a constrained shared
environmental background. Physical interpretation is gated on the unexplained
center-of-mass remainder. See the package README for the manifest format, CLI,
force-law choices, diagnostics, and generated outputs.

Launch the Translational Force Studio from the project root:

```powershell
.\.venv\Scripts\python.exe -m two_particle_mediated_force.translational
```

The script is located at:

```text
C:\Users\gmalb\OneDrive\Documents\Video Tracking Chin Lab\track_orthogonal_3d.py
```

There is also a shorter launcher at:

```text
C:\Users\gmalb\OneDrive\Documents\Video Tracking Chin Lab\track.py
```

## Install

From this folder:

```powershell
python -m pip install -r requirements.txt
```

If `python` is not on your PATH, install Python from python.org or Anaconda, then rerun the command in a new terminal.

The `av` and `imageio-ffmpeg` packages are included so PIMS can read common video formats such as MP4, MOV, and AVI.

## Quick Start With One Combined Screen Recording

Run:

```powershell
.\.venv\Scripts\python.exe track.py --gap-mm 10 --invert --save-calibration --force-calibration --select-focus-roi --select-initial-particles --particle-count 2 --camera-search-range-px 6 --seed-max-distance-px 30 --seed-size-margin 0.5
```

The script will ask you to choose one video. If `calibration.json` does not exist, it will show the first frame and ask you to select two camera-panel crops:

1. front camera ROI: click top-left, then bottom-right of the left/front panel
2. side camera ROI: click top-left, then bottom-right of the right/side panel

After that it will show each cropped camera view and ask you to click four calibration points:

1. bottom-left plate edge
2. bottom-right plate edge
3. top-left plate edge
4. top-right plate edge

With `--select-focus-roi`, it will also ask for a smaller search box inside each camera panel. Use this to exclude reflections, chamber edges, labels, and background noise.

With `--select-initial-particles --particle-count 2`, it will show frame 0 and ask you to click the two particles you care about in the front crop and then in the side crop. Click them in the same order in both views.

Use `--invert` when particles are dark on a bright background. Omit it when particles are bright on a dark background.

## Run With One Explicit Combined Video

```powershell
.\.venv\Scripts\python.exe track.py `
  --video "C:\path\to\combined_screen_recording.mp4" `
  --force-calibration `
  --calibration calibration.json `
  --gap-mm 10 `
  --diameter 11 `
  --minmass 100 `
  --camera-search-range-px 6 `
  --seed-max-distance-px 30 `
  --seed-size-margin 0.5 `
  --select-initial-particles `
  --particle-count 2 `
  --invert
```

If you already know the crop coordinates, pass them directly as `x,y,width,height`:

```powershell
.\.venv\Scripts\python.exe track.py `
  --video "C:\path\to\combined_screen_recording.mp4" `
  --front-roi "0,0,970,580" `
  --side-roi "970,0,1075,580" `
  --gap-mm 10 `
  --save-calibration `
  --invert
```

Those sample ROI numbers match the approximate layout shown in the screenshot: left camera panel on top-left, right camera panel on top-right, and the lower UI excluded. Adjust them if your actual video resolution or window layout differs.

## Two Separate Videos Still Work

```powershell
.\.venv\Scripts\python.exe track.py `
  --front "C:\path\to\front_video.mp4" `
  --side "C:\path\to\side_video.mp4" `
  --calibration calibration.json `
  --invert
```

## Outputs

By default, results are written to:

```text
C:\Users\gmalb\OneDrive\Documents\Video Tracking Chin Lab\outputs
```

The main files are:

- `tracks_3d.csv`: linked 3D trajectories with `particle`, `frame`, `x_mm`, `y_mm`, and `z_mm`.
- `initial_particle_seeds_3d.csv`: clicked starting particles converted to approximate 3D coordinates, when seed mode is used.
- `matched_3d_features.csv`: frame-by-frame front/side matches before trajectory linking.
- `front_detections_mm.csv`: front-camera detections converted to `x_mm` and `front_z_mm`.
- `side_detections_mm.csv`: side-camera detections converted to `y_mm` and `side_z_mm`.
- `tracks_3d.png`: 3D track plot inside the chamber wireframe.
- `track_projections.png`: XY, XZ, and YZ projections.
- `z_match_errors.png`: diagnostic histogram of front-vs-side z disagreement.
- `summary.json`: counts and chamber metadata.

## Calibration Notes

The real chamber dimensions are known, but every camera still needs pixel calibration. You can either click the ROI and calibration points interactively or copy `calibration.example.json` to `calibration.json` and replace the sample pixel coordinates.

For combined videos, `front_roi` and `side_roi` are full-frame coordinates. The `front` and `side` calibration points are coordinates inside their cropped ROI images. The bottom plate diameter is used at `z = 0`, the top plate diameter is used at the configured gap (10.0 mm by default), and scale is interpolated between those heights. Pass `--gap-mm` whenever the measured spacing differs; with an existing calibration, combine it with `--save-calibration` to persist the corrected value.

Use `--force-calibration --save-calibration` whenever the screen-recording layout, crop boxes, or visible plate edges change.

## Tuning Trackpy

Start with a short test:

```powershell
.\.venv\Scripts\python.exe track.py --max-frames 100 --diameter 11 --minmass 100 --select-initial-particles --particle-count 2 --camera-search-range-px 6 --seed-max-distance-px 30 --seed-size-margin 0.5 --invert
```

Useful knobs:

- `--diameter`: approximate particle diameter in pixels; Trackpy requires an odd integer.
- `--minmass`: increase to reject dim/noisy features, decrease if real particles are missed.
- `--max-features-per-frame`: keeps only the strongest N detections per frame after Trackpy finds candidates.
- `--select-focus-roi`: lets you click a smaller particle search region inside each camera crop.
- `--select-initial-particles`: lets you click the actual particle(s) on the starting frame and keeps only the nearest linked tracks.
- `--camera-search-range-px`: maximum 2D jump per frame while following clicked particles in each camera.
- `--seed-max-distance-px`: how close the first linked 2D track must be to your click.
- `--seed-size-margin`: seeded mode keeps detections within this fractional Trackpy-size range of the clicked particles. `0.5` means within 50%.
- `--seed-mass-filter --seed-mass-margin 0.5`: optional extra filter using brightness/mass of clicked particles.
- `--max-feature-size`, `--max-mass`, `--max-ecc`: optional manual blob rejection filters.
- `--threshold`: optional brightness threshold.
- `--separation`: optional minimum feature spacing in pixels.
- `--z-match-tolerance-mm`: increase if front and side z calibration disagree.
- `--search-range-mm`: maximum 3D particle displacement per frame during linking.
- `--memory`: number of frames Trackpy can bridge when a particle is temporarily missing.
- `--stub-length`: removes trajectories shorter than this many frames.

## Important Assumptions

For a single combined screen recording, synchronization is already frame-for-frame because both views are in the same video frame. For two separate video files, the two videos should be synchronized frame-for-frame.

Pairing front and side detections uses same-frame `z` agreement. This is robust for sparse particles, but ambiguous when several particles share nearly the same `z` in the same frame.

Trackpy expects a reasonably stable particle image size and contrast. Check `matched_3d_features.csv` and `z_match_errors.png` after a first run before trusting the final trajectories.

## Background Subtraction App

There is also a separate tracker that does not use Trackpy linking:

```text
C:\Users\gmalb\OneDrive\Documents\Video Tracking Chin Lab\background_subtraction_tracker.py
```

It asks you to choose the tracking video and background image/video from File
Explorer when paths are not given, asks you to click the particles on frame 1,
and follows local movement in the background-subtracted image. Each run creates
one folder inside `outputs_background\tracks`.

The short default tracker command is:

```powershell
.\.venv\Scripts\python.exe background_subtraction_tracker.py
```

The tracker now performs position smoothing before it writes `tracks_3d.csv`.
The original measurements remain in `x_mm`, `y_mm`, and `z_mm` (with explicit
`x_raw_mm`, `y_raw_mm`, and `z_raw_mm` aliases). The smooth track is saved in
`x_smooth_mm`, `y_smooth_mm`, and `z_smooth_mm`, together with raw and smooth
front/side pixel projections for verification. Smoothing uses a 7-point,
second-order Savitzky-Golay filter by default and never crosses a missing-frame
gap. Change it at tracking time with `--smooth-window 9 --smooth-polyorder 2`,
or use `--smooth-window 1` to make the smooth copy equal the raw track.

The tracker always asks you to choose a background image/video unless you pass
one directly with `--background-file`, `--background-image`, or
`--background-video`.

Manual correction is on by default. The tracker accepts weaker particle signal
than before and waits 3 missed frames before it asks for help. Close-particle
prompts are off by default; turn them on with
`--manual-correction-close-px N` only when you want that extra warning. When a
particle is lost, the tracker asks you to re-click the particle IDs in order on
the background-subtracted frame. The correction window also has an `End track
here` button; use it when a collision/sticking event makes every later frame
useless. The output is clipped to the last clean frame before that prompt. If
the yellow labels are already correct, use `Accept current` to keep those
coordinates without re-clicking. To run without correction prompts:

```powershell
.\.venv\Scripts\python.exe background_subtraction_tracker.py --no-manual-correction
```

Current less-sensitive defaults are equivalent to:

```powershell
--min-difference 4 --threshold-fraction 0.35 --manual-correction-after-misses 3 --manual-correction-close-px 0
```

Within each frame, the tracker now assigns visible background-subtraction
components one-to-one: one bright component cannot be claimed by both particle
IDs. If two IDs would otherwise collapse onto the same visible component, one
is treated as missing instead of silently duplicating the same coordinate.
When one camera view genuinely sees the particles overlapped for many frames,
use an overlap range so that camera may assign both IDs to the same visible
component without repeated correction prompts:

```powershell
.\.venv\Scripts\python.exe background_subtraction_tracker.py --front-overlap-frames 2200-2350
```

Use `--side-overlap-frames 2200-2350` for the side view. Multiple inclusive
ranges can be comma-separated, such as `--front-overlap-frames 2200-2350,2600-2680`.
When overlap mode is used, all tracking CSVs are tagged with
`overlap_mode_enabled`, per-camera overlap settings, and per-frame overlap
flags.

By default, `--identity-authority auto` uses the clear camera to preserve
particle IDs during one-camera overlap ranges: side view is authority during
front overlap, and front view is authority during side overlap. Disable this
with `--identity-authority none`, or force one camera with
`--identity-authority front` or `--identity-authority side`.

There is also an optional z-pairing check before `tracks_3d.csv` is written:

```powershell
.\.venv\Scripts\python.exe background_subtraction_tracker.py --z-pairing-check
```

For each frame, it compares the same-ID front/side z pairing against the
best front/side z-agreement pairing, then remaps side IDs only if the z-error
improvement is at least `--z-pairing-min-improvement-mm` (default `0.05` mm).
This is off by default because noisy calibration or particles with very similar
z positions could otherwise cause false swaps.

If IDs still switch after inspection, correct an existing CSV with:

```powershell
.\.venv\Scripts\python.exe swap_particle_ids.py --frames 2200-2350
```

Choose the CSV from File Explorer. The script swaps particles `0` and `1` by
default, writes a corrected CSV inside the selected trial's `corrections`
folder, and logs the correction in `identity_swap_log.json`.

To detect likely ID switches from abnormal 3D jumps and review them one by one:

```powershell
.\.venv\Scripts\python.exe detect_identity_switches.py
```

Choose `tracks_3d.csv` from File Explorer. Suspicious frames are found by
comparing the normal same-ID jump cost against the cost if particle IDs were
swapped. Manual review lets you choose `Swap From Here`, `Swap This Frame`,
`Keep`, or `Stop Review`. Outputs are written beside the trial in:

```text
outputs_background\tracks\TRIAL_NAME\corrections\identity_correction_TIMESTAMP
```

The corrected CSV can be used directly with `verify_tracking_overlay.py`,
`calculate_motion.py`, and `analyze_motion_correlations.py`.

If the automatic detector misses a visually obvious problem, force review over
a known frame range:

```powershell
.\.venv\Scripts\python.exe detect_identity_switches.py --review-frames 3800-4300
```

Use `--review-frame-step 10` to inspect every 10th complete frame in that range.

By default, `detect_identity_switches.py` reviews only frames already marked by
the tracker as manual-correction frames. Use `--include-auto-candidates` to
review automatic jump, forced-range, and size-signature candidates instead. If
you want both automatic candidates and tracker manual-correction frames in one
run, add `--review-manual-corrections` explicitly.

The size-signature mode can use one camera at a time:

```powershell
.\.venv\Scripts\python.exe detect_identity_switches.py `
  --csv "path\to\tracks_3d.csv" `
  --video "Videos\MI22.mkv" `
  --calibration "calibrations\MI22_calibration.json" `
  --include-auto-candidates `
  --size-signature `
  --size-camera front `
  --size-polarity abs
```

To choose the background image or video from File Explorer explicitly:

```powershell
.\.venv\Scripts\python.exe background_subtraction_tracker.py --choose-background
```

Use `--difference-mode bright` for particles that become brighter than the background, or `--difference-mode dark` for dark particles on a brighter background. If frame 0 already contains the particles, use a later `--seed-frame` and expect a faint subtraction ghost at the starting position.

To preview what background subtraction looks like before tracking, run:

```powershell
.\.venv\Scripts\python.exe background_subtraction_preview.py
```

Video names containing an `MI##` or `SP##` trial ID automatically use the
matching `calibrations\<trial>_calibration.json` file. For example, an `SP1`
video resolves to `calibrations\SP1_calibration.json`.

The preview also asks you to choose a background image/video unless you pass one
directly. To choose the background image/video from File Explorer explicitly:

```powershell
.\.venv\Scripts\python.exe background_subtraction_preview.py --choose-background
```

Preview videos are written into separate run folders in:

```text
outputs_background\previews
```

The folder name follows the tracking-output style: video name plus timestamp.
The MP4 inside uses the video name, such as `MI8_online-video-cutter.com_preview.mp4`.
Use `--output-name my_preview` if you want to name the run folder.

The preview writes the whole video by default; use `--max-frames N` only when
you want a shorter test preview. The background-subtraction preview and tracker
ask you to choose fresh focus boxes every run. The default
`--focus-box-mode follow` permanently deletes everything outside the starting
box except small moving windows around already-clicked particles, so particles
can leave the original box without bringing the rest of the outside junk back.
If the preview still clips a particle as it leaves, increase
`--particle-window-radius-px` such as `--particle-window-radius-px 80`.
If dim specks reappear only around the moving particle window, increase
`--outside-window-display-threshold` such as `--outside-window-display-threshold 16`.
Use `--focus-box-mode hard` only when you want particles to disappear when they
leave the box. Use `--focus-box-mode seed-only` only when the box should affect
the initial click view and nothing else. Add `--no-focus-box` only when you want
to disable focus boxes. Exact boxes can also be passed as
`--front-focus-box "x,y,width,height"` and `--side-focus-box "x,y,width,height"`;
these coordinates are inside each camera crop, not full-video coordinates.
Direct background paths can be passed with `--background-file`,
`--background-image`, or `--background-video`. If the chosen background is a
video, frame 0 is used by default; if it is an image, the image is used directly
as the background frame.

## Diameter and Mass Estimates

To estimate approximate particle diameters and ice masses from a preview video
or image, run:

```powershell
.\.venv\Scripts\python.exe estimate_particle_masses.py
```

Choose the preview file, then type `front` or `side` depending on which camera
view is better focused for that trial. Results are written to:

```text
outputs_background\mass
```

Each trial gets its own folder containing `diameters.csv`,
`diameter_summary.csv`, and `metadata.json`.

## Velocity and Acceleration

After running the background subtraction tracker, calculate motion from a
`tracks_3d.csv` file with:

```powershell
.\.venv\Scripts\python.exe calculate_motion.py
```

Choose the `tracks_3d.csv` file from File Explorer. The script defaults to
`60 fps`; pass the raw video's actual rate (the SP cohort uses `--fps 120`). It
writes a timestamped run folder under:

```text
outputs_background\motion
```

Outputs include:

- `velocities_accelerations.csv`
- `motion_summary.csv`
- `metadata.json`
- `position_coordinates.png`: raw and Savitzky-Golay-smoothed x/y/z coordinates
- `velocity_comparison.png`: frame-difference and smooth-curve velocity components and speed
- `acceleration_comparison.png`: frame-difference and smooth-curve acceleration components and magnitude
- `motion_plots.png`

For background-subtraction tracks, `calculate_motion.py` consumes the smooth
positions already stored by the tracker; it does not replace them with a newly
smoothed coordinate track. The canonical velocity and acceleration columns in
`velocities_accelerations.csv` come from derivatives of the same local
Savitzky-Golay polynomial whenever the saved tracker window supports them. The
CSV also retains finite-difference values in columns containing `_raw_` for
comparison. Tracks are split at missing-frame gaps before differentiation. The
motion script's `--smooth-window` and `--smooth-polyorder` options remain only
as a compatibility fallback for older or Trackpy CSVs that do not contain the
tracker-generated smooth columns.

Use `--fps 120` or another value if the video frame rate is different.

## Translational State Vectors and Comparison Overlay

Build state vectors from the new motion CSV with:

```powershell
.\.venv\Scripts\python.exe build_state_vectors.py `
  --csv "outputs_background\motion\RUN_NAME\velocities_accelerations.csv" `
  --state-source raw
```

If the motion CSV contains only one particle, the builder automatically uses
single-particle mode. To select one particle from a multi-particle CSV, pass its
particle ID explicitly:

```powershell
.\.venv\Scripts\python.exe build_state_vectors.py `
  --csv "outputs_background\motion\RUN_NAME\velocities_accelerations.csv" `
  --particles 0 `
  --state-source raw
```

Single-particle mode writes `particle_state_vectors.csv` and `metadata.json`.
Pair, relative, and cross-particle prediction outputs are omitted because there
is no second particle.

The canonical `x_mm` through `az_mm_s2` state remains the raw, non-smoothed
state by default. `particle_state_vectors.csv`,
`pair_state_vectors.csv`, and `relative_state_vectors.csv` also retain explicit
raw and smoothed columns plus separate JSON vectors, so the two representations
can be audited without rebuilding. Pass `--state-source smoothed` when the
canonical columns consumed by prediction or force analysis should instead use
smoothed positions and smooth-curve derivatives.

To compare both representations directly on the preview video, run:

```powershell
.\.venv\Scripts\python.exe verify_state_vectors_overlay.py `
  --preview "debug_video_samples\MI22_cleanup_smoke_preview.mp4" `
  --tracks-csv "outputs_background\tracks\MI22\tracks_3d.csv" `
  --state-csv "outputs_background\state_vectors\RUN_NAME\particle_state_vectors.csv" `
  --calibration "calibrations\MI22_calibration.json" `
  --state-source both
```

Comparison mode is the default and writes four separate videos so neither
representation is removed or visually mixed with the other:

- `*_state_vectors_raw_scaled.mp4`
- `*_state_vectors_raw_normalized.mp4`
- `*_state_vectors_smoothed_scaled.mp4`
- `*_state_vectors_smoothed_normalized.mp4`

Use `--state-source raw` or `--state-source smoothed` only when you intentionally
want the corresponding two-video subset.

## Rotation State Vectors With Overlay

After `estimate_particle_rotation.py` has created `rotation_state_vectors.csv`,
combine it with `velocities_accelerations.csv` and create a full state-vector
preview overlay with:

```powershell
.\.venv\Scripts\python.exe build_rotation_state_vectors_with_overlay.py `
  --rotation-csv "outputs_background\rotation\MI22_rotation_state_vectors\rotation_state_vectors.csv" `
  --motion-csv "outputs_background\motion\MI22_identity_correction_20260710_145437_20260710_151106\velocities_accelerations.csv" `
  --preview "debug_video_samples\MI22_cleanup_smoke_preview.mp4" `
  --calibration "calibrations\MI22_calibration.json" `
  --output-name "MI22_rotation_state_vectors"
```

That creates `particle_reduced_state_vectors.csv`, which contains the original
linear motion columns plus `n_x,n_y,n_z`, `q_w,q_x,q_y,q_z`, and
`omega_x/y/z_rad_s` from the rotation CSV. The overlay uses the motion CSV's
front/side image positions and draws position markers, velocity arrows,
acceleration arrows, reconstructed orientation axes, omega arrows when finite,
clean zoom panels, and per-particle numeric readouts for position, velocity,
acceleration, quaternion `q`, and angular velocity `w`.

The combined state-vector CSV is written under:

```text
outputs_background\rotation_state_vectors\RUN_NAME
```

The automatic preview overlay is written separately under:

```text
outputs_background\rotation_overlays\RUN_NAME
```

Overlay outputs include `{preview_name}_rotation_state_vectors_overlay.mp4` and
`preview_rotation_overlay_metadata.json`. Use `--no-overlay` to build only the
combined CSV, or `--max-frames N` for a short overlay test run. Velocity and
acceleration arrows are normalized by default so their directions are visible;
use `--motion-vector-mode scaled` to draw physically scaled arrows instead.

## Reduced Rotation State Vectors

To estimate the observable tumbling rotation from the tracked particle images,
run:

```powershell
.\.venv\Scripts\python.exe estimate_particle_rotation.py `
  --video "Videos\MI22.mkv" `
  --tracks-csv "outputs_background\tracks\MI22\tracks_3d.csv" `
  --calibration "calibrations\MI22_calibration.json" `
  --background-file "Backgrounds\MI22_Frame.png" `
  --difference-mode abs `
  --fps 60
```

The script crops each camera panel around the known centroids in
`tracks_3d.csv`, estimates the projected particle long axis in the front and
side views, reconstructs one 3D body-axis direction `n_x,n_y,n_z`, accumulates a
minimum-twist reduced-attitude quaternion, and estimates observable tumbling
angular velocity.

Front and side views are calibrated independently, including their vertical
`z` pixel scale. If one view is blurrier or has a broader apparent particle
extent, tune only that camera instead of changing both:

```powershell
.\.venv\Scripts\python.exe estimate_particle_rotation.py `
  --video "Videos\MI22.mkv" `
  --tracks-csv "outputs_background\tracks\MI22\tracks_3d.csv" `
  --calibration "calibrations\MI22_calibration.json" `
  --background-file "Backgrounds\MI22_Frame.png" `
  --difference-mode abs `
  --side-crop-height-px 55 `
  --side-minimum-anisotropy 0.10
```

The 3D reconstruction also weights each camera by its measured axis quality, so
a blurrier valid view contributes less than a sharper, more anisotropic view.

The main output is:

- `rotation_state_vectors.csv`
- `metadata.json`
- `rotation_diagnostics.png`, unless `--no-plots` is used
- `outputs_background\rotation_overlays\RUN_NAME\rotation_axis_overlay.mp4`, unless `--no-overlay` is used
- `outputs_background\rotation_overlays\RUN_NAME\{preview_name}_rotation_overlay.mp4`, unless `--no-preview-overlay` is used
- `outputs_background\rotation_overlays\RUN_NAME\preview_rotation_overlay_metadata.json`, unless `--no-preview-overlay` is used

The native `rotation_axis_overlay.mp4` and the optional preview overlay both
replace the lower unused area with clean front/side zoom crops for each
particle, without markers or axis overlays inside those crops. If a particle is
temporarily missing during a manual-correction gap, its zoom crop holds on the
latest tracked-axis frame until a newer tracked-axis row appears. The preview
overlay uses the same preview-frame mapping style as
`verify_state_vectors_overlay.py`. Cyan lines show the raw per-camera 2D PCA
axes when available. Yellow lines show the reconstructed 3D body axis projected
back into each camera view. Add `--draw-omega` to draw the observable tumbling
angular-velocity projection, or use `--axes reconstructed-3d` to show only the
reconstructed axis. If no `--preview` path is supplied, the script asks you to
choose the matching background-subtracted preview video. Use `--no-zoom-panels`
to disable the clean zoom crops, or `--no-preview-overlay` to skip the preview
prompt/output.

If you already ran `calculate_motion.py`, pass its output to also create the
combined modeling table:

```powershell
.\.venv\Scripts\python.exe estimate_particle_rotation.py `
  --video "Videos\MI22.mkv" `
  --tracks-csv "outputs_background\tracks\MI22\tracks_3d.csv" `
  --motion-csv "outputs_background\motion\MI22\velocities_accelerations.csv" `
  --calibration "calibrations\MI22_calibration.json" `
  --background-file "Backgrounds\MI22_Frame.png" `
  --difference-mode abs
```

That writes `particle_reduced_state_vectors.csv` with state vectors ordered as:

```text
x_mm,y_mm,z_mm,
vx_mm_s,vy_mm_s,vz_mm_s,
ax_mm_s2,ay_mm_s2,az_mm_s2,
n_x,n_y,n_z,
omega_x_rad_s,omega_y_rad_s,omega_z_rad_s
```

These are reduced rotation states, not full rigid-body attitude states. The
software reports `orientation_type = minimum_twist_reduced_attitude`,
`angular_velocity_type = observable_tumbling_component`,
`axial_spin_observable = false`, and
`axial_spin_assumption_rad_s = 0.0`. Spin about the visible particle axis is not
measured by this pipeline.

To re-run only the preview overlay later from an existing rotation CSV, run:

```powershell
.\.venv\Scripts\python.exe verify_rotation_overlay.py `
  --preview "debug_video_samples\MI22_cleanup_smoke_preview.mp4" `
  --rotation-csv "outputs_background\rotation\MI22_rotation\rotation_state_vectors.csv" `
  --calibration "calibrations\MI22_calibration.json" `
  --preview-start-frame 1
```

## Motion Correlations

After creating `velocities_accelerations.csv`, calculate particle-to-particle
motion/acceleration correlations and distance-volatility correlations together
with:

```powershell
.\.venv\Scripts\python.exe analyze_combined_correlations.py
```

Choose `velocities_accelerations.csv` from File Explorer. This runs the same
outputs that `analyze_motion_correlations.py` and
`analyze_distance_volatility.py` would produce individually, using one CSV
selection.

To run only the velocity/acceleration correlation script:

```powershell
.\.venv\Scripts\python.exe analyze_motion_correlations.py
```

Results are written to:

```text
outputs_background\correlations
```

The main output is `pairwise_correlations.csv`. It contains both unlagged and
lagged correlations; unlagged rows are the rows where `lag_frames` is `0`.
Use `--max-lag-frames 0` if you only want unlagged correlations.

Other outputs include:

- `best_lag_summary.csv`
- `vector_alignment_by_frame.csv`
- `vector_alignment_summary.csv`
- `zero_lag_correlations.png`
- `lagged_correlations.png`

## Tracking Verification Overlay

To visually check whether `tracks_3d.csv` follows the particles in a preview
video, run:

```powershell
.\.venv\Scripts\python.exe verify_tracking_overlay.py
```

Choose the preview video, then choose the matching `tracks_3d.csv`. The script
draws one colored marker per particle on both the front and side preview panels
for each matching frame. Every run writes two separate videos so the raw track
is never removed or mixed with the smoothed one:

- `*_verification_raw.mp4`
- `*_verification_smoothed.mp4`

Results are written to:

```text
outputs_background\verification
```

By default, preview video frame 0 is matched to tracking frame 1, which matches
the current preview/tracker defaults. If you made a preview with a different
start frame, pass `--preview-start-frame N`.

## SP history prediction and Physics Model v1

The SP Trajectory Studio now offers **Future-position prediction v2** and
**Physics force model v1** as separate categories. New v2 analyses use causal
motion history, buffered time blocks, separate uncertainty calibration, and
one-video evaluation. The force model adds a weak-form effective correction to
explicit thermophoresis, drag, and gravity, with required independent physical
inputs. Existing analysis results remain available under the legacy protocol.

Open `launch_sp_transition_v2.py` or `launch_sp_physics_v1.py` using the project
Python environment. Training starts only when requested in the website. See
`trajectory_ml/README.md` for scientific assumptions, protocol details and outputs.
