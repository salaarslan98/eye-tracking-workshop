"""
step_04_label_intervals.py -- turn watching a clip into labelled intervals.

Used for two things in this workshop, same tool both times:
  1. Blink labels on a blinks_<name>.mp4 clip: mark where the eyes are
     actually closed, to check the step_02 fixed-threshold detector against
     real ground truth and to train/evaluate the blink classifier (step_05).
  2. Gaze-direction labels on an eyes_only_<name>.mp4 clip: mark which
     direction the person is actually looking, to train/evaluate the
     gaze-direction classifier (step_06).

Why a tool instead of typing remembered timestamps: the looks and blinks in
a clip were not recorded at known times, so the only honest source of
ground truth is watching the footage and judging it, frame by frame. This
script makes that judging fast and saves it as a CSV the later steps read.

It does NOT run MediaPipe. It reuses the per-frame EAR / gaze numbers
already saved by step_02 / step_03, and only opens the video for display.
That keeps it fast to start and matches the project rule: save
intermediate results once, do not recompute them.

Controls:
  , / .     step one frame back / forward
  SPACE     play / pause (plays at the clip's own fps)
  b         [blink mode] toggle "eyes closed" on/off at the current frame
  1-5       [gaze mode] set the current label: 1 centre 2 left 3 right 4 up 5 down
  u         undo the most recently finished interval
  s         save progress to CSV without quitting
  q         save and quit

Before running, set three things below: MODE, CLIP_NAME and LABELLER.
Run it once per clip and mode (change the settings and rerun):
  python step_04_label_intervals.py

In a group, every blinks clip and every eyes_only clip needs labelling, so
share the work out. The output file is named after the clip AND the
labeller (e.g. labels_blinks_alex__sam.csv is Sam's labelling of Alex's
clip), so one clip can also be labelled independently by two people. That
is useful for checking how much people agree on where a blink actually
starts and ends (see STUDENT_TASKS.md task 1b).

Agree on your definitions before anyone starts, and write them down: what
counts as "closed", and whether "left" means left in the picture or the
person's own left. Do not decide frame by frame as you go.
"""
import os
import cv2
import pandas as pd

# ---- set these before running ----
MODE = "blink"      # "blink" or "gaze"
CLIP_NAME = "replace_with_clip_name"   # the <name> part of the clip's file name,
                                        # i.e. who is IN the clip
LABELLER = "replace_with_your_name"    # who is doing the labelling (you)

if MODE == "blink":
    VIDEO = f"blinks_{CLIP_NAME}.mp4"
    FEATURES_CSV = f"ear_blinks_{CLIP_NAME}.csv"         # made by step_02
    VALUE_COLS = ["ear"]
    LABELS = {ord("b"): None}       # 'b' toggles; handled specially below
    LEGEND = "b = toggle eyes closed/open"
elif MODE == "gaze":
    VIDEO = f"eyes_only_{CLIP_NAME}.mp4"
    FEATURES_CSV = f"features_eyes_only_{CLIP_NAME}.csv"  # made by step_03
    VALUE_COLS = ["gaze_x", "gaze_y"]
    LABELS = {ord("1"): "centre", ord("2"): "left", ord("3"): "right",
              ord("4"): "up", ord("5"): "down"}
    LEGEND = "1 centre | 2 left | 3 right | 4 up | 5 down"
else:
    raise ValueError("MODE must be 'blink' or 'gaze'")

# TRY IT: the video is shrunk to this many pixels wide so it fits on the
# screen. Make it larger if you have the room: small eye movements and the
# first frame of a blink are easier to judge on a bigger picture.
DISPLAY_WIDTH = 1000

# stop early with a plain message instead of a confusing error later on
if CLIP_NAME == "replace_with_clip_name" or LABELLER == "replace_with_your_name":
    raise SystemExit("Open this file and set CLIP_NAME and LABELLER (near the top) first.")
if not os.path.exists(VIDEO):
    raise SystemExit(f"{VIDEO} not found. Check CLIP_NAME and the clip's file name (see README.md).")
if not os.path.exists(FEATURES_CSV):
    raise SystemExit(f"{FEATURES_CSV} not found. Run the step_02 and step_03 notebooks first.")

STEM = os.path.splitext(os.path.basename(VIDEO))[0]
# two underscores separate the clip from the labeller, so the later steps
# can tell them apart even if a name itself contains an underscore
OUT_CSV = f"labels_{STEM}__{LABELLER}.csv"


class IntervalRecorder:
    """Tracks one label active at a time and finalises it into intervals.

    Kept free of cv2/display code on purpose, so the labelling logic can be
    tested without opening a video or a window.
    """

    def __init__(self, start_frame, start_label):
        self.done = []                      # finished (label, start, end) frames, end inclusive
        self.cur_label = start_label
        self.cur_start = start_frame

    def set_label(self, frame_idx, label):
        """Close the current interval and open a new one, if the label changed."""
        if label == self.cur_label:
            return
        if frame_idx > self.cur_start:
            self.done.append((self.cur_label, self.cur_start, frame_idx - 1))
        self.cur_label = label
        self.cur_start = frame_idx

    def undo(self):
        if self.done:
            self.cur_label, self.cur_start, _ = self.done.pop()

    def snapshot(self, up_to_frame):
        """Finished intervals plus the still-open one, closed at up_to_frame."""
        rows = list(self.done)
        if up_to_frame >= self.cur_start:
            rows.append((self.cur_label, self.cur_start, up_to_frame))
        return rows

    def to_dataframe(self, up_to_frame, fps):
        rows = self.snapshot(up_to_frame)
        return pd.DataFrame(
            [{"label": lab, "start_frame": s, "end_frame": e,
              "start_s": round(s / fps, 3), "end_s": round(e / fps, 3),
              "duration_s": round((e - s + 1) / fps, 3)} for lab, s, e in rows]
        )


def main():
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {VIDEO}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    feats = pd.read_csv(FEATURES_CSV)

    initial_label = "open" if MODE == "blink" else "centre"
    rec = IntervalRecorder(0, initial_label)
    closed = False  # blink-mode toggle state; always follows rec.cur_label

    cache = {}       # small frame cache: idx -> frame, so scrubbing near a
    cache_order = []  # transition doesn't re-decode the same frame repeatedly
    CACHE_MAX = 300

    def get_frame(idx):
        if idx in cache:
            return cache[idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            frame = cache[cache_order[-1]]  # fall back to the last good frame
        cache[idx] = frame
        cache_order.append(idx)
        if len(cache_order) > CACHE_MAX:
            del cache[cache_order.pop(0)]
        return frame

    print(f"Labelling {VIDEO} in '{MODE}' mode ({n_frames} frames, {fps:.1f} fps)")
    print(f"Controls: , / . step | SPACE play/pause | {LEGEND} | u undo | s save | q save+quit")

    idx = 0
    playing = False
    while True:
        frame = get_frame(idx).copy()
        w = frame.shape[1]
        if w > DISPLAY_WIDTH:
            scale = DISPLAY_WIDTH / w
            frame = cv2.resize(frame, None, fx=scale, fy=scale)

        row = feats.iloc[idx] if idx < len(feats) else None
        value_txt = " ".join(f"{c}={row[c]:+.3f}" for c in VALUE_COLS) if row is not None else "no feature row"
        # the recorder is the single source of truth for the current label,
        # so what is shown is always what will be saved
        state_txt = rec.cur_label
        lines = [
            f"frame {idx}/{n_frames - 1}  t={idx / fps:5.2f}s",
            value_txt,
            f"current label: {state_txt}",
            LEGEND,
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, 25 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow(f"step 4 - label intervals ({MODE})", frame)
        wait_ms = max(1, int(1000 / fps)) if playing else 0
        key = cv2.waitKey(wait_ms) & 0xFF

        if playing:
            idx = min(idx + 1, n_frames - 1)
            if idx == n_frames - 1:
                playing = False
            if key == 255:  # no key pressed this tick, keep playing
                continue

        if key in (ord(","),):
            idx = max(idx - 1, 0)
        elif key in (ord("."),):
            idx = min(idx + 1, n_frames - 1)
        elif key == ord(" "):
            playing = not playing
        elif key == ord("u"):
            rec.undo()
            # undo reopens the previous interval, so take the toggle state
            # from the recorder instead of trusting the old flag
            closed = rec.cur_label == "closed"
        elif key == ord("s"):
            rec.to_dataframe(idx, fps).to_csv(OUT_CSV, index=False)
            print(f"Saved progress to {OUT_CSV}")
        elif key == ord("q"):
            rec.to_dataframe(idx, fps).to_csv(OUT_CSV, index=False)
            break
        elif MODE == "blink" and key == ord("b"):
            closed = not closed
            rec.set_label(idx, "closed" if closed else "open")
        elif MODE == "gaze" and key in LABELS:
            rec.set_label(idx, LABELS[key])

    cap.release()
    cv2.destroyAllWindows()

    df = rec.to_dataframe(idx, fps)
    print(f"\nSaved {OUT_CSV}: {len(df)} intervals")
    if len(df):
        print(df.groupby("label")["duration_s"].agg(["count", "sum"]).rename(columns={"sum": "total_s"}))


if __name__ == "__main__":
    main()
