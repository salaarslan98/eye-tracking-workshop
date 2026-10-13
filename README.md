# Gaze and blink tracking workshop

Everything needed to run the workshop is in this folder. Work through the
steps in order: each one makes files that a later one reads.

The steps run on short video clips that **you record yourself**: three
clips per person. You work in a group of two or three, so a group has six
or nine clips, and the models can be trained on more than one person.

Throughout the notebooks and the labelling script, comments marked
**`TRY IT`** point at a value you are meant to change. Run the step once as
it is, then go back and experiment.

## Get the files

With git installed, copy the repository to your laptop and go into it:

```bash
git clone https://github.com/salaarslan98/eye-tracking-workshop.git
cd eye-tracking-workshop
```

No git? On the repository's GitHub page choose *Code > Download ZIP* and
unpack it. All commands below are run from inside this folder.

## 1. Before you start (once)

You need **Python 3.12** (64-bit). Python 3.10 and 3.11 should also work
(untested). **Python 3.13 does not work**: the pinned MediaPipe version
(0.10.14) has no build for it. Check what you have:

| System | Check | If missing |
|---|---|---|
| Windows | `py -3.12 --version` | Install 3.12 from python.org. Tick "Add python.exe to PATH" in the installer. |
| macOS | `python3 --version` | Install 3.12 from python.org, or `brew install python@3.12` if you use Homebrew. |
| Linux / WSL | `python3 --version` | `sudo apt install python3 python3-venv` (Ubuntu). |

Then open a terminal **in this folder** and make an environment (a private
set of packages, so nothing here clashes with your other Python projects).
Pick your system. You do this once.

**Windows (PowerShell):**

```powershell
py -3.12 -m venv my-venv
.\my-venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell refuses with "running scripts is disabled on this system", run
this once in the same window and activate again. It only affects that
window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Using the old Command Prompt (`cmd`) instead? Activate with
`my-venv\Scripts\activate.bat`.

**macOS (Terminal):**

```bash
python3 -m venv my-venv
source my-venv/bin/activate
pip install -r requirements.txt
```

**Linux or WSL:** the same three lines as macOS.

When the environment is active, your prompt starts with `(my-venv)`. **Every
time you open a new terminal, activate it again** (the middle line only):
`.\my-venv\Scripts\Activate.ps1` on Windows, `source my-venv/bin/activate`
on macOS, Linux and WSL. From here on, `python` means the one in the
environment on every system.

Then check the install:

```bash
python setup_01_check_install.py
```

It should end with `Face mesh loaded OK`. MediaPipe also prints warnings
(libEGL, TensorFlow Lite, "feedback manager", protobuf). They are harmless.

If the install fails or the check crashes:

- **"No matching distribution found for mediapipe==0.10.14"**: your Python
  is the wrong version (for example 3.13). Make the environment again with
  3.12.
- **A DLL error on Windows when importing `cv2` or `mediapipe`**: installing
  the current "Microsoft Visual C++ Redistributable" usually fixes it
  (a common cause, not verified on every machine).
- Do not "fix" things by upgrading MediaPipe or installing plain
  `opencv-python`. The scripts rely on the versions in `requirements.txt`.

These steps were only tested on Linux (WSL). The Windows and macOS commands
are the standard ones but untested, so say so if one fails.

## 2. Start Jupyter

With the environment active, in this folder:

```bash
jupyter lab
```

A browser tab should open. If none does (normal in WSL), copy the
`http://localhost:8888/lab?token=...` link from the terminal into your
browser. Open a notebook from the file list on the left and choose
*Run > Restart Kernel and Run All Cells*.

`jupyter` is installed by `requirements.txt`. If `jupyter lab` says
"command not found" (or "not recognized" on Windows), the environment is not
active (activate it again, see section 1) or the install was skipped (run
`pip install -r requirements.txt`). Check which `jupyter` you are running:
`where jupyter` on Windows, `which jupyter` on macOS, Linux and WSL. The
first path listed should contain `my-venv`.

Prefer VS Code? Open the folder, install the Python and Jupyter
extensions, open a notebook and choose the `my-venv` interpreter as the
kernel. Step 4 still runs from a terminal.

## Record your clips (do this before step 1)

Each person records **three clips**. The notebooks find clips by their file
name, so the names must follow this rule exactly:

| Kind of clip | File name | Example |
|---|---|---|
| Blinks | `blinks_<name>.mp4` | `blinks_alex.mp4` |
| Eyes move, head still | `eyes_only_<name>.mp4` | `eyes_only_alex.mp4` |
| Head moves, eyes fixed | `head_only_<name>.mp4` | `head_only_alex.mp4` |

`<name>` is whoever is in the clip. Use lower-case letters and digits only:
no spaces, no capitals. Use the same `<name>` for all three of your clips.

### Working in a group of two or three

- Everyone records their own three clips.
- Put **all** the group's clips into this folder on **one** laptop. Every
  notebook picks up every clip it finds, so nothing in the code needs
  changing when a clip is added.
- Before recording, agree on one recording script for the whole group (see
  the exercise below). If each person does something different, you cannot
  tell whether a difference in the results comes from the person or from
  the script.
- Before labelling (step 4), agree on what counts as "closed" and write it
  down. "Left" and "right" are not up for discussion: they always mean
  **left and right as seen in the video** (see section 4).
- In steps 5 and 6, the setting `TEST_CLIP` lets you train on some people
  and test on a person the model has never seen.

Working alone is fine too: every step runs with one person's three clips.

### How to record

Use the Windows Camera app, a phone, or any camera app, and save as `.mp4`.
On a Mac, QuickTime Player and Photo Booth save `.mov`, which must be
converted to `.mp4` (see the check below) because the notebooks look for
`.mp4` names. For all three clips:

- **Face and eyes fully in frame** for the whole clip, roughly centred, with
  the camera at about eye height and an arm's length away (a laptop webcam
  at normal sitting distance is fine). About 1280x720 at 30 frames per
  second is plenty.
- **Light on your face, not behind you.** A window behind you turns your
  face into a silhouette. Avoid strong shadows across the eyes.
- **Hair, hats and glasses off the eyes** if you can. Glasses reflections
  can hide the iris (untested; see the check below).
- **Landscape orientation** if you use a phone, and keep the phone still
  (stand or prop it up).
- **Not mirrored.** Some apps show a mirrored preview but save the file
  correctly. Check the *saved* file: hold a printed word up to the camera
  and make sure it reads the right way round when you play the video.
- Each clip about **30 seconds**. Start and end with a second of looking
  at the camera or screen centre.

The three clips are three different experiments. What changes between them
is the whole point of the workshop: keep to the **head** and **eyes**
columns of the table below, because the later steps depend on them.

**Exercise: write your own recording script.** Before you press record,
spend about ten minutes as a group and write down a short script for each
clip: what you will do, in what order, for how long, and what you will do
about mistakes (restart the clip, or carry on?). Then think about:

- How many looks of each direction do you need so that some can be held back
  for testing (step 6 splits by whole look, not by frame)?
- How many blinks do you need for precision and recall to mean anything? A
  score from five blinks can change a lot if one blink is missed.
- Should a person read out timed prompts, or a screen show them? What would
  that change, and could it introduce a pattern the model learns instead of
  the eyes (for example, evenly spaced blinks)?
- Would you want to log the true times as you record? How would you compare
  that log with what you label by hand in step 4?

The table is a **starting point** you can adopt or change.

| File name | Eyes | Head | Suggested script (about 30 s) |
|---|---|---|---|
| `eyes_only_<name>.mp4` | move | **still** | Keep your head still. Look at the **centre** of the screen, then **left**, back to centre, **right**, centre, **up**, centre, **down**, centre. Hold each look for about 2 seconds, and look clearly far in each direction (up and down are the hardest for the model, so look well up and well down). Do the whole round **three times** if it fits. Use the same targets each round, for example the middle of each screen edge. |
| `head_only_<name>.mp4` | **fixed on one point** | move | Pick one target (a sticker on the screen, or the camera lens) and keep your eyes on it for the entire clip. Slowly turn your head left, back to centre, right, centre, up, centre, down, centre, moving a moderate amount in each direction (enough that your nose visibly moves, small enough that you can still see the target without moving your eyes). Do the whole round about **twice**. |
| `blinks_<name>.mp4` | blink | **still** | Keep your head still and look at the centre of the screen. **Blink about 15 to 20 times**, with a gap of one to two seconds between blinks (not at regular, metronome intervals) and spread across the whole clip. Include **a few blinks while looking up and a few while looking down** (hold the look, blink, then return to centre). A couple of slow, deliberate blinks are useful too. |

Why these designs, in one line each:

- `eyes_only`: with the head still, any movement of the iris is eye
  movement. This is the cleanest data for gaze direction.
- `head_only`: your eyes do not change where they look, so the true gaze is
  "centre" the whole time even though the head moves. Anything the model
  calls "left" or "up" here is a mistake, caused by head movement. It tests
  the model without any hand labelling.
- `blinks`: lots of blinks in different gaze directions, because open-eye
  EAR changes with where you look, and that is what breaks a fixed
  threshold.

### Check each clip before moving on

After saving your clips under their final names, run
`step_01_face_mesh_check.ipynb`. It checks every clip in the folder and
prints a face-detection rate for each, which should be close to 100%. If
one is clearly lower, the face was lost for part of that clip: record it
again with better light or the face further inside the frame.

If a video will not open (some phones save HEVC, and Macs save `.mov`),
convert it to H.264 `.mp4` first, for example with
`ffmpeg -i input.mov -c:v libx264 -an eyes_only_alex.mp4`. This is untested.

### What results will look like

Expect noisy results. A few 30-second clips are a small teaching dataset,
not a benchmark, and another face, camera or light shifts every number.
Comparing results inside your group, and with other groups, is part of the
exercise. A model that does *worse* than the simple rule is a real result
too, as long as you can say why.

## 3. The steps, in order

`<name>` below stands for each person's name, so a group gets one file per
person wherever it appears.

| Step | File | Purpose | What you do | Needs | Makes |
|---|---|---|---|---|---|
| 1 | `step_01_face_mesh_check.ipynb` | Check that MediaPipe finds a face and draws the eye and iris landmarks on every clip | Run all cells | all your clips | nothing saved |
| 2 | `step_02_ear_blink_threshold.ipynb` | Measure the eye aspect ratio (EAR) per frame and detect blinks with one fixed threshold. This is the simple baseline, and its weaknesses motivate step 5 | Run all cells | `blinks_<name>.mp4` | `ear_blinks_<name>.csv` |
| 3 | `step_03_gaze_head_features.ipynb` | Turn each frame into numbers: iris offset (gaze) and nose offset, face position and roll (head). Saved so later steps do not rerun MediaPipe | Run all cells | all your clips | one `features_<clip>.csv` per clip |
| 4 | `step_04_label_intervals.py` | Make the ground truth: you watch the footage and mark when the eyes are closed or where they look. Models can only be scored against labels | Label footage by hand in a video window (see below) | the CSV files from steps 2 and 3 | `labels_blinks_<name>__<labeller>.csv`, `labels_eyes_only_<name>__<labeller>.csv` |
| 5 | `step_05_blink_classifier.ipynb` | Train a logistic regression blink classifier and compare it with the step 2 fixed threshold (precision, recall, F1) | Run all cells | `features_blinks_<name>.csv` and its label file | nothing saved |
| 6 | `step_06_gaze_classifier.ipynb` | Train a random forest to predict gaze direction and compare it with a naive threshold rule. Also tests whether head movement is mistaken for eye movement, using the `head_only` clips | Run all cells | `features_eyes_only_<name>.csv` and its label file, `features_head_only_<name>.csv` | nothing saved |

After step 6, pick one task from `STUDENT_TASKS.md`.

Each notebook starts with an "At a glance" table that repeats what it needs
and what it makes. Steps 1 to 3 run MediaPipe on every frame, so they take
longer the more clips there are (roughly 10 to 20 seconds per clip).

### Training on more than one person (steps 5 and 6)

Steps 5 and 6 load **every labelled clip** in the folder. The first code
cell of each has one setting that decides how the model is tested:

- `TEST_CLIP = None` (the default): every clip contributes to both training
  and testing. Step 5 trains on the first 70% of each clip and tests on the
  rest. Step 6 holds back 30% of all the looks. This works with a single
  person.
- `TEST_CLIP = "blinks_alex"` (step 5) or `TEST_CLIP = "eyes_only_alex"`
  (step 6): that whole clip is held back. The model is trained on the other
  people and tested on Alex, whom it has never seen. This needs at least two
  labelled clips.

Run each notebook once with `None`, then once per person as the held-out
clip, and compare. The held-out score is usually lower. It is also the more
honest answer to "would this work for a new user?".

## Optional extension: a live camera instead of a recorded clip

The guided steps read saved video files, so a live camera is an extension
for after step 6. No code is provided: this section describes how to
approach it. None of it has been tested here.

### Getting the camera to work

- **WSL has no webcam** (`/dev/video*` does not exist). Run Python
  natively on Windows, Linux or macOS instead. On macOS, the terminal
  app must be allowed to use the camera (System Settings > Privacy &
  Security > Camera).
- Steps 1 to 3 open the video with `cv2.VideoCapture(...)`, using a file
  name. Passing the number `0` instead opens the default camera. Live
  video has no end, so you also need a way to stop (for example press
  `q`).
- Keep the video **unmirrored**, as the clips are. A mirrored picture
  flips the sign of the left/right features.

### How to approach a model that works on live video

The main idea: **a model is not trained "on live video". It is trained on
saved, labelled data and then *used* on live video.** Training needs the
true answer for every frame, and a live stream has none. So the work
splits into two parts: getting labelled data to train on, and a program
that uses the trained model frame by frame.

**1. Get labelled training data. There are two ways, and you can mix
them.**

*Option A: use the recorded clips you already have.* This is the easier
route, and it needs no new recording. The clips and hand labels from steps
1 to 4 are exactly the kind of data a live model needs, so the model you
train in step 5 or 6 can be the one you use on the camera. To make it
carry over:

- Record the clips with the **same camera, seat and lighting** you will
  use live. A model trained on phone clips and used on a laptop webcam
  sees shifted numbers.
- Build the features the backward-looking way described in point 3 below,
  in training as well as live.
- More people and more clips in the training data make it more likely to
  work on whoever sits down in front of the camera (see `TEST_CLIP`).

*Option B: collect new data from the camera with prompts.* With a live
camera you can get labels without step 4's hand labelling, by telling the
person what to do and recording what you told them:

- Show a prompt on screen ("look LEFT", "blink now", or a dot to look at),
  compute the same features as step 3 for every frame, and save each row
  with the prompt that was showing as its label.
- People need time to react. Throw away roughly the first half second
  after each prompt changes, or the label says "left" while the eyes are
  still moving there.
- A prompt is what you *asked for*, not what happened. People blink, glance
  away or react late. Play some of it back and check before you trust it.
- Save everything to a CSV, as the steps do. Then you can retrain as often
  as you like without recording again.

**2. Train offline, exactly as in steps 5 and 6.** With option A this is
those notebooks as they are. With option B the saved CSV has the same
columns, so the same logic applies. Two rules carry over:

- Split by whole prompt (one "look" or one blink), never by frame, or hold
  out a whole recording session. Neighbouring frames are near-identical.
- Save the trained model, and anything fitted alongside it such as step
  5's scaler, to a file. `joblib` is already in `requirements.txt` for
  this. The live program loads it instead of training again.

**3. Use the model live, one frame at a time.** For each new frame: run
MediaPipe, compute the features, call the model's `predict`. Three things
are different from the notebooks:

- **The features must be computed the same way as in training**: same
  columns, same order, same scaling. A mismatch here gives confident
  nonsense, not an error message.
- **You cannot look into the future.** Step 5's `ear_min3` uses the frames
  just before *and after* each frame, and step 3 smooths with a centred
  window. Live, the later frames do not exist yet. Keep a short buffer of
  the last few frames and use only those, and train the model with
  features built the same backward-looking way.
- **Single-frame predictions flicker.** Smooth the output, for example by
  taking the most common prediction over the last 5 to 10 frames, or by
  requiring a direction to hold for some time before acting on it (dwell
  time). More smoothing means a steadier result and a slower reaction.
  That trade-off is the design decision for anything that controls a
  device.

**4. Calibrate for the person in front of the camera.** Open-eye EAR and
the resting iris position differ between people and sessions. A few seconds
of "look at the centre, eyes open" at the start gives a personal baseline
to subtract or divide by before the features reach the model. Use the same
correction on the training data.

**5. Test it honestly.** Record a separate prompted session, ideally on
another day or with another person, and score the live predictions against
the prompts the same way steps 5 and 6 do. Report the delay too (how long
after a look or a blink does the output change?) and how many frames per
second your laptop manages, since both matter for a live system and neither
shows up in an accuracy number.

Expect it to work worse than the notebooks suggest at first. The models in
steps 5 and 6 only ever saw your group's clips, and a different person,
light, camera or distance shifts the EAR baseline and the iris offsets.

## 4. Step 4 is a script, not a notebook

Step 4 opens a video window and you press keys to mark what you see. It is
run from the terminal, not from Jupyter.

First check that a window can open at all:

```bash
python setup_02_check_window.py
```

Then open `step_04_label_intervals.py` in a text editor and set three lines
near the top:

```python
MODE = "gaze"        # "blink" or "gaze"
CLIP_NAME = "alex"   # who is IN the clip (the <name> in the file name)
LABELLER = "sam"     # who is doing the labelling (you)
```

and run it:

```bash
python step_04_label_intervals.py
```

Keys: `,` and `.` step one frame back or forward, `SPACE` plays or pauses,
`u` undoes, `s` saves, `q` saves and quits. In gaze mode `1` to `5` set
centre, left, right, up, down. In blink mode `b` switches between open and
closed.

**Go to the last frame before you press `q`.** The file only covers the
frames up to where you quit. Step 5 treats every frame after that as
"open", and step 6 ignores them. If you press `q` earlier, a red warning
appears in the video window and you must press `q` a second time to quit
anyway. Any other key carries on.

**Your labels are not overwritten without asking.** If a label file for
the same clip and labeller already exists, the script asks in the terminal
whether to overwrite it (`y` or `n`) before the window opens. Answer `n`
and nothing is changed. To keep both sets of labels, use a different
`LABELLER` name.

What needs labelling:

- **Blink mode, every `blinks_<name>.mp4`.** It writes
  `labels_blinks_<name>__<labeller>.csv`. Decide what "closed" means
  before you start (for example: from the first frame the lids visibly
  cover the pupil until the first frame they uncover it) and apply that
  rule the same way every time.
- **Gaze mode, every `eyes_only_<name>.mp4`.** It writes
  `labels_eyes_only_<name>__<labeller>.csv`. **"Left" and "right" always
  mean left and right as seen in the video**: label "left" when the eyes
  move towards the left edge of the picture. The video is not mirrored, so
  this is the opposite of what the person did. If you looked to your own
  left while recording, it shows as a look to the right, and you label it
  "right". Label what you see, not what you remember doing. Give every
  separate look its own label, including the returns to centre.

  This convention is fixed because step 6's simple rule is written for
  it: a positive `gaze_x` means the iris sits towards the right of the
  picture, and the rule calls that "right". Labels made the other way
  round would silently turn that comparison upside down.

  One thing that looks like a contradiction but is not: the landmark
  names in the code (`R_EYE`, `L_EYE`, "right iris") use the **person's
  own** right and left, because that is how MediaPipe numbers a face. So
  the person's right eye is on the left of the picture. The names say
  *which eye* a point belongs to. The gaze labels say *which way the eyes
  are pointing in the picture*. They are two different things.
- Run step 4 **after** steps 2 and 3, because it reads the CSV files they
  make. Change `MODE` and `CLIP_NAME` and run it again for the next clip.
- In a group, share the labelling out. Labelling someone else's clip is a
  good idea: you are less tempted to label what you *meant* to do.
- If one clip ends up with two label files (two labellers), steps 5 and 6
  stop and ask you to choose one by setting `LABELLER` in their first code
  cell.

The `head_only` clips need no labelling. The eyes stay on one point for the
whole clip, so the true gaze label is "centre" in every frame.

## 5. Files in this folder

What you start with:

- `README.md`: this file.
- `requirements.txt`: the Python packages. Do not upgrade MediaPipe and do
  not add plain `opencv-python`.
- `setup_01_check_install.py`, `setup_02_check_window.py`: quick checks.
- `step_01` to `step_06`: the workshop steps (see section 3).
- `STUDENT_TASKS.md`: open-ended tasks for after step 6.

What you add and what the steps make (`<name>` = each person):

- `blinks_<name>.mp4`, `eyes_only_<name>.mp4`, `head_only_<name>.mp4`: your
  recordings.
- `ear_blinks_<name>.csv`: EAR per frame, from step 2.
- `features_<clip>.csv`: gaze and head features per frame, from step 3.
- `labels_<clip>__<labeller>.csv`: your hand labels, from step 4.

## 6. If something goes wrong

- **"No clips found"**: the file names do not follow the rule. Check the
  spelling, the underscores and the `.mp4` ending (see "Record your clips").
- **"No such file" for a `.csv`**: an earlier step was skipped. Check the
  "Needs" column above and run the step that makes that file.
- **Step 5 or 6 says "No labelled ... clips found"**: step 4 has not been
  done for that kind of clip, or `CLIP_NAME` did not match the clip.
- **Step 5 or 6 says a clip has several label files**: set `LABELLER` in the
  first code cell, or move the extra file out of the folder.
- **Step 5 stops because a part has no closed frames**: the blinks are all
  at one end of the clip, or labelling stopped early. Label to the end, or
  record a clip with the blinks spread out.
- **Step 6 stops with "Only N labelled segments"**: too few separate looks
  were labelled. Label each look as its own interval.
- **`NameError` in a notebook**: a cell was run out of order. Use
  *Restart Kernel and Run All Cells*.
- **No video window in step 4**: run `setup_02_check_window.py` first. The
  window may be behind other windows. In WSL it opens on the Windows
  desktop (this needs WSLg). On macOS, click the window once so it has
  keyboard focus before pressing keys.
- **`python` is not found or is the wrong one**: the environment is not
  active. Activate it (section 1) and try again.
- **You re-recorded a clip**: run steps 2 and 3 again, and label it again.
  The old CSV and label files describe the old recording.

## Privacy

Your clips and the CSV files made from them are recordings of faces and
eyes. Keep them on the group's own machines and do not put them in a public
repository. If you record anyone outside your group, ask them first.

## How the pieces fit together

A fair question before starting: **MediaPipe already finds the eyes, so why
train any models at all?**

Because MediaPipe answers a different question. It tells you *where the
points of the face are*. It does not tell you what they mean. For every
frame it returns 478 positions, and that is all. It does not say "this is
a blink" or "this person is looking left". Getting from positions to a
decision is the part you build, and it is what the workshop is about.

The whole workshop is one pipeline with four stages:

| Stage | What happens | Where | Who does the work |
|---|---|---|---|
| 1. Detect | A video frame goes in, 478 face points come out | Step 1 | MediaPipe (a ready-made neural network) |
| 2. Measure | The points are turned into a few meaningful numbers: how open the eye is (EAR), where the iris sits, how the head is turned | Steps 2 and 3 | You, with geometry |
| 3. Decide | The numbers are turned into an answer: open or closed, left / right / up / down / centre | Steps 2, 5 and 6 | A rule you write, or a model that learns the rule |
| 4. Use | The answer does a job: count blinks, steer a toy car, notice tiredness | `STUDENT_TASKS.md` | Your project |

Stage 3 is where the two machine learning models come in, and each is set
against a simple hand-written rule:

- **The simple way: a rule.** "Closed if EAR is below a cutoff" (step 2),
  "left if the iris offset is beyond a cutoff" (step 6). It takes minutes
  to write and it is easy to understand. It also breaks in predictable
  ways: looking down lowers EAR without a blink, and turning the head
  moves the iris without a look.
- **The learned way: a model.** Instead of you choosing the cutoff, the
  model is shown frames together with the right answer (your labels from
  step 4) and works out its own rule. It can combine several numbers at
  once, which a single cutoff cannot.

| | Logistic regression (step 5) | Random forest (step 6) |
|---|---|---|
| Question it answers | Is the eye open or closed? | Which of five directions is the gaze? |
| How it decides | Gives each number a weight, adds them up, and draws one boundary between "open" and "closed" | Builds many small decision trees, each asking yes/no questions about the numbers, and lets them vote |
| What it sees that the rule does not | How fast EAR is changing, and where the person is looking | The head position, so a head turn need not be mistaken for a look |
| Why this model | Two answers, and a simple model is enough to test the idea | Five answers, and the numbers interact: the same iris offset means something else when the head is turned |

Both were chosen because they train in seconds on a laptop, cope with very
little data, and can be inspected afterwards (which numbers did the model
rely on?).

**The model is not guaranteed to win.** Each notebook ends with a direct
comparison between the rule and the model, and with so little data the
rule sometimes comes out ahead. That comparison is the point. The habit to
take away is the order of work: measure, try the simple rule, try a model,
compare them fairly on data the model has not seen, and say honestly which
did better and how sure you are.

**How this connects to the open tasks.** Every task in `STUDENT_TASKS.md`
is this same pipeline with one part changed:

- Tasks 1 to 3 improve or test stage 3: more data, better labels, a better
  rule, a fairer comparison.
- Tasks 4 to 6 change stage 4, and feed the model different numbers or ask
  it a different question. A fatigue classifier is the same kind of model
  given blink rate over half a minute instead of single frames.
  Gaze-to-screen calibration predicts a position instead of a category.
- Deep learning (the "Going further" section of `STUDENT_TASKS.md`)
  replaces stages 2 and 3 together: a network looks at the picture and
  learns its own measurements. The rule and the two models from this
  workshop then become the baseline it has to beat.

One honest footnote: newer versions of MediaPipe can also output ready-made
scores such as "eye blink". Those come from a second model that somebody
else trained in exactly this way, on labelled examples. Building the stages
yourself is how you learn what such a score can and cannot be trusted for.
