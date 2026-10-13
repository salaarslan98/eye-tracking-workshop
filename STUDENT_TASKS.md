# Open-ended tasks

You've now built the guided part: face/eye tracking (`step_01`-`step_03`), hand
labelling (`step_04`), and two small classifiers that each try to beat a
simple rule (`step_05` blinks, `step_06` gaze direction). Everything below is
optional and open-ended — pick **one**, not all of them (or bring your own
idea, see task 7). None of these have a
single correct answer; the point is to practice the same loop you just saw:
have a simple rule, question it, measure instead of assuming, say honestly
what you found.

**You are not expected to finish in the workshop.** There is not enough
time for that, and it is not the goal. The goal is to get a real start:
a clear question, a first version that runs, and one honest result. You
can carry the project on in your own time and put it in your own GitHub
repository as something to show and talk about (see "After the workshop"
at the end of this file).

## Two kinds of task

The tasks fall into two groups. Choose the kind of work you would enjoy
first, then a task within it.

| | Tasks 1 to 3: improve the method | Tasks 4 to 6: build an application |
|---|---|---|
| **The question** | "Is what we built actually right, and can it be made better?" | "What could this be used for?" |
| **Kind of work** | Data science and research: more data, better labels, a better rule, a fairer test | A project: a new use built on the features and models you already have |
| **You end with** | A number you can defend, and an honest account of how far to trust it | Something that does a job, plus an honest account of where it fails |
| **Tasks** | 1a more data, 1b labeller agreement, 2 dip-based blink detector, 3 vertical-gaze offset | 4 fatigue classifier, 5 gaze-to-screen calibration, 6 live demo |
| **Suits you if** | You like asking "but is that really true?" | You like asking "what if it could do this?" |

Neither kind is easier or more valuable. An application built on a method
nobody checked is fragile, and a well-checked method nobody uses helps no
one. If neither list has what you want to do, see task 7: your own idea.

Whichever you pick, "How to structure your project" at the end of this
file applies to all of them. Read it before you start.

## How each task is written

Each task below gives you: why it's interesting, a starting point (sometimes
a short code sketch, not a full script — you write the rest), what to
measure, and how hard/risky it is. Use the existing step notebooks as your
template for structure (load CSV → compute → print a summary → save output).

**Live camera, if you want it.** Every task can be done on recorded clips
alone. Each one also has a short "With a live camera" note: an idea for
how a live camera could fit in, not an instruction. Whether you use one,
and how, is your decision. Read "Optional extension: a live camera" in
`README.md` first: it explains how training on saved data and running on
live video fit together, and what changes when you cannot look at future
frames. None of the live ideas have been tested here, so treat getting the
camera to open in Python as the first thing to check.

---

# Part A: improve the method (tasks 1 to 3)

These tasks take what steps 1 to 6 built and ask whether it holds up. The
result is usually a comparison: before and after, one method against
another, one labeller against another.

## 1a. Make the blink/gaze evaluation trustworthy with more data (easiest, do this first if unsure)

**Why:** `step_05` and `step_06` both end on the same caveat: a few short
clips, one labeller each. That's a real weakness, not just a disclaimer.
With so little data we can't tell if a model generalizes or just fit the
clips it saw.

**Starting point:** you already have the tool for this. Both notebooks load
every labelled clip in the folder, and `TEST_CLIP` in their first code cell
holds one whole clip back for testing. So:

1. Add data. Record a second set of clips of yourself on another day, in
   different light, or with a different camera (use a new `<name>`, for
   example `alex2`). Or swap clips with another group. Run steps 2 and 3,
   then label the new clips with `step_04_label_intervals.py`.
2. Run `step_05`/`step_06` with each clip in turn as `TEST_CLIP` and write
   the scores into a small table (one row per held-out clip).

**Measure:** precision/recall/F1 (blinks) or the confusion matrix (gaze) on
each held-out clip. Does the model hold up on a clip it never trained on,
or only on clips it has seen part of?

**Thinking questions:**
- Does adding a third or fourth clip to the training data help the
  held-out score, or does it level off? What would you need to record to
  find out?
- If performance drops a lot on one clip, is that the model's fault, or
  could it be that its labeller judged borderline frames differently? (See
  task 1b: that's not a rhetorical question, it's measurable.)
- How would you label the *same* clip twice, a week apart, to check how
  consistent your own labelling is?

**With a live camera (optional):** a camera is a fast way to get the extra
data this task needs. A prompted recording ("look LEFT", "blink now") gives
you labelled frames without hand labelling. You could treat a live session
as the held-out test: train on the recorded clips, test on what the camera
sees today. Is a prompt as trustworthy a label as one you made by hand?

**Difficulty:** low. Mostly patience with the labelling tool.

---

## 1b. Measure how much labellers agree with each other

**Why:** people do not agree on exactly where a blink starts and ends. One
person's definition of "closed" might be *fully shut, open again at the
first hint of opening* — strict at the boundary. A fixed EAR-threshold rule has its own,
different implicit boundary (wherever the EAR curve happens to cross the
cutoff number), and someone else watching the same footage might draw the
line differently again. Three different "closed" definitions, same clip.
This is a real, unsolved problem in any labelled dataset — it's usually
called inter-rater (or inter-annotator) agreement, and it's worth measuring
rather than assuming it's fine. It may also explain some of what you saw in
`step_05`. Compare the "predicted" and "true" closed-frame counts it
prints: if the fixed rule predicts many more "closed" frames than the hand
labels contain, a wider implicit boundary could be part of that gap,
separate from whether the rule is otherwise any good.

**Starting point:** before anyone looks at the footage, each person writes
down their own one-sentence rule for when an eye counts as closed. Then at
least two people independently run `step_04_label_intervals.py` on the
*same* clip with their own `LABELLER` name (MODE can be `"blink"` or
`"gaze"` — both are worth trying). You'll end up with two CSVs like
`labels_blinks_alex__sam.csv` and `labels_blinks_alex__kim.csv` (Sam's and
Kim's labelling of Alex's clip).
To compare them frame by frame:
```python
import pandas as pd, numpy as np
from sklearn.metrics import cohen_kappa_score

def stamp(intervals, n_frames):
    labels = np.full(n_frames, None, dtype=object)
    for _, row in intervals.iterrows():
        labels[int(row.start_frame):int(row.end_frame) + 1] = row.label
    return labels

n_frames = len(pd.read_csv("ear_blinks_alex.csv"))  # or features_eyes_only_<name>.csv for gaze
a = stamp(pd.read_csv("labels_blinks_alex__sam.csv"), n_frames)
b = stamp(pd.read_csv("labels_blinks_alex__kim.csv"), n_frames)

agreement = (a == b).mean()
kappa = cohen_kappa_score(a, b)
print(f"Raw frame agreement: {agreement:.1%}  |  Cohen's kappa: {kappa:.3f}")
```
Raw agreement alone can look deceptively high (most frames in a clip are
"open"/"centre", and both labellers will trivially agree on those). Cohen's
kappa corrects for that by accounting for how much agreement you'd expect
by chance — it's the more honest number here.

**Measure, your own code from here:** for the blink case specifically, pull
out just the "closed" intervals from each labeller and, for ones that
clearly overlap, compare their start and end times rather than treating
the whole blink as one unit — that's where the real disagreement should
concentrate, per the motivating example above. Report it in frames or
milliseconds, not just "agree/disagree."

**Thinking questions:**
- If two labellers agree a blink happened but disagree on exactly where it
  starts and ends by a few frames, is that a small disagreement or a big
  one? Does the answer depend on whether you're *counting* blinks or
  *timing* them (e.g. for blink-duration-based fatigue detection)?
- Of the three possible "closed" definitions here (yours, a labmate's, the
  threshold rule's implicit one), is there a "correct" one, or does the
  project just need to pick one and document it consistently? What would
  change if you picked the strictest vs. the most lenient?
- If `step_05`'s model recall problem turns out to be partly a boundary
  mismatch rather than the model actually missing distinct blinks, does
  that change what "fixing" the model should even mean?

**With a live camera (optional):** prompts give you a third "labeller" to
compare against the human ones: the times at which the person was *asked*
to blink or look. How far do the prompt times sit from where each human
drew the boundaries? That gap is partly reaction time, which you can
measure.

**Difficulty:** low-medium, and a good one to do as an actual group —
needs at least two people.

---

## 2. Build a dip-based blink detector

**Why:** you already know *why* a fixed EAR threshold fails (gaze-dependent
baseline). A smarter rule-based detector — no ML — looks for local dips in
the EAR signal instead of a fixed cutoff, and should survive the "looking
down" failure case better. This also gives you a second, independent
baseline to compare the `step_05` classifier against.

**Starting point:**
```python
from scipy.signal import find_peaks
# find_peaks looks for PEAKS, so feed it the inverted signal to find dips
dips, props = find_peaks(-ear, prominence=0.03, width=(2, 15))
```
`prominence` controls how deep a dip has to be relative to its surroundings
(not an absolute threshold — that's the whole point). `width` keeps a
single noisy frame from counting as a blink. Both numbers need tuning by
eye against the EAR plot.

**Measure:** precision/recall/F1 against your hand labels from `step_04`,
same as `step_05` already does for the fixed-threshold rule — add this as a
third method in the same comparison.

**Thinking questions:**
- Does this beat the fixed threshold on the "looking down" portions of a
  clip specifically? (Split your evaluation by gaze direction to check.)
- Does it beat the ML classifier from `step_05`, or only the simple rule?
  What does that tell you about when you need ML vs. a better rule?

**With a live camera (optional):** a dip can only be recognised once the
eye has opened again, so a live dip detector always answers late. How late?
`find_peaks` works on a finished signal: think about how you would run it
on a short buffer of the most recent frames instead, and what the smallest
workable buffer is. A blink counter on screen is a simple way to see it
working.

**Difficulty:** low-medium. `requirements.txt` already includes scipy.

---

## 3. Fix the vertical-gaze offset

**Why:** `step_06`'s own baseline rule assumes "centre" means `gaze_y ≈ 0`,
but resting `gaze_y` is usually not 0. Look at the middle panel of your
`step_03` plot while you look at the centre: does it sit a little above or
below zero? If so, that is a per-clip (maybe per-person) offset, not noise.
A model or rule that corrects for this baseline before thresholding might
read vertical gaze much better than what you saw in `step_06`.

**Starting point:** in a quiet "centre" stretch at the start of a clip
(first 1-2 seconds, before anything else happens), compute the median
`gaze_y`. Subtract that from every frame's `gaze_y` before applying any
threshold or feeding it to a model.

**Measure:** rerun `step_06`'s baseline rule and model with and without this
correction. Does the up/down row of the confusion matrix improve?

**Thinking questions:**
- Is this offset the same across your three clips, or does it differ per
  clip? Is yours similar to the others' in your group? What would that imply about using it on a *new* person
  the model has never seen?
- Could face geometry (not gaze) explain part of this offset, e.g. how the
  iris sits relative to the eyelid at rest? How would you check that
  without re-recording?

**With a live camera (optional):** this is the same idea as a calibration
step at the start of a live session: ask the person to look at the centre
for a few seconds, take the median, subtract it from then on. You could
test whether a baseline measured at the start still holds five minutes
later, or after the person shifts in their seat.

**Difficulty:** low. Good first task if you want something small but real.

---

# Part B: build an application (tasks 4 to 6)

These tasks use the features and models as building blocks for something
new. The measuring does not go away: each one still asks you to say how
well it works and where it breaks.

One idea that motivated this workshop is **gaze control** as assistive
technology: letting someone who cannot use their hands, for example a
child with limited mobility, steer a small motorised toy vehicle by
looking. The view is divided into zones (forward, left, right, stop), and
looking at a zone for long enough gives that command. Tasks 5 and 6 are
steps towards it, and it is called "the gaze-control idea" below.

## 4. Fatigue classifier

**Why:** a system that watches the eyes could do more than follow where
someone is looking. It could also notice when they are tired or losing
attention, which matters for anything from driving to a long session with
an assistive device.

**Starting point:** using 30-second sliding windows over a clip, compute:
blink rate, mean blink duration, fraction of time EAR is below some
"mostly closed" level, and the spread (std) of `gaze_x`/`gaze_y`. You
decide which clip stands in for "bored" and which for "alert" (for
example, one recorded after a long, dull task and one recorded when
fresh). The clips carry no fatigue label, so compare your results with
chance, and do not describe them as a validated fatigue measure.

**Measure:** a confusion matrix against chance level, evaluated on a
*different* recording session than it was tuned on.

**Thinking questions:**
- Nothing here was labelled by a tired/not-tired ground truth. What
  exactly would you be measuring: fatigue, boredom, or just "sitting still
  for a while"? How would you word your conclusion so you're not
  overclaiming?
- What's the ethical line between "flag low attention" and "surveil
  someone's tiredness"? Who should see this output, and who shouldn't?

**With a live camera (optional):** 30-second clips are too short for this
task, and a camera removes that limit. You could log blink rate and the
other window features continuously over a long, real activity (reading,
a lecture, a game) and plot how they drift. Decide beforehand what you
would record, for how long, and who gets to see it.

**Difficulty:** medium. The hardest part is being honest about what the
label actually means.

---

## 5. Gaze-to-screen calibration and regression

**Why:** this is what the gaze-control idea needs first: mapping eye
features to *where on a screen* someone is looking, not just a direction.

**Starting point:** you'll need a way to show dots at known screen
positions and log which face features were measured at each dot — this
needs a camera route your laptop can actually support (a live webcam feed
if your OS allows Python to access it, or a recorded-video approach like
the rest of this workshop). Ask before assuming your setup works. Once you
have (features, screen x, screen y) pairs, try: a centre-of-screen
baseline, linear regression, then (if time) a random forest.

**Measure:** error in pixels and in degrees of visual angle (you'll need
your screen size and distance from it), **always testing on dot positions
never used in training** — group your train/test split by dot, not by
frame, or neighbouring frames of the same dot will make your score look
better than it is.

**Thinking questions:**
- How much worse is accuracy at the screen edges vs. the centre? Why might
  that be, geometrically?
- If you moved your head slightly after calibrating, how much would
  accuracy drop? What does that tell you about how practical this is for
  a real assistive device?

**With a live camera (optional):** this task fits a live camera best,
because the program that shows the dot can save the features at the same
moment, so every frame gets its true screen position for free. Leave out
the frames just after the dot moves, while the eyes are still travelling.
Without a camera, the alternative is to record a clip while a dot sequence
plays, and line the two up afterwards by time.

**Difficulty:** high. Needs a working camera capture route — check this
works before committing time to it.

---

## 6. Live demo: gaze dot or heatmap

**Why:** turns the features you've already built into something visible
and interactive, and is a nice capstone if your laptop's camera works in
Python.

**Starting point:** reuse `step_01`'s face mesh loop, but instead of
drawing landmarks, map `gaze_x`/`gaze_y` (or a trained model's output, if
you did task 5) to a point on a blank image and draw it there each frame.
A heatmap is the same idea accumulated over time (`cv2.GaussianBlur` on an
accumulator array works well).

**Measure:** this one's more about seeing it work than a formal metric —
but you could report how jittery the dot is when you hold your gaze still
(precision), separately from whether it's roughly where you're actually
looking (accuracy).

**Thinking questions:**
- Does the dot feel "sticky" or does it jitter if you don't smooth the
  signal? How much smoothing is too much (adds lag)?
- What happens to the demo if someone else — different face shape, maybe
  glasses — tries it? Would you expect it to transfer, and does it?

**With a live camera (optional):** this task is the live one, but there
is still a choice to make: draw the raw features, or the prediction of a
model trained on your recorded clips (step 6), or of one trained on
prompted live data. If you add command zones for the gaze-control idea
(look left for long enough = "turn left"), the dwell time is yours to tune:
too short and it fires by accident, too long and it feels broken.

**Difficulty:** medium, mostly straightforward if your camera route works.

---

## 7. Your own idea

**Why:** the six tasks above are suggestions, not the limit. If you can
think of a use for blink and gaze signals that interests you more, do that
instead. Students from medical engineering and from IT tend to see
different uses, and a mixed group is a good place to find one.

**Starting point:** you have two signals that work reasonably well (is the
eye open or closed, and roughly which of five directions it is looking)
and several that are weak (exact gaze position, vertical gaze, anything
when the head moves a lot). A good idea is built on the first kind. Some
directions, only to start you thinking:

- Communication: blink patterns or looks as yes/no answers or a small menu.
- Control: looking at a zone for long enough triggers an action (the
  gaze-control idea described at the start of Part B).
- Monitoring: how blink rate or gaze changes during reading, screen work or
  a game.
- Accessibility: something a person with limited hand movement could use.
- Measurement: a question about eyes that you are simply curious about.

**Before you build anything**, write half a page that answers the
questions in "How to structure your project" below and show it to the
instructor. It saves you from an idea that needs a signal the webcam
cannot give.

**Difficulty:** depends on your idea. Keep the first version small enough
to get running in the workshop. The rest can follow afterwards.

---

# How to structure your project

These points apply to every task above, method or application, recorded
clips or live camera. Work through them in order and write your answers
down. Short answers are fine.

**1. Say what you are trying to find out or build, in one sentence.**
"We want to know whether..." or "We want to build something that...". If
it takes a paragraph, the idea is still too big.

**2. Say who it is for and why it matters.** Someone steering a device by
gaze, a researcher who needs trustworthy labels, yourself out of curiosity. The
answer decides which mistakes are serious. A missed blink and a false
blink are not equally bad for every use.

**3. Decide what "it works" means before you look at any results.** Which
number, compared with what? Precision and recall, a confusion matrix, an
error in pixels, a delay in seconds. Choose it first, or you will end up
choosing whichever number looks best.

**4. Name the simple thing you will compare against.** A fixed threshold,
"always guess centre", chance level, or the result from step 5 or 6. A
result means little on its own. "Our model gets 0.8" says nothing until
you know the simple rule gets 0.75.

**5. Plan the data.** What do you need to record, of whom, how much, and
how will it get its labels (by hand, by prompt, by design as with
`head_only`)? Which part is kept back for testing, and is it truly
separate (another clip, another person, another day) or just other frames
of the same recording?

**6. Start with the smallest version that runs from start to finish.**
This is the part to aim for inside the workshop. One
person, one clip, the simplest method, a printed number at the end. Only
then improve one thing at a time. A rough version that runs teaches you
more than a clever version that is half built.

**7. Change one thing at a time and write down what happened.** Keep a
short log: what you changed, what you expected, what you got. The entries
where you were wrong are the valuable ones.

**8. Look for where it fails.** Looking down, glasses, a turned head, a
new person, poor light. Go and find the cases that break it, and show one.
A project that reports its failures is more convincing, not less.

**9. Say how sure you are.** How many blinks, looks or people is the result
based on? Would it survive a different split or a different labeller? If
a result has more than one possible explanation, name them.

**10. Think about the person in front of the camera.** Who is recorded,
did they agree, who sees the output, and could it be used against them?
For anything that monitors a person (attention, tiredness), this is part
of the design, not an afterthought.

**At the end of the workshop:** be ready to show, in about five minutes,
where you got to. That means the one-sentence goal, what runs so far, any
result you have with its number, and what you would do next. An unfinished
project with a clear plan is a perfectly good thing to present.

---

# After the workshop: make it your own project

If you want to keep going, this makes a good portfolio project. It is
small, it uses real data you collected yourself, and it has an honest
evaluation, which is what people reading a repository look for.

**Putting it on your own GitHub:**

- Fork the workshop repository (the *Fork* button on its GitHub page), or
  start a new repository of your own and copy in the files you use. Say in
  your README that it started from this workshop.
- **Never upload the recordings.** The `.gitignore` in this folder already
  keeps video files, CSV files and images out of git, so keep that file.
  A face is personal data, yours and your group's, and once it is public
  you cannot take it back.
- **Clear notebook outputs before every commit** (*Edit > Clear Outputs of
  All Cells*). Step 1 draws frames of a face straight into the notebook,
  and `.gitignore` cannot remove what is saved inside a notebook file.
- If you want to show a result plot (an EAR curve or a confusion matrix,
  not a face), `.gitignore` blocks `.png` files by default. Add the one
  file on purpose with `git add -f my_plot.png`, after checking that
  nobody can be recognised in it.
- If clips of other people in your group were part of your results, ask
  them before you publish anything derived from their data.

**What to write in your README**, so that a stranger understands it in two
minutes. It is the "How to structure your project" list, written up:

1. What you set out to find out or build, in one sentence, and why.
2. What data you used: how many people and clips, how it was recorded and
   labelled. No names needed.
3. What you compared against what, and how you kept the test data separate.
4. The result, with numbers, and a plot if it helps.
5. Where it fails and how sure you are. This is the part that shows you
   understood what you built.
6. What you would do next.
7. How to run it, so that someone can record their own clips and try.

**Good ways to keep going on your own:**

- Record more data: more sessions, different light, and other people who
  agree to it.
- Do a second task from this file on top of your first. A method task and
  an application task go well together.
- Finish the parts you had to skip: the baseline, the held-out test, the
  failure cases.
- If you have access to a GPU and want to try deep learning, see "Going
  further: computer vision models" below.

A small project that is finished, measured and honestly described is worth
more to show than a large one that only half runs.

---

# Going further: computer vision models (optional, own time)

This is not part of the workshop tasks. It is a map of what to look up if
you carry the project on and want to try deep learning. Nothing here has
been tested in this workshop.

## What the workshop does, and what the alternative is

You have already used one computer vision model: MediaPipe Face Mesh is a
neural network that turns a picture into 478 points. Everything after it
is *classical* machine learning. We turn the points into a few hand-made
numbers (EAR, iris offset, nose offset) and give those to a small model.

The alternative is to let a neural network look at the **pixels** and
learn its own features:

| | Workshop approach | Computer vision model |
|---|---|---|
| Input to your model | A handful of numbers per frame | The image itself, usually a crop of the eyes or face |
| Who designs the features | You do (EAR, iris offset) | The network learns them |
| Data needed | A few short clips are enough to start | Thousands of labelled images, or a model someone already trained |
| Hardware | Any laptop | A GPU for training. Running a small trained model is often fine without one |
| Can you explain a prediction | Fairly well | Much harder |

Neither is "better". A small model on good features often beats a large
network when data is scarce, which is the situation you are in.

## Things to search for

- **Getting more from MediaPipe without a GPU.** The newer MediaPipe
  "Face Landmarker" can also output *blendshapes*: scores such as
  `eyeBlinkLeft` or `eyeLookUpRight`, made by a second small network. They
  make a good extra baseline to compare against your EAR and iris
  features, and they run on a laptop.
- **A network on eye crops.** Use the eye landmarks to cut out a small
  image of each eye, then train a convolutional neural network (CNN) to
  say open/closed or left/right/up/down. Search for "eye state
  classification CNN".
- **Transfer learning.** You do not have enough data to train a network
  from nothing. Take one already trained on millions of images (search for
  "MobileNet" or "ResNet" with "fine-tuning" in PyTorch or Keras) and
  retrain only its last part on your eye crops. This is the realistic way
  to use deep learning with little data.
- **Appearance-based gaze estimation.** This is the research name for
  predicting gaze direction straight from the image. Datasets and models
  to look up: MPIIGaze, GazeCapture (and its model iTracker), ETH-XGaze,
  Gaze360, L2CS-Net. Some come with trained models you can run.
- **Models over time.** A blink is a shape in time, not a single frame. A
  small 1D CNN or an LSTM over a window of EAR values is a natural step up
  from step 5 and is light enough to train without a GPU.
- **Public datasets for blinks and drowsiness.** Look up the MRL Eye
  Dataset and Closed Eyes in the Wild (eye open/closed), and NTHU-DDD or
  UTA-RLDD (drowsiness). These could give task 4 the labelled data that
  your own clips lack.

## Before you start

- **Read the licence of any dataset or model.** Many gaze and face
  datasets are for research use only and need a sign-up. Say in your
  README what you used.
- **Getting a GPU.** A gaming laptop, a university cluster, or a free
  cloud notebook such as Google Colab. **Think before uploading face
  recordings to a cloud service**: it leaves your machine, and for other
  people's faces you need their agreement. Uploading eye crops or features
  instead of full videos reduces what you share, but does not remove the
  question.
- **Use a separate Python environment.** Deep learning libraries (PyTorch,
  TensorFlow) are large and may want different package versions. Do not
  install them into the workshop environment, and do not upgrade its
  MediaPipe.
- **The evaluation rules do not change.** Split by clip or by person, not
  by frame. With image models, leaking the same person into training and
  test inflates the score even more, because a network can simply
  memorise a face.
- **Always compare against what you built here.** The fixed threshold and
  the small models from steps 5 and 6 are your baselines. "A neural
  network got 0.9" means little until you know the 20-line rule got 0.85.
  If the big model does not win, that is a result worth reporting.

---

## General reminders for whichever task you pick

- Save your outputs (CSVs, plots, printed summaries) like the existing
  steps do, so you can show your work, not just your conclusion.
- If a result could have more than one explanation, say so — don't round
  "consistent with X" up to "proves X."
- Face and eye recordings are sensitive. If you record anyone other than
  yourself, get their consent first and don't share the clips or feature
  files outside the group.
