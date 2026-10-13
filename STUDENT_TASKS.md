# Open-ended tasks

You've now built the guided part: face/eye tracking (`step_01`-`step_03`), hand
labelling (`step_04`), and two small classifiers that each try to beat a
simple rule (`step_05` blinks, `step_06` gaze direction). Everything below is
optional and open-ended — pick **one**, not all of them. None of these have a
single correct answer; the point is to practice the same loop you just saw:
have a simple rule, question it, measure instead of assuming, say honestly
what you found.

Each task below gives you: why it's interesting, a starting point (sometimes
a short code sketch, not a full script — you write the rest), what to
measure, and how hard/risky it is. Use the existing step notebooks as your
template for structure (load CSV → compute → print a summary → save output).

---

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

**Difficulty:** low. Mostly patience with the labelling tool.

---

## 1b. Measure how much labellers agree with each other

**Why:** this came out of a real observation while building this workshop:
one definition of "closed" was *fully shut, open again at the first hint of
opening* — strict at the boundary. A fixed EAR-threshold rule has its own,
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

**Difficulty:** low-medium, and a good one to do as an actual group —
needs at least two people.

---

## 2. Finish the dip-based blink detector (left for you on purpose)

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

**Difficulty:** low. Good first task if you want something small but real.

---

## 4. Fatigue classifier (step 7 from the original plan)

**Why:** the longer-term motivation for this whole workshop is assistive
tech — the project notes mention noticing when someone is tired or losing
attention, not just where they're looking.

**Starting point:** using 30-second sliding windows over a clip, compute:
blink rate, mean blink duration, fraction of time EAR is below some
"mostly closed" level, and the spread (std) of `gaze_x`/`gaze_y`. You'll
need to decide which clip stands in for a boring task and which stands in
for an alert one (ask about it — the clips don't have a built-in fatigue
label, you'll need a sensible standby: compare against chance, don't claim
a validated fatigue measure).

**Measure:** a confusion matrix against chance level, evaluated on a
*different* recording session than it was tuned on.

**Thinking questions:**
- Nothing here was labelled by a tired/not-tired ground truth. What
  exactly would you be measuring: fatigue, boredom, or just "sitting still
  for a while"? How would you word your conclusion so you're not
  overclaiming?
- What's the ethical line between "flag low attention" and "surveil
  someone's tiredness"? Who should see this output, and who shouldn't?

**Difficulty:** medium. The hardest part is being honest about what the
label actually means.

---

## 5. Gaze-to-screen calibration and regression (the original "step 4/5" plan)

**Why:** this is the real prerequisite for the vehicle-control extension —
mapping eye features to *where on a screen* someone is looking, not just a
direction.

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

**Difficulty:** medium, mostly straightforward if your camera route works.

---

## General reminders for whichever task you pick

- Save your outputs (CSVs, plots, printed summaries) like the existing
  steps do, so you can show your work, not just your conclusion.
- If a result could have more than one explanation, say so — don't round
  "consistent with X" up to "proves X."
- Face and eye recordings are sensitive. If you record anyone other than
  yourself, get their consent first and don't share the clips or feature
  files outside the group.
