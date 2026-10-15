# Outreach drafts

Nothing here has been sent. Each message goes out under the project owner's own account.

---

## A. Dataset-owner messages (Hugging Face → dataset page → "Community" → "New discussion")

Purpose: real-world confirmation without access to hardware. Keep each message short, specific,
and easy to answer with yes or no.

### A1. `khanhnd61/so101-multi-task-clean`

**Title:** Possible unpowered wrist_roll motor in most episodes?

> Hi! I've been building an open-source tool that checks robot-learning datasets for physical
> recording problems by comparing the leader arm's commands with the follower arm's measured
> positions ([demodyne](https://github.com/shoggi211/demodyne)).
>
> On this dataset it found that **`wrist_roll` on the follower barely moves in 34 of the 44
> episodes I analysed**, even though the leader commands about 37 units of motion (the follower
> covers about 0.2). That usually means the motor was unpowered, disconnected or not responding
> during recording.
>
> Could you tell me whether that matches what happened? Either answer helps me check whether the
> tool is right. Thanks for sharing the data!
>
> To reproduce: `pip install "git+https://github.com/shoggi211/demodyne"` then
> `demodyne audit khanhnd61/so101-multi-task-clean`

### A2. `lirislab/so100_demo`

**Title:** Actions appear ~2 frames out of step with observations

> Hi! I'm working on an open-source checker for robot-learning datasets
> ([demodyne](https://github.com/shoggi211/demodyne)) that measures timing between the leader
> commands (`action`) and the follower's measured positions (`observation.state`).
>
> In this dataset the follower appears to move **about 1.7 frames (≈56 ms) before** the command that
> should have caused it, consistently across joints. A real arm can't do that, so the most likely
> explanation is that the `action` column is shifted by about two frames relative to the
> observations, perhaps from the recording script or a conversion step.
>
> Does that ring a bell? If it's real, a policy trained on this data learns slightly mistimed
> actions. `demodyne realign lirislab/so100_demo -o ./fixed` writes a corrected copy, but I'd first
> like to know whether the diagnosis is right.

### A3. `blanchon/play_orange12`

**Title:** wrist_roll on the follower seems not to follow the leader

> Hi! An open-source dataset checker I'm developing
> ([demodyne](https://github.com/shoggi211/demodyne)) flagged that in this dataset the follower's
> **`wrist_roll` covers only about 3% of the range the leader commands** and stays off-target for
> most frames (episode 23 shows no movement at all). Common causes are a calibration difference
> between the leader and follower wrist, or a motor that isn't responding.
>
> Is that consistent with your setup? I'm trying to confirm whether the tool's findings match
> reality, so any answer is useful. Thanks!

---

## B. Community post (LeRobot Discord `#general`/`#datasets`, or discuss.huggingface.co)

**Title:** We audited 233 public SO-100/101 datasets for physical recording problems

> I built an open-source tool, **demodyne**, that compares what the leader arm commanded with what
> the follower arm did in LeRobot datasets. From the parquet data alone (no video, no robot) it
> measures each arm's reaction time and flags recordings that are physically impossible.
>
> Running it on a random sample of 233 community SO-100/101 datasets found:
> - **Reaction time varies 2.3× between arms** of the same design (54–123 ms, 5th–95th
>   percentile). The measurement repeats to within ~2 ms, so the spread is real.
> - **Action timing depends on the recording pipeline.** Datasets from the current `record` loop
>   are consistent; older and custom pipelines often log actions 1–2 frames late. 28% are out of
>   step with the current convention.
> - **8% have a serious hidden defect**: unpowered motors, leader/follower calibration
>   mismatches (usually `wrist_roll`), frozen sensor reads, swapped channels.
>
> In simulation, mixing data that is out of step by one frame or more made a trained controller
> worse than leaving the data out (up to +130% tracking error). Realigning the timing first
> restored full performance.
>
> Try it:
> ```
> pip install "git+https://github.com/shoggi211/demodyne"
> demodyne report <your/dataset>
> demodyne pool-check <ds_a> <ds_b>
> ```
> Code, method and all results: https://github.com/shoggi211/demodyne
>
> **Looking for help:** it hasn't been checked on physical hardware yet. If you have an SO-100 or
> SO-101 arm, recording a short session and posting `demodyne report` output in a GitHub issue
> would be very valuable, especially with a known setup change such as a longer USB cable, a
> different power supply or a different `--fps`.

---

## C. Potential-user interviews (LinkedIn, email or community DM)

**Who to ask:** people who train robot policies on teleoperation data, for example
- robot-learning engineers at humanoid and manipulation startups,
- PhD students in imitation-learning labs who maintain multi-robot datasets,
- teams running data-collection operations (teleop data vendors),
- maintainers of open datasets (LeRobot, Open X-Embodiment contributors).

Pick people from public talks, papers or posts about robot data. Ask about their work, and don't
pitch.

**Message template (≤ 80 words):**

> Hi {name}, I saw your {talk/paper/post} on {topic}. I'm researching data-quality problems in
> robot teleoperation datasets, such as timing mismatches between actions and observations,
> unpowered motors and calibration drift, and recently audited 233 public LeRobot datasets
> (write-up: {link}). Would you be open to a 20-minute call about how your team checks training
> data today? I'm not selling anything; I'm trying to learn whether this problem matters in practice.

**Questions for the call** (listen more than talk):
1. Walk me through the last time bad training data cost you time. What happened, and how did you find it?
2. What do you check before training on a new batch of demonstrations? Who does it, and how long does it take?
3. Do you combine data from different robots, sites or recording setups? What goes wrong?
4. Have you measured your robots' command-to-response latency? Does it differ between units?
5. If a tool flagged these issues automatically, where in your pipeline would it run? What would it need to do for you to rely on it?
6. Who else should I talk to?

**Signal to look for:** they describe a recent, specific, costly incident (go deeper), versus
"in theory that matters" (weak). Aim for 8–10 conversations before deciding to build more.
