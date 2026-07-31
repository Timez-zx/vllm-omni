# What we added to vllm-omni

Goal: let one person keep a camera and microphone on and **talk to the model for a long time**,
with every reply arriving quickly.

The original could manage a few turns. It broke down over longer conversations. Below, one section
per change, in the order it was added, each written as: **scenario → what used to happen → what
happens now**.

The code is in this repository, branch `live-agent`. Every new capability is **off by default** — an
unconfigured deployment behaves exactly like the original.

---

## What it can do now

| capability | in one line |
|---|---|
| no repeated work in a long chat | at turn 20 it processes only the new pictures, not all 19 earlier turns again |
| pictures shrunk automatically | the phone sends 1280×720; the server reduces it to 640×352 before the model sees it |
| controllable frame pacing | a still scene no longer leaves the model blind to *now*; a busy scene no longer floods the prompt |
| the conversation can run forever | near the internal limit it swaps the underlying request, invisibly to the user |
| warning before the limit | every turn prints how far from the limit it is — previously invisible |
| failures announce themselves | three kinds of silent freeze now report; the log used to say nothing at all |

---

## 1. The longer you talk, the slower it gets — because earlier turns were redone

*2026-07-31 00:07 – 01:24 · `e3ebb356` one request per conversation · `a53a0834` drain the frame buffer · `fa52c693` refuse overlapping turns*

**Scenario.** A user holds up a camera and has 20 turns of conversation.

**What used to happen.** Every question was treated as a **brand-new job**: the pictures and words
from the previous 19 turns had to be **walked through again** before the 20th could be answered. The
longer the chat, the slower each reply.

**What happens now.** The whole conversation is **one job** from start to finish, and each turn only
appends **the new pictures and the new sentence**. Everything earlier stays in GPU memory and is not
recomputed.

**Effect.** The speaking stage no longer slows down just because the chat is long. What used to be
the worst case — tens of thousands of words of accumulated context — is now *faster* than throwing
the history away: **keeping the history costs less than discarding it.**

---

## 2. The pictures arriving from the phone were too big

*2026-07-31 00:01 / 00:37 · `d1abbf8f` shrink on arrival · `6ec99006` move it off the audio path*

**Scenario.** The client camera is 1280×720 and every frame was passed through at that size.

**What used to happen.** The model processed each frame at full size. One 720p frame becomes 880
"pieces" to compute, and that count is what determines the time spent.

**What happens now.** A frame is **shrunk as soon as it arrives** (e.g. to 640×352, which is 220
pieces) and everything downstream uses the smaller version. Aspect ratio is preserved; it only ever
shrinks, never enlarges.

**One trap that had to be avoided.** The shrinking must happen **first** — the check for "does this
frame look like the last one", the caches, and what finally reaches the model must all see **the same
version** of the picture. Otherwise the check compares big images while the model reads small ones,
and they disagree.

**And a second, performance trap.** The shrinking was originally done inline, which meant it occupied
**the same path that pushes audio out to the user** — so time spent resizing was added directly to
somebody's wait for a reply. It was moved off to the side.

---

## 3. A still scene went blind; a busy scene flooded the prompt

*2026-07-31 00:04 · `f4837245`*

**Scenario A.** The user points at a blank wall, says nothing for half a minute, then asks something.
**Scenario B.** The user is walking, so every frame differs from the last.

**What used to happen.** The system decided what to keep purely by "does this frame look like the
previous one". So:

- with the blank wall it might keep **nothing for tens of seconds**, leaving the model blind to *now*
- while walking it might keep **a dozen frames at once**, flooding the prompt and slowing the reply

**What happens now.** The gap between two kept frames has both a **floor and a ceiling**:

- floor: no matter how much changes, it cannot keep frames back to back
- ceiling: no matter how still it is, once the gap is reached one frame must be kept

The "do they look alike" logic is **completely unchanged** — the bounds only limit how often it is
allowed to answer. Counted in frames rather than seconds, so behaviour does not shift when the client
changes frame rate.

---

## 4. At some turn, the conversation just stopped

*2026-07-31 02:42 · `b866eb15`*

**Scenario.** At turn 27, suddenly: the model is still **typing** (text keeps coming) but **says
nothing at all**. Two minutes later, **not a single error in the log**.

**Why.** This was the hardest one to find. The part that produces speech has a capacity limit, and on
crossing it either the whole process dies, or the request is **quietly skipped on every turn** with
nobody reporting anything.

**The key point: what fills up is not what you would assume.**

> That limit does not hold "chat history". It holds **the speaking stage's own ledger** — both what
> each turn adds *and* **the voice data it generated itself**. On a talkative turn, the voice data is
> the bigger part.
>
> **So how fast it fills depends on how much the model SPEAKS, not on how many turns you take.**

That explains a counter-intuitive observation: **50 turns of short answers were fine, while 27 turns
of long-winded answers crashed** — the longer conversation was the safer one.

**What happens now.**

- **Every turn prints how much is used against the limit.** Before this, the situation was
  **completely invisible until something died**; no existing indicator hinted that it was close.
- An optional allowance: on reaching it, **end the conversation cleanly** and tell the client why,
  instead of letting a stage crash or silently freeze.

This is only a **guardrail**, not a solution — when it triggers, the conversation is over. The
solution is the next section.

---

## 5. Letting the conversation run indefinitely

*2026-07-31 03:46 · `4502b142`*

**Scenario.** The user wants to keep talking and not be cut off because something filled up inside.

**What happens now.** As the limit approaches, the system **swaps in a new underlying request** behind
the scenes:

1. retire the old one
2. open a new one
3. tell the new one **the last few turns of conversation, as text**, as an opening
4. hand it the pictures still in hand, along with that turn

**The user notices nothing** — only that this one reply takes longer, because the new one has to warm
up.

**What is lost in the swap.** Only the **text** carries over; the **picture memory does not**. So the
model still sees *now* (the frames in hand went with it) — what it loses is **detail it saw earlier**.

**Why the carried-over history must be limited.** Whatever is carried has to be read again by the new
request. Without a cap, **each swap carries more, until the opening alone fills the limit again** —
which would defeat the swap entirely. So by default only the last 8 turns go across.

In one line: **unlimited in time, necessarily limited in memory.**

**Effect.** A 40-turn conversation with 2 swaps along the way: no crash, no freeze. The same settings
**without swapping ended the conversation at turn 27**. The cost is that the swapping turn is about
3× slower, which spreads out to a few tens of milliseconds per turn.

---

## 6. The system had work in hand but believed it was idle

*2026-07-30 23:59 – 07-31 12:32 · `c27654de` count the real state (the fix) · `b1f423f1` count received-but-unprocessed data as work · `1aed4032` clear cancelled jobs · `9fdee244`, `b7392f3c` two attempts that missed*

**Scenario.** On the turn right after a swap, the conversation froze again — and this time without
even a sign of being busy.

**Why.** Internally there is a count used to decide "is there still work to do". That count **fails to
be decremented in some situations**, so it drifts. And **being off by one exactly cancels out one real
job** — so the system concluded "no work", **went to sleep**, and the work sat there with nobody to
pick it up.

An analogy: the tally at the door is broken and still says "1 person inside" long after they left; a
new customer gets cancelled out against it, the attendant sees nobody, turns off the lights and goes
home.

**What happens now.** That count is no longer trusted — **the real state is counted directly** each
time.

Two related problems were fixed alongside: data **already received and waiting to be processed** was
not counted as "work to do"; and cancelled jobs left sitting in a queue were later picked up by other
logic and killed the process.

**One judgement worth recording.** I tried **repairing** the broken count, and that made things worse
— it pushed it negative instead. **Reporting that somebody else's bookkeeping is broken is useful;
reaching in and editing it is not.**

---

## 7. Making failures speak up

*2026-07-31 01:19 – 04:21 · `921314e8`, `625ad170` freeze detection · `53a709a5`, `241aa5fd` state dumps · `cd29fe65` arrival/departure log · `713e5361` empty-stage report*

**Scenario.** What all the freezes above had in common: **the log said nothing.** The system sat there
quietly, and there was no way to tell "idle and healthy" from "frozen".

**What happens now.** Several self-checks were added. They stay quiet normally and speak only when
something is wrong:

- a stage that **holds work but makes no progress** → reported, together with a dump of its state
- a stage that goes from **holding work to holding none** → reported (meaning the job was lost before
  it even arrived)
- **every job's arrival and departure** logged, with the reason on departure ("finished normally" and
  "was cancelled" have nothing in common, and previously looked identical)

**There is a lesson here worth more than any single bug.**

All the self-checks originally sat at the **end** of one loop. When the loop stopped, they **all went
silent together** — and that silence reads as "everything is fine", which is precisely backwards.

What finally located the broken count was a probe placed **outside the part that can stop**: it watches
the decision to go to sleep itself.

> **Rule: keep at least one probe outside the region that can stop, and have it report "I am here and
> this is what I see" — never infer health from the absence of errors.**

---

## 8. The measurement tooling moved into the repo

*2026-07-31 13:40 · `404b16b0`*

The scripts that run the experiments, the scripts that produce the reports, and the plots all now live
in `benchmarks/live_agent/`. The reason is simple: **they measure this code, so they have to
be versioned with it** — kept apart, they drift out of sync.

---

## 9. Some of the numbers were fake

*2026-07-30 23:36 / 23:41 · `d6ba9563` report input size for every stage · `38589f1f` actually measure the stage-to-stage transfer*

**Scenario.** We wanted to know how much input the speaking stage receives each turn — because it
was the suspected cause of the slowness.

**What used to happen.** Two things made the question unanswerable:

- the code counting input size was behind a condition that **only counted the first stage**. The
  speaking stage's input size **was simply not counted at all**.
- the time and bytes for passing data between stages were **hardcoded to 0** — not mismeasured,
  never measured, with a zero filled in (the original author left their own TODO there).

**What happens now.** Every stage reports its own input size, and the building and sending of each
inter-stage transfer is genuinely timed, switchable on demand.

**Why this comes last.** It changes no behaviour and no user can feel it. But **every section above
depends on it** — without those two numbers, a question like "is the speaking stage the cause?" can
only be guessed at.

---

## Where things stand

**Working**

- One user, camera and microphone on, long continuous conversation
- Only new pictures processed per turn; pictures shrunk automatically; frame pacing bounded
- The conversation **can run indefinitely** (automatic request swap, invisible to the user)
- Warning before the limit, and the distance to it visible every turn
- Three kinds of silent freeze now report themselves

**Verified**

- 40 turns with 2 swaps: no crash, no freeze. Same settings **without swapping ended at turn 27**
- 50 turns without swapping: all 50 completed, clean shutdown
- Memory in a long conversation did not get worse (**as long as no swap happens**)

**Explicitly not done or not measured**

- **Memory across a swap was never tested.** In principle text carries over and pictures do not, so
  the **expectation** is "remembers what was said, forgets what was seen earlier" — but an
  expectation is not a result, and this has not been measured.
- **Single user only.** Several people using it at once is completely untouched.
- There is a self-heal that recovers a stuck job, but **why it gets stuck that way is still unknown.**
- The broken count is only **worked around** here; it has **not been fixed upstream.**

---

## Appendix: settings and where the code is

The italic line under each section heading is that change's date and commits.
Everything is on this repository's `live-agent` branch: 23 commits, 6 files, about 1,600 lines.

The new settings all live in `session.config`, with these defaults:

`max_frame_width` / `max_frame_height` = no shrinking · `frame_jpeg_quality` = 90 ·
`frame_filter_min_gap` / `max_gap` = unbounded · `session_scoped_request` = off ·
`session_talker_token_budget` = none · `session_roll_at_talker_tokens` = off ·
`session_roll_history_turns` = 8 turns · `session_roll_settle_s` = 1 second
