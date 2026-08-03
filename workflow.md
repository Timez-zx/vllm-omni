# What we added to vllm-omni

Goal: let one person keep a camera and microphone on and **talk to the model for a long time**,
with every reply arriving quickly.

The original could manage a few turns. It broke down over longer conversations. Below, one section
per change, in the order it was added, each written as: **scenario → what used to happen → what
happens now**.

The code is in this repository, branch `live-agent`. Every new capability is **off by default** — an
unconfigured deployment behaves like the original, with **one exception**: section 10 changes what one
model stage hands another, and it is **on** by default. `VLLM_OMNI_TALKER_TEXT_ONLY=0` turns it off.

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

## 10. The speech stage was carrying pictures it never looks at

*2026-07-31 16:45 · `796795ac`*

**Scenario.** A conversation cannot run forever: eventually the part that produces speech runs out
of room, and the whole request has to be swapped out (section 5). So the question is what actually
fills that room up.

**What used to happen.** Two things fill it — what the user showed the camera, and what the model
itself said. The first one is the surprise: **the speech stage is handed one slot for every piece of
every picture.** One sharp frame is 880 slots. It gets them even though **it never looks at
pictures** — its job is only to know which words to pronounce. And those slots are never given back.

*Example, from a real turn:* three pictures in the turn, and the speech stage was carrying **817
slots**. The words themselves accounted for about 24 of them. Everything else was pictures and sound
it had no use for.

**What happens now.** The pictures and the recording are simply **not handed to the speech stage**.
It gets the words: what the user asked, and what the model is about to say. That same turn now costs
**33 slots instead of 817** — about **25× less**.

**Why we think this is safe.** Two reasons, one borrowed and one measured:

- Another team's model built the same way (MiniCPM-o 4.5) gives its speech part **only the words and
  no pictures whatsoever**. So the design does not require them.
- Measured here, five turns each way: **every single turn still produced speech**, on both settings.
  The two halves of the bookkeeping agreed exactly every time, which is the thing that would
  otherwise go wrong quietly.

**What we do not know.** Two honest gaps:

- **Whether the voice sounds different.** Five turns, one conversation each way, and nobody listened
  to the audio. This test can catch "it stopped talking"; it cannot catch "it sounds a little flat".
  This model was trained *with* the pictures present, so a small loss is possible.
- **How much longer a conversation can now run.** Not measured yet. In the test the model happened to
  answer about twice as wordily on the new setting, and **the model's own speech fills the same
  room** — so the extra talking ate half of the saving. Needs a rerun with the model's randomness
  turned off, so both runs say the same thing.

**Reversible.** One setting puts the old behaviour back. It is kept deliberately, so that the
comparison stays reproducible rather than becoming a story about how things used to be.

---

## 11. You can now talk to it from a browser

*2026-07-31 19:00 – 20:30 · `2452f4dd` bring this work onto upstream main · `ccff770e` the browser client · `5dfc170d` a deploy config that fits*

**Scenario.** You want to hold an actual conversation with it — your own camera and
microphone, its voice back — rather than driving it from a benchmark script.

**What used to happen.** The only clients were measurement harnesses. They send
frames and audio from files on disk, on a fixed schedule, and they ask their
question as text. Useful for numbers, useless for finding out whether talking to
the thing feels like anything.

**What happens now.** A web page. It captures your camera and microphone, streams
both continuously, and plays the reply. One command each for the server and the
page, one forwarded port, and `localhost` counts as a secure context so the
browser grants camera and microphone with no certificate to set up.

Most of it was **not written from scratch** — upstream ships a browser client for a
different model, and its microphone capture and stylesheet are used verbatim. What
had to be rewritten is the part that speaks to *this* server, because the two
protocols differ in three ways that each fail quietly:

* audio **up** is raw samples; audio **down** is a complete sound *file* per chunk.
  Gluing those files together end to end puts a 44-byte header into the audio every
  chunk, which is a click you would blame on the model.
* pictures are their own message here, not an attachment to the audio.
* **something has to say "I've finished speaking"**, and on this model that
  something has to be the page. See below.

**The honest part.** This model has no way to decide when to talk. There is no such
choice in its vocabulary, and the server never starts a reply on its own. So the
page decides — either it notices you have gone quiet for 0.7 s, or you hold a
button. **The page says this on screen**, because the other model in this repo
*does* make that decision itself, and someone comparing the two deserves to know
which one is actually driving.

**Effect.** Verified without a browser first, so the result is about the server
rather than about one laptop: five turns, every one produced text and speech, first
sound at **340–376 ms**. That is the same order as the 513 ms measured before,
which is reassuring but is **not** the same experiment — different engine version,
fewer concurrent slots.

**And a bug the check nearly missed.** "Does it produce audio?" passed. Then the
same question was asked with a longer reply, and the answer did not change: a
34-character reply and a 104-character reply both deliver **0.22 seconds** of
sound. Only the first small piece of speech reaches the page; the rest is made and
lost. A second run with our own newest change switched off produced the same 0.22 s,
so **that change is not the cause** — the loss is in the path this branch inherited.
Recorded, not worked around, and the page is not worth listening to until it is
fixed.

---

## 12. Pictures are now understood while you are still talking

*2026-08-01 · `a9e4e455` the feature, and the two bugs underneath it*

**Scenario.** You hold up the camera and talk for a few seconds. Pictures keep arriving the
whole time.

**What used to happen.** Each picture was **set aside** until you stopped talking. Only then
were they all turned into something the model can read — and that work sat directly in front of
your answer. So the pictures were paid for at the worst possible moment: after the question,
while you were waiting.

**What happens now.** Each picture is turned into model-readable form **the moment it arrives**,
while you are still speaking. By the time you stop, that work is already done and only the
question itself remains.

**Why this matters more than the milliseconds.** The old way meant every extra picture per turn
made the reply slower, which is exactly why the camera was kept deliberately slow. Doing it on
arrival breaks that link: sending pictures more often stops costing you waiting time.

**What it took, and this is the part worth reading.** Turning this on killed the speech engine
outright — every time, on the first reply. It took **five** attempts, and the first four all
failed in exactly the same way, which was itself the misleading clue: it looked like one cause
not yet found, and it was actually **two separate faults happening at once**, where the first
one crashed the process before the second could ever be seen.

> **Fault one.** When the model produces a word, the system writes a **placeholder** where the
> real word should go, and fills the real one in on the next step from a short-lived copy. That
> copy only reaches requests that were busy on the previous step. Our new "understand this
> picture now" work briefly sets the speech part aside — so when it came back, the real word was
> gone and only the placeholder was left. The speech part then looked that placeholder up in a
> table it does not belong to, and the process died.
>
> **Fault two.** The picture-understanding step was handing the speech part a **bundle of the
> wrong shape** — one meant for "here is a lot of new context", labelled as "here is one more
> word". Nobody could see this until fault one was fixed, because the crash always arrived first.

**Two habits that came out of it, both learned the hard way.**

*Do not send a note; look at the thing itself.* Three times we tried to **tag** the
picture-understanding work so the later stages would recognise it, using three different ways to
carry the tag. All three arrived **blank**, silently — the field the tag rode on had already been
overwritten by the next piece of work. What finally worked was asking a question about the work
in front of us instead: *did this step take in a lot of new material and then stop?* If yes,
there is nothing to say out loud. That fact cannot get lost in transit, because it is not being
carried anywhere.

*A missing log line is not evidence.* Twice a check was written to confirm something by the
**absence** of a message, and twice it was wrong — the code had simply not run. Every check here
now confirms things by **presence**: the message has to appear.

**What it cost, honestly.** Four wrong attempts, each needing a full engine restart, before the
approach changed from "guess a cause and test it" to "read the whole path at once and let the
code say". The lesson generalises: when several different fixes fail in **identical** ways, stop
looking for one missing cause and start suspecting two.

**The clean number, finally (re-measured 2026-08-01 after every fix in section 13).** Tripling
the pictures per turn costs **+41 ms with this off and +10 ms with it on** — about **4× less
sensitive to frame rate**, and the reply time's spread tightens from 43 ms to 8 ms, which is
the half that matters for predictability. An earlier **16×** figure was measured while some
replies were being credited to the wrong turn; it was inflated and is withdrawn. This is what
"measure again after the bookkeeping is fixed" exists for: the effect is real, and it is a
quarter of the size first claimed.

---

## 13. The voice went muffled and kept getting cut off — one symptom, three faults

*2026-08-01 · `9624cac1` `3093205d` — heard by ear, pinned by a five-way audit of the code and the logs*

**Scenario.** With pictures-on-arrival switched on (section 12), you talk to it for a few
turns. The first reply sounds fine. From the second or third, the voice goes muffled, and
sometimes it stops mid-sentence and never finishes.

**The one-line background.** Every kept picture spawns a small job: *"read this, say
nothing."* The trouble was that only the **reading** part of the model ever received the
"say nothing" half. The **speaking** part treated every one of those jobs as something to
answer: it hummed a little meaningless audio, and then announced *"done speaking"* — a real
announcement, the same kind a real answer ends with.

**Fault one — the announcement claimed the wrong owner.** That "done speaking" takes about a
second to travel from the speech engine back to the front desk. If your next question was
asked inside that second, the front desk heard "done speaking" and concluded *your answer*
had finished — half a sentence in. The rest of the real answer arrived moments later, was
ruled "no turn is running", and was thrown away. One session lost **27.5 seconds** of real
speech that way; every single broken turn in the logs sits exactly on one of these
collisions, and no healthy turn does.

The fix is not a smarter flag — flags read *at the moment something arrives* are exactly
what lost this race. The fix is a **ledger of submissions**: work goes into the engine
through one door and comes out finished in the same order, so the k-th "done speaking"
belongs to the k-th job, full stop. A read-this job's humming and its announcement are now
identified by name and discarded; only your question's own announcement can end your turn.

**Fault two — the seams stopped being sewn.** The voice is built in small blocks, and each
block must be stitched to the tail of the previous one — like matching the pattern when
hanging wallpaper. One counter in that stitching arithmetic **never resets**, and it was
being used to measure a list that **resets every turn**. From the second turn of every
conversation onward, the second block of every reply was stitched with **no overlap at
all** — that seam is the muffle. The same broken arithmetic, fed a very short piece, computed
a negative length and threw the piece away entirely. The counter that resets at the right
time already existed in the codebase; a sibling model's code was already using it.

**Fault three — the humming had no limit.** Only visible once the first two were fixed: the
meaningless humming was **unbounded**. One job hummed for **50 seconds**, and since the
speaking part does one job at a time, the *voice* of the next real answer stood in line
behind it — its text ready in 2 seconds, its sound arriving at 52. Now any job that arrived
with nothing to say may make at most one sound before it is stopped. The stopping mechanism
already existed too; it had simply never been wired to the speaking stage.

**The habit this run adds to the list.** Sections 5 and 12 each ended with "it was two
faults, not one". This is the third time — and the count is the lesson now. When a symptom
survives a correct-looking fix, the next hypothesis should not be "the fix was wrong" but
"there is another fault under it". Also worth keeping: the decisive evidence was already
sitting in old logs — every broken turn in weeks of history had the feature on, every
feature-off session was clean. A natural experiment nobody had to run, only to read.

**Verified, not assumed.** Replayed the browser's exact rhythm against the live engine —
static camera forcing one read-this job every 2 seconds, eight questions timed to land inside
the collision window. All eight turns came back whole, every announcement was claimed by its
rightful owner, the cap armed and released on every job, and the 50-second tail is gone. The
same pressure before the fixes broke the second turn within seconds.

---

## 14. Sixteen people at once — the waiting holds up, the lifetime doesn't

*2026-08-01 first pass (bf16) · 2026-08-02 fully re-measured on the FP8 engine · `e91735b3` `cd399ae0` `ba64c261` `73131a2b` — a 15-cell matrix: 1/2/4/8/16 users × three kinds of picture*

**Scenario.** Everything so far was one person. This run asks the cloud question: put
N people on the same card at once — does the waiting get worse, and how fast? Each
simulated person is a real connection through the same door the browser uses, pumping
2 real video frames a second and holding a 30-turn conversation (ask, listen to the
whole answer, think a few seconds, ask again). Three kinds of picture from the July
study: a still screen, a talking head, a walking handheld shot. Every cell starts on a
freshly booted engine, and reply length was pinned by the prompt (27–29 characters in
all 15 cells), so a difference in waiting can only come from the crowd.

**The waiting, measured** (FP8 engine; all 15 cells completed in full, zero deaths.
The three numbers are P50 / P95 / P99 in milliseconds. P99 excludes each cell's
first-turn cold start — the FP8 engine's very first turn after boot costs 2–4.5 s,
paid once per boot; and with 60–480 samples per cell, P99 should be read as "the
worst few turns seen", not a strict percentile):

| users | still screen | talking head | handheld walk |
|---|---|---|---|
| 1 | 380 / 454 / 499 | 385 / 473 / 496 | 399 / 517 / 551 |
| 2 | 439 / 695 / 695 | 421 / 743 / 743 | 453 / 902 / 902 |
| 4 | 575 / 794 / 813 | 614 / 797 / 832 | 673 / 977 / 1,048 |
| 8 | 693 / 965 / 1,074 | 700 / 932 / 985 | 839 / 1,317 / 1,406 |
| 16 | 783 / 1,250 / 1,416 | 798 / 1,069 / 1,121 | **1,405 / 4,673 / 5,368** |

**Where sixteen users actually get slow.** The table's one ugly cell (handheld × 16)
is not jitter — it is steady decay: 0.9 s at turn 2, 3.3 s at turn 30, and the
slowest 5% of turns all land after turn 22. Split each turn in two and look at
ABSOLUTE milliseconds (16 users, early → late, P50/P95):

| segment | early (turns 2–8) | late (turns 18–25) |
|---|---|---|
| thinking (question → first text char) | 202 / 541 | **1,151 / 2,404** |
| speaking (first char → first sound) | 686 / 954 | 1,297 / 2,216 |

Then pull out the per-second hardware samples for exactly the seconds when the
slowest 5% of turns were running: **the SM (compute units) sit at 93% while memory
bandwidth sits at 18%**; split by process, the thinker owns 64% of the compute
during slow turns (50% on median turns) while the talker FALLS from 30% to 21% — it
did not get slower, it **cannot get the card**. Both segments slow down for one
root cause: the thinker eats the compute.

**Why many pictures make COMPUTE the wall, not memory bandwidth.** The engine only
does two kinds of work:

- **prefill (digesting input)**: one frame becomes 220 tokens entering as ONE
  batch — the whole history is fetched from memory ONCE, and all 220 tokens each
  run their multiplications against that one fetch. Fetch ×1, compute ×220.
- **decode (producing the answer)**: one character at a time — the same full
  history is fetched, but only ONE token's worth of compute uses it. Fetch ×1,
  compute ×1.

This card's constitution: per byte fetched, the compute units can afford ~110
multiplications. Decode uses 1–2 per byte (bandwidth clogs first — the origin of
the old saying "LLM inference is bandwidth-bound"); prefill uses ~350 per byte
(compute clogs first). A concrete ledger at 45k tokens of history: one frame's
prefill takes **3 ms of fetching and 40 ms of computing** — compute is 13× the
fetch, so the fetch pipeline idles. And our load is lopsided: ~30 decode tokens per
turn versus ~1,500 frame tokens — **96% of the work is prefill-shaped**, so the
whole card behaves like prefill: SM pinned, bandwidth idle. Total demand in one
product: **users × frames-per-second × history thickness per frame**. The first two
factors are fixed; the third climbs every turn — handheld × 16 crosses the card's
capacity around turn 13, and everything after that is queueing.

**The counter-example proves the rule: a duplex WITHOUT pictures hits the bandwidth
wall first.** We have measured Moshi (speech-only full-duplex) under multi-user
load: it has no batch-shaped input at all — audio enters step by step, one time
slice every 80 ms, so its load is 100% decode-shaped at 1–2 multiplications per
byte, on a dense model that re-fetches full weights every step — and under
multi-user load **memory bandwidth broke first**, the exact mirror of this table.
Two systems, two opposite walls, one criterion (multiplications per fetched byte)
explaining both. In one sentence: **the moment a live agent grows eyes, its
bottleneck migrates from memory bandwidth to compute** — and the medicine changes
with it: the bandwidth wall wants fewer bytes moved (quantize, batch wider), the
compute wall wants fewer multiplications done (cap the history, admit by the
product) — the two prescriptions do not transfer.

**The wall (bf16 history — no longer reached in the FP8 re-measure).** Every conversation's context only ever grows, and the thinking stage's
memory pool is a fixed size regardless of how many people share it. So the pool runs
out at a **predictable turn number**: pool ÷ (people × growth per turn). One number —
how many pictures per second the filter keeps — sets the growth, and it predicted every
death in the matrix, including one written down before the cell ran (16 users ×
handheld: predicted around turn 4, died turns 3–5). Light content keeps 25% of frames
and 8 people die near turn 18; the handheld shot keeps 56%, so **4 people die near turn
18 too** — "four users is safe" is a statement about the camera, not the card.

**How FP8 tore that wall down.** Quantizing the thinker's weights and KV cache
to FP8 (vLLM converts the OFFICIAL weights at load time; talker and vocoder
stay bf16) grows the memory pool from 107k to **731,904 tokens**; the table
above IS the full re-measure on that engine — 15 cells, zero deaths, text
verbatim identical to bf16, latency flat. Three lessons from the road: a
community pre-quantized checkpoint loaded fine and spoke gibberish
(discarded); the 0.74 memory fraction was tuned for bf16 — fp8 kernels'
workspace ate the headroom and OOM'd at 8 users, so 0.70 trades 83k pool
tokens for 4 GB of transient working memory, the real concurrency constraint;
and the human-ear gate passed (normal Mandarin conversation confirmed) with
ONE session-level oddity on record — a session that spoke Cantonese from its
first reply onward, gone on reconnect, mechanism pointing at a borderline
language call on the first spoken utterance then locked in by session
history. If it recurs, keep the session and read its first line.

**Why each crash happened.**

1. **Eight users deadlock together — a design problem: a parked session's lost signal
   is never redelivered.** When the pool fills, the scheduler parks a session request
   to make room — normal operation. But the "done speaking" signal is delivered
   exactly once; miss it while parked and it is gone forever. The turn can then never
   close, and every later question is refused with "a turn is still running". The
   health check tests HTTP, not audio, so the deadlock reads as "fine" throughout.
   **The latency problem became a lifetime problem.**
2. **Sixteen users crash the engine — an implementation problem: the memory fraction
   does not govern runtime peaks.** `gpu_memory_utilization` only sizes the memory
   pool; nothing bounds the instantaneous activation memory of sixteen concurrent
   picture prefills. The moment that peak exceeds the card's remaining VRAM, stage 0
   hits CUDA OOM and the process dies.
3. **Two users in one batch crash — an implementation problem: the multi-request
   batching code had never run** (fixed, `e91735b3`). Stage 2 picks its processing
   path by whether the batch's TOTAL length divides by 16; one 1-token stub in the
   batch breaks that, the code flattens N requests into one row and then slices it as
   N — index out of bounds, engine down. Fix: pad each request individually. Two
   collateral bugs in the same code: a request missing the "seam" field zeroed
   **everyone's** seam parameters (fixed, same commit); different-sized chunks in one
   batch still mis-place a seam by ~23 ms (open; audio output frozen during the
   benchmark).

---

## 15. Context compression — the notebook swap users can no longer feel

*2026-08-02, commits `0b37129c` / `22ce6ea6`.*

**The problem.** The longer a conversation runs, the thicker the model's notebook gets.
Thick is bad twice over: with many users, every turn gets slower than the last (the
climb section 14 measured); and a single user who talks past 65,536 tokens hits the
model's memory ceiling — what happens there has never been verified. The old fix was
"swap notebooks the moment the page runs out": at the instant the user asked a
question, the old conversation was deleted and rebuilt, and that turn waited 3.70×.

**The fix now: prepare the new notebook in the background, then just flip.** Four steps:

1. **Keep count.** There are two models in this system, each with a notebook that
   fills up — the thinker (looks at pictures, writes the answers) fills its notebook
   mostly with **pictures**; the talker (turns answers into speech) fills its own with
   **everything it has said**. The two fill at unrelated speeds (lots of picture and
   short answers fills the first; a still scene and a chatty model fills the second),
   so each is watched separately, and either one crossing its line starts preparation.
2. **Prepare.** In a silence gap, quietly open a new conversation and pre-load it with
   the text of the last few exchanges. The old conversation keeps serving; the user
   feels nothing. While preparing, the talker mumbles a short burst of junk speech at
   the seed text; the system **waits for the mumbling to finish and throws it away
   before calling the new notebook ready** — that rule was bought with a measured
   failure, see below.
3. **Flip.** The moment the user asks the next question, it goes to the new
   conversation. It was prepared long ago, so this turn is as fast as any other. Any
   exchanges that happened during preparation ride along as text — nothing recent is
   lost.
4. **Delete.** The old conversation is deleted only after this turn finishes speaking,
   freeing its memory. Waiting is deliberate: the deletion never races the
   conversation in progress (the old mechanism slept 1 second to dodge exactly this).

After a flip the new conversation holds: the recent exchanges as **text**, plus the
**current pictures**. The old pictures are gone — that is the price of each
compression. Measured, the price is affordable: text carried 3/3 recall probes ("what
is my name", "what is my favorite color") across the flip; only visual detail from old
frames becomes unanswerable.

**When it triggers.** Each notebook has two lines; the defaults in numbers:

| Notebook | Preparation line (start preparing) | Forced line (only if preparation keeps failing: swap on the spot, 3.70×) |
|---|---|---|
| Thinker's (mostly pictures) | 49,152 (= 75% of the 65,536 model limit) | 60,293 (= 92% of the limit, so 65,536 is never reached) |
| Talker's (mostly speech) | 38,250 (= 85% of 45,000) | 45,000 (hitting this wall kills the engine — no gambling, swap immediately) |

The line is **the same for one user and for many — 75% of the model limit, uniformly**
(Xiao's call): compression has exactly one job, lifetime — the conversation must never
hit 65,536. The latency that a thick history causes under load is a different problem
that belongs to scheduling; compression does not moonlight as a latency knob. The
uniform policy carries one hard capacity rule: **concurrent sessions ≤ KV pool ÷
trigger line** (this card: 732k ÷ 49,152 ≈ 14; take 13 for margin) — beyond it the
pool runs dry BEFORE the trigger, see the measured sixteen-user death below.

**What it measures like (three scales, one 75% line).**

**One user × 48 turns:** p50 371 / p95 409 ms, a straight line throughout. At turn 34
the context reached 49,217 and triggered automatically; turn 35 swapped invisibly
(409 ms against neighbors at 373/365). **From turn 44 on, the session lived past its
old grave** — without compression the accumulated input would have hit 65,536 around
turn 43; this is the first session ever to outlive that point. Memory across the swap
was probed separately (a smoke run with the line forced to 2,500 and two swaps):
recall questions like "what is my name" went 3/3.

**Thirteen users × 48 turns (inside the capacity boundary, 13 × 49,152 = 87% of the
pool):** 624/624, zero timeouts, 13 swaps all invisible, 0 fallbacks. The curve is a
textbook sawtooth:

| Turn | per-turn p50 (ms) |
|---|---|
| 10 | 821 |
| 26 (sawtooth peak) | 2,689 |
| 30 (after the swaps) | **815** |
| 38–48 | 900–970, flat |

The 2.7 s peak is the price of the 75% policy at thirteen users — compression manages
lifetime, not this; flattening that peak is the scheduler's knife (priority for urgent
work), a separate one.

**Sixteen users × 48 turns (outside the boundary, 16 × 49,152 = 107% of the pool):**
**everyone died, one step short of the trigger.** At ~45k per user (98% of the pool in
aggregate) the KV pool ran dry; every session stuck in a wait for memory blocks that
never came, turn signals lost, each user written off after three timeouts; zero swaps
all run — nobody lived to 49,152. The self-checks from section 7 dumped the whole
anatomy in real time. This death is the measured proof of the capacity rule above.

*(Side note: the mechanism itself CAN moonlight as a latency knob — sixteen users with
the line hand-tuned to 16,000 once measured p95 1,320 ms, flat across 40 turns, zero
deaths. Current policy does not use it that way; recorded here only to mark the
mechanism's envelope.)*

**Three traps.**

1. **Preparation must not be "read without speaking".** That ships the talker an EMPTY
   first page; when the first real content arrives after the flip, its numbering
   assumes the full text while its data covers only the new part — mismatch, and the
   stage-1 engine dies on the spot. Fix: the seed reads and then says two throwaway
   tokens before stopping, so the talker's first page looks exactly like any ordinary
   conversation's first page.
2. **"Ready" must wait for the junk speech to finish.** Work inside one conversation
   runs strictly in line: flip too early and the user's real words queue behind the
   mumbling — measured as 8 of 640 turns where text arrived in 0.1 s but sound took
   6–26 s, every one on a swap. After the rule change: zero recurrences; all the
   waiting happens in the background while the old conversation keeps serving.
3. **Shadows occupy engine concurrency slots too.** Leave headroom (20 slots for 16
   users); a shadow that cannot get a slot just swaps one turn later — never an error.
   But headroom itself costs VRAM: 34 slots pushed the waveform stage into a boot-time
   crash (third confirmation that the memory fraction budgets the KV plan, not runtime
   buffers).

**Knobs** (all in `session.config`): `context_compression_trigger_tokens` = the
preparation line, default auto at 75% of the model limit, 0 disables ·
`context_compression_target_tokens` = how much text rides across a swap, default 4096 ·
`context_compression_warmup_timeout_s` = how long preparation may take, default 30 s;
on timeout the old blocking swap is the fallback. Two hard deployment rules:
`max_num_seqs ≥ sessions + shadow margin`, and concurrent sessions ≤ KV pool ÷ trigger line.

---

## Where things stand

**Working**

- One user, camera and microphone on, long continuous conversation
- Only new pictures processed per turn; pictures shrunk automatically; frame pacing bounded
- The conversation **can run indefinitely** (automatic request swap, invisible to the user)
- Warning before the limit, and the distance to it visible every turn
- Three kinds of silent freeze now report themselves
- The speech stage no longer carries the pictures — **25× less to hold per turn**
- **A browser client**: your own camera and microphone in, its voice out, one forwarded port

**Verified**

- 40 turns with 2 swaps: no crash, no freeze. Same settings **without swapping ended at turn 27**
- 50 turns without swapping: all 50 completed, clean shutdown
- Memory in a long conversation did not get worse (**as long as no swap happens**)
- Speech still produced on **every** turn with the pictures withheld from the speech stage
  (5 turns each way)
- The browser path end to end on the newer engine: **5 turns, all produced speech, first sound
  340–376 ms** — verified without a browser, so it is a statement about the server

**Explicitly not done or not measured**

- **Memory across a swap was never tested.** In principle text carries over and pictures do not, so
  the **expectation** is "remembers what was said, forgets what was seen earlier" — but an
  expectation is not a result, and this has not been measured.
- **Multi-user is measured (section 14) but not survivable.** The waiting scales fine; the
  session **dies** at a predictable turn when the shared memory pool fills. The two fixes that
  would change that — surviving being parked, and admitting people by context growth instead of
  head-count — are designed on paper and not built.
- There is a self-heal that recovers a stuck job, but **why it gets stuck that way is still unknown.**
- The broken count is only **worked around** here; it has **not been fixed upstream.**
- **Nobody has listened to the audio** since the pictures were withheld from the speech stage, and
  **how much longer a conversation now runs is not measured** (section 10).
- **The browser's audio is truncated to 0.22 s per turn** (section 11). The chain works; the sound
  does not yet. Ruled out as ours by an A/B, cause not yet found.
- **The numbers on the newer engine are not the old numbers.** 513 ms and the 40-turn roll run were
  measured on the older engine with a different config; they need re-running before anything is
  compared across the two.

---

## Appendix: settings and where the code is

The italic line under each section heading is that change's date and commits.
Everything is on this repository's `live-agent` branch: 26 commits, 8 files, about 1,700 lines.

Section 10 is the one setting that is *not* in `session.config` — it changes what one model stage
hands another, below the level a per-conversation setting can reach, so it is an environment
variable: `VLLM_OMNI_TALKER_TEXT_ONLY`, **on by default**, and `=0` restores the original behaviour.

The rest all live in `session.config`, with these defaults:

`max_frame_width` / `max_frame_height` = no shrinking · `frame_jpeg_quality` = 90 ·
`frame_filter_min_gap` / `max_gap` = unbounded · `session_scoped_request` = off ·
`session_talker_token_budget` = none · `session_roll_at_talker_tokens` = off ·
`session_roll_history_turns` = 8 turns · `session_roll_settle_s` = 1 second ·
`context_compression_trigger_tokens` = auto (75% of the model limit; 0 = off) ·
`context_compression_target_tokens` = 4096 · `context_compression_warmup_timeout_s` = 30 seconds
