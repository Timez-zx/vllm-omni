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

**The setup these numbers hold for.** One 96 GB card (RTX PRO 6000 Blackwell), all
three model stages sharing it: the thinker runs FP8 weights + FP8 memory (KV); the
talker and the waveform stage stay bf16. The memory split: the thinker gets 70%
(31.4 GB weights + a 33.5 GB memory pool), the talker 10%, the waveform stage 8%, the
rest is runtime headroom. **The memory pool holds 731,904 tokens, shared by every
session, fixed at boot.** 20 concurrency slots; every frame shrunk to 640×352 (220
tokens); the content is the most expensive one (handheld, high motion); one frame
every 0.5 s, 4–5 surviving the filter per turn; short questions, one-or-two-sentence
answers.

**Why this card tops out at 13 users.** Under the uniform 75% policy every session's
memory grows to at most 49,152 tokens before it swaps — so each user effectively
claims a plot of up to 49,152 in the shared pool:

| Concurrency | peak share of the pool | outcome |
|---|---|---|
| 13 users | 639k = **87%** | measured pass (the remaining 13% covers the brief old+new coexistence at a swap, plus working churn) |
| 14–15 users | 94–101% | theoretical edge, not measured |
| 16 users | 786k = **107%** | measured death — the pool ran dry at ~45k/user (98% aggregate), before anyone reached the swap line |

Three ways to hold more people: a card with a bigger pool, a lower trigger line (which
is the latency-knob usage, not current policy), or true fine-grained deletion that
actually frees memory (the second knife, still on the shelf).

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

**Latency summary for every scenario in this section (TTFA, ms; all on the uniform
75% line except the last row).**

| Scenario | p50 | p95 | **p99** | max | over 2 s | shape |
|---|---|---|---|---|---|---|
| 1 user × 48 turns | 371 | 409 | **437** | 437 | 0% | flat |
| 13 users × 48 turns | 968 | 2,915 | **3,537** | 3,885 | 14.6% | sawtooth (peak at turns 22–29) |
| 16 users × 48 turns (over capacity) | 1,161 | 3,765 | **5,347** | 5,915 | 26.0% | surviving turns before the deaths; turn signals start vanishing at turn 24 |
| Reference: 16 users, no compression (section 14 baseline) | 1,378 | 4,477 | **5,368** | 10,194 | 38.4% | a climb that never comes back |
| Side note: 16 users @ hand-tuned 16,000 | 871 | 1,320 | **1,860** | 6,498 | 1.0% | low sawtooth (not current policy) |

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
`context_compression_target_tokens` = how much content rides across a swap, default
16,384 (see the next section; it used to be 4,096 of pure text) ·
`context_compression_carry_frames` = whether frames ride along, default true, false
restores the text-only carry · `context_compression_warmup_timeout_s` = how long
preparation may take, default 30 s; on timeout the old blocking swap is the fallback.
Two hard deployment rules: `max_num_seqs ≥ sessions + shadow margin`, and concurrent
sessions ≤ KV pool ÷ trigger line.

---

## 16. Carrying the picture across a swap — and what it costs under load

*2026-08-03, commits `4680cd57` / `14a34076`.*

**The problem.** The swap in the previous section carries only TEXT. Text answers
"what is my name", but every visual detail — the colour of someone's top, the object
on the shelf — is gone. Section 9 measured exactly that: with a text notebook, spoken
recall held 8/8 while fine visual detail scored 0/8. So each swap made the model
forget, in the eyes rather than the ears.

**What it does now: the carry becomes a rolling window of the real thing.** The newest
turns cross verbatim, frames included (budgeted at 250 tokens per frame); older turns
degrade to text-only as a floor; older still are dropped. The window fills to
`context_compression_target_tokens` (16,384). This is the shape Gemini Live's own docs
describe — discard the oldest, let the result begin at a complete user turn, always
keep the system instructions — with "what is kept" upgraded from text to the original.

**What it measures like.**

The cleanest evidence is a 10-turn single-user A/B in which the camera CHANGES SOURCE
mid-session: turns 1-4 look at a room (a person in a dark blue top, a trophy on a
shelf), from turn 5 the feed becomes a screencast, and the visual questions are asked
AFTER a swap. None of those details is ever mentioned in the text transcript:

| Asked after a swap | Frames carried (now) | Text-only (old behaviour) |
|---|---|---|
| Colour of the top | "dark blue shirt" ✓ | "black and white patterned top" ✗ (invented) |
| Object on the shelf | "trophy on the shelf" ✓ | "silver award" (half-lucky) |
| My name (text control) | ✓ | ✓ |
| Swap-turn TTFA | 451 ms | normal median 396 ms |

Frames also survived two consecutive swaps.

**One user, 48 turns: free.** p50 415 ms, flat end to end (first half 414, second half
416), the swap turn itself 371 ms, 119 frames riding in the seed. Visual memory costs
a single user nothing.

**13 users, 48 turns: not free.** 624/624 with zero timeouts, 23 invisible swaps, zero
fallbacks — but the whole latency distribution moves up a step:

| | Text-only (4,096) | Frames carried (16,384) |
|---|---|---|
| Completed | 624/624 | 624/624 |
| p50 | 980 ms | **1,355 ms** |
| p95 | 3,008 | **5,080** |
| p99 | 3,696 | **7,190** |
| max | 5,953 | **11,088** |
| turns over 2 s | 15.2% | **32.1%** |
| sawtooth peak (per-turn p50) | 3,110 (turn 25) | **4,745** (turn 27) |
| post-swap plateau | 910 | **2,059** |

**Why, in one line:** after a swap each user's context now restarts at ~14k instead of
~4k, and per-frame prefill is charged by how thick the history already is (the unit
price table in section 13) — so even the post-swap plateau doubles. This is not a
defect; it is the price tag on visual memory under load, and flattening the peak
remains the scheduling lane's job.

**Three parameters, each set from measurement.**

1. **Window 16,384, not 32,768.** A 32k seed took a median 9.5 s to warm, while 13
   users crossing together leaves only 5.8 s per warm-up inside the 11,141-token runway
   between the preparation line and the hard line — sessions queued behind it grew to
   58,680 (hard line 60,293), nearly consuming the runway. Halving the window closes
   the deficit. The price, stated plainly: the window reaches ~62 frames back instead
   of ~125.
2. **At most 2 warm-ups in flight.** The pool cannot fund a third: 13 sessions at the
   trigger already hold 87% of it, two 16k seeds fit in the remaining 92,928 tokens and
   three do not (the run peaked near 110% of the pool with two in flight).
3. **A warm-up that cannot get a permit QUEUES; it must not skip.** Skip-and-retry
   relied on frame arrivals and turn boundaries, and congestion had slowed exactly those
   to ~1/s — measured as 1-2 permits idle for ~10 s with 9 sessions waiting. With one
   waiter per session, first over the line first served, all 16 queued warm-ups got a
   permit and nothing fell back.

**Two engine defects fixed, one still open.**

1. **One marker was fatal.** A chunk carrying a single `<|im_start|>` made
   `torch.nonzero(...).squeeze()` collapse a [1,1] result to 0-d, and `torch.cat`
   refuses it — stage 1 died. Now `squeeze(-1)` (exact for every count), and the site
   logs the marker count plus head/tail ids: that shape was previously unknowable.
2. **A split segment shipped only its tail.** Under load the engine splits one
   segment's prefill across steps, and the payload builder overwrote its pending entry
   each step, shipping the LAST step's rows and sizing the talker's ids from that count
   — dropping the leading `<|im_end|>\n<|im_start|>user`. The steps now accumulate.
3. **Still open: 0.8% of turns still ship a fragment.** 5 of 624 turns in the final run,
   with the split counter at zero — so another path can make computed rows fall short of
   the segment. The crash is guarded, so this is now a LOGGED known defect rather than a
   silent one. The real repair is one payload per SEGMENT instead of per step, upstream.

**Two dead ends, recorded so they are not walked again.**

- **A "prompt tokens shipped to the talker" watermark cannot measure a segment.**
  Prefill-only appends (frames prefilled on arrival) ship nothing by design, so their
  tokens count against the next segment forever — the reported shortfall was exactly 1,
  2 and 4 frames' worth.
- **A partial payload cannot be withheld until the segment completes.** The stages are
  coupled step by step, so the next step's decode-only payload is then read as this
  segment's prefill — `KeyError: 'prefill'`, engine dead. A partial segment is therefore
  REPORTED, not repaired.

### At most 8 frames buffered per user — the largest win one parameter bought

*Xiao's call. `max_frames` goes 256 → 8 in both client configs.*

**Why frames pile up at all.** An arriving frame is supposed to be prefilled immediately
(section 12), but that path is shut for the WHOLE duration of a turn — the test is "has
this turn finished", not "is the engine busy", because frames share a queue with the
turn's own content and slipping one in makes the answer's segment boundary belong to the
wrong thing. It is shut during a compression warm-up too (the prefill would land in KV
about to be discarded). A refused frame is never retried; it simply waits for the next
turn to submit the whole buffer at once. Measured in the 13-user run: **2,727 frames were
prefilled on arrival and 4,903 were refused — 64% of frames never got the optimisation**,
and it fails exactly when congestion makes it most valuable.

**Why the cap is 8.** A 47-turn single-user regression gives the price of a turn:

```
TTFA ≈ 348 ms + 59 ms × frames in this turn
```

(measured medians: 1 frame 392, 2 → 426, 3 → 462, 4 → 521, 7 → 567, 13 → 876 ms). The
348 ms floor is everything that does not depend on frames; 59 ms is the marginal price of
one. At 13 users the floor rises to ~1,000 ms and a frame costs roughly 150-180 ms under
contention — which reproduces the 19-frame peak: 1,000 + 19×180 ≈ 4,400 against 4,745
measured. So the cap is not a guess, it is **the latency budget for frames ÷ the price of
a frame**: 8 frames = at most 472 ms single-user, ~1,400 ms at 13 users. Beyond the cap
the OLDEST frame is evicted — on a live feed the newest frames are the ones the answer is
about.

**What it bought (three scales, all with frames carried and the cap on).**

| | 1 user | 3 users | 13 users |
|---|---|---|---|
| completed | 48/48 | 144/144 | 624/624 |
| p50 | 409 ms | 573 | 1,168 |
| p95 | 544 | 828 | 3,535 |
| turns over 2 s | 0% | 1.4% (turn 1 only) | 18.1% |
| worst/best user p50 | — | 1.09× | 1.60× |
| is a swap visible | no | **no** | yes |
| frames dropped, whole run | 2 | 15 | 149 |

**Before and after the cap:**

| | 1 user: none → 8 | 13 users: none → 8 |
|---|---|---|
| p50 | 415 → 409 | 1,355 → **1,168** |
| p95 | 876 → **544** | 5,080 → **3,535** |
| max | 4,465 → **601** | 11,088 → 9,524 |
| turns over 2 s | — | 32.1% → **18.1%** |
| swap-window peak | 876 (13 frames) → **no bump at all** | 4,745 → 4,041 |
| post-swap plateau | — | 2,059 → **1,165** |
| worst/best user p50 | — | 2.06 → **1.60×** |
| playback starvation p95 | — | 834 → **558 ms** |

**Two things to read out of that.** First, the single-user spikes — the swap turn's 876 ms
AND the first turn's 4,465 ms — simply vanish, at a cost of 2 dropped frames in a whole
run; both were "accumulate a batch, then pay for it in one turn". Second, at 13 users the
TAIL and the PLATEAU are where the win is (p95 −30%, plateau −43%, fairness 2.06→1.60)
while the 4 s peak only drops 15%: that peak is not frame accumulation, it is thirteen
seed warm-ups and thirteen turns contending for one card — **which separates the two
problems cleanly: accumulation is a quota problem, the peak is a scheduling problem.**

**So carrying the picture across a swap is FREE up to 3 users** (573 ms, no visible swap,
fairness 1.09) and starts costing at 13, where the cost is mostly in the tail — and one
parameter already removed a third of it.

**Two notes.** The cap is PER USER: the buffer is a per-session local (each stores its
own) while the frames all land in one engine competing for one per-step token budget
(everyone shares the compute) — so this cap is also a crude per-user quota. And dropped
frames are counted into each turn's closing log line (`frames_dropped=`): a latency
control that discards input silently is one nobody can audit.

---

## 17. How many people fit, in three scenarios — the camera sets the ceiling

*2026-08-03, commit `1210772d`.*

One-sentence conclusion first: **the same engine seats 32 audio-only users with no
knee in sight, about 24 users of near-still video, and 13 users of high-motion
video already visibly slowing down. The three ceilings have three different causes, and
none of them is "the model is too slow".**

**How it was measured.** The three scenarios differ in exactly one variable: the
camera. Questions are text in all three (standing in for transcribed speech) and
answers are always spoken, so the output side carries the same load everywhere;
only the input side changes — no frames at all (`--content none`, added for this
study), a near-still feed (talkinghead, which the similar-frame filter cuts down
to the forced-through 25%), or a high-motion feed (handheld, where 57% of frames
count as "new"). 48 turns per user per cell, same protocol as section 16.

### Scenario 1: audio only — no knee at 32 users; the ceiling is configuration

| users | p50 (ms) | p95 | over 1 s | thinker p50 | speech p50 | rtf p10 |
|---|---|---|---|---|---|---|
| 1 | 317 | 319 | 0% | 33 | 284 | 1.65 |
| 4 | 504 | 616 | 0% | 36 | 464 | 1.45 |
| 8 | 607 | 645 | 0% | 39 | 569 | 1.38 |
| 13 | 633 | 677 | 0% | 40 | 593 | 1.34 |
| 16 | 639 | 698 | 0% | 42 | 596 | 1.32 |
| 20 | 663 | 731 | 0% | 43 | 620 | 1.29 |
| 24 | 668 | 746 | 1.0%* | 45 | 622 | 1.29 |
| 28 | 690 | 762 | 0% | 47 | 641 | 1.25 |
| 32 | 702 | 798 | 0% | 48 | 652 | 1.23 |

\* All of the 1% at 24 users is turn 1 right after an engine restart (a cold
start, with kernels still being compiled on the spot), not a concurrency effect.

Three things worth reading off this table:

1. **The latency is almost entirely the speech side's fixed startup.** The thinker
   (first token of text) takes 33–48 ms throughout; everything else is the
   text-to-sound pipeline getting going, and it barely moves with user count.
2. **The only step is between 1 and 8 users, and it is two discrete prices, not a
   slope.** Solo 315 ms, ensemble 585: at 4 users the two kinds of turns split
   44% / 56% with only ~13% landing between the two — the step is "is someone
   else generating speech at this same moment". From 8 users on, nearly every
   turn pays the ensemble price, which then rises only ~15% more all the way
   to 32 (607 → 702).
3. **What is actually thinning is the rtf margin** (how much faster speech is
   generated than played; at 1 the audio starts to stall): 1.65 → 1.23. After
   filling the 20-slot boot we restarted with a 32-slot config
   (`deploy_mu_fp8_s32.yaml`) and the curve stayed flat — so audio-only hits the
   **slot count (configuration)** first, and only later the speech-side rtf wall.
   Extrapolating the 13→32 slope: the margin falls through 1.1 at roughly 50
   users and reaches 1.0 (where audio starts to stall) at roughly 70;
   **both are extrapolations, not measurements**.

### Follow-up: is audio-only bottlenecked on GPU memory bandwidth? — Measured: no, not close

Decode classically eats memory bandwidth (every generated token re-reads the
weights from GPU memory), so the suspicion is natural. Verification: re-run
audio-only at 8 / 16 / 32 users while sampling, once per second — DRAMA (the
fraction of time the memory interface is actually moving data), sm% (the
fraction of time any compute kernel is running), and the per-process split
(pmon). No clock-locking experiments: the card is shared.

| audio-only | idle | 8 users | 16 users | 32 users |
|---|---|---|---|---|
| DRAMA median / p90 | 0 | 0.07 / 0.10 | 0.10 / 0.15 | **0.17 / 0.22** |
| sm% median / p90 | 0 | 27 / 63 | 34 / 82 | **58 / 90** |
| power (W, 600 cap) | 81 | 119 | 138 | 184 |

Three judgements:

1. **Bandwidth is nowhere near saturation.** 32 users (2.5× the video capacity)
   use 17%, and it grows roughly linearly with users (0.07→0.10→0.17); at that
   slope even 100 users would sit near half — while the rtf-margin wall
   extrapolates to 50–70 users, which arrives first. **Memory bandwidth will
   not be this engine's bottleneck at any reachable user count.**
2. **The shape is "many small kernels", not "starved for data".** sm% p90 is
   already 90 (some kernel is almost always running), but true SM occupancy
   (SMACT) is only 0.23 — the kernels that run use a small slice of the compute
   units. That is a scheduling / small-batch profile, and it corroborates the
   discrete ensemble-price step: the cost is queueing and co-batching, not a
   resource running dry.
3. **Split per process, the speech side is the biggest consumer — but the
   thinker is catching up.** At 32 users, median sm%/mem%: talker 29 / 6,
   thinker 15 / 4, code2wav 4 / 0. The thinker does produce output every turn,
   but only a dozen-odd text tokens finished in a few hundred ms, then back to
   waiting; the talker has to follow the whole delivery window at 87 codec
   tokens per second of audio (25 per chunk, one chunk per 288 ms). Note the
   thinker's share grew from 6% (16 users) to 15% (32) — it keeps growing with
   user count.

One honest correction: the back-of-envelope made beforehand ("the talker
re-reads its 6.26 GiB of weights every step") predicted DRAMA 0.4–0.6 at 32
users; the measurement says 0.17 — the talker process only has kernels running
29% of the time, meaning speech is generated in bursts on the chunk cadence,
not decoded wall-to-wall at delivery speed. The estimate was 3× high; the
measurement stands.

### Scenario 2: near-still video — usable to ~24 users; but long sessions once locked up all 20

The 48-turn slope first (24 and 28 users ran with a lowered compression trigger —
the reason follows):

| users | trigger | p50 | p95 | over 1 s | over 2 s | thinker p50 | rtf p10 |
|---|---|---|---|---|---|---|---|
| 13 | default | 758 | 988 | 4.0% | 0% | 154 | 1.30 |
| 16 | default | 802 | 1,071 | 12.2% | 0% | 177 | 1.27 |
| 20 | default | 873 | 1,219 | 25.3% | 0% | 210 | 1.22 |
| 24 | 20,480 | 991 | 1,755 | 48.7% | 2.4% | 265 | 1.14 |
| 28 | 16,384 | 1,141 | 2,252 | 68.8% | 9.6% | 312 | 1.06 |

The source of the climb is plain: even a still picture gets a quarter of its
frames forced through the filter, so history still thickens by ~850 tokens per
turn, and the price of a frame is proportional to how thick the history is
(section 16). The thinker share grows 154 → 312; the speech side gets squeezed
too (frame chewing and speech generation share one card). By the standard of
"keep p50 under a second and don't let the rtf margin fall through 1.1", **the
practical ceiling for this scenario sits around 24 users**.

**The long-session collective lock-up, and the one-parameter rescue.** 72 turns
× 20 users, all defaults: **all twenty** users time out (first timeouts fall in
turns 46–62, 17 of them in 46–55), and all 20 sessions get flagged "sampled zero output tokens" (once per stage, thinker and
talker, 40 log lines). The engine log states the cause plainly:

    waiting=0 skipped_waiting=20 running=0

Every session holds its own KV cache (the intermediate results the model keeps
for everything it has read, resident in GPU memory) while waiting for the next block of memory;
the pool (731,904 tokens on that 20-slot boot) is already full at 20 × ~37k,
nobody can yield, nobody can proceed. **Not a crash — a livelock**: the
preemption counter stays 0 throughout (every per-request dump line reads
preemptions=0), the engine loop parks over and over, and
only when users time out and disconnect does memory free up and the survivors
move again.

The key point: **compression never fired at all.** All 20
sessions had compression armed, but the trigger is **per session** — 49,152
tokens — while pool ÷ 20 users = 36,595. **The pool fills 25% before the trigger
is reached.** The trigger watches one user at a time; the pool belongs to everyone.

The rescue is exactly Xiao's "just reduce the context": drop the trigger below
pool ÷ users (trigger 24,576, window 8,192), same 72 turns × 20 users:

| | default trigger | trigger at 24,576 |
|---|---|---|
| completed | 1,007 (60 timeouts, 373 given up) | **1,440 / 1,440** |
| zero-output wedge log lines | 40 | **0** |
| invisible swaps | 0 | 40 (~2 per user) |
| p50 | 904 (completed turns) | 927 |

The cost is 23 ms of p50 (35 ms if compared only against the dead run's
pre-lock-up turns, whose p50 is 892). The curve shows the swap knocking the thickness back
down: turns 37–40 climb to 1,150, then fall back to ~900 and never run away
again.

The obvious next step (**not built**): make the trigger
min(0.75 × context limit, 0.75 × pool ÷ active sessions) — one line of formula,
and the table above is its evidence.

### Scenario 3: high-motion video — 13 users, bottlenecked on frame prefill, the step that reads new frames into the model (measured in section 16)

### The three scenarios side by side (13 users, same protocol)

| 13 users | audio only | near-still video | high-motion video |
|---|---|---|---|
| p50 | 633 | 758 | 1,168 |
| p95 | 677 | 988 | 3,535 |
| over 1 s | 0% | 4.0% | 63.6% |
| thinker p50 | 40 | 154 | 416 |
| speech p50 | 593 | 608 | 724 |

How to read it: audio-only and near-still video pay almost the same speech share
(593 / 608); high motion squeezes the speech side up a fifth as well (724).
**But the main difference across the columns is still the thinker share
(40 / 154 / 416)** — what the camera points at decides how many
frames the thinker chews per turn, and whether this engine is lightly or heavily
loaded.

### Fixed along the way

- mu_bench's engine-log path pointed at a file that stopped growing on 08-02, so
  every cell's engine-side probes since then were counting an empty slice (client
  metrics were unaffected). `MU_ENGINE_LOG` now points them at the live boot log,
  and the probe vocabulary matches what the code actually prints.
- The `counter_leak_clamped` guard fires occasionally under swap pressure (up to
  29 times in one cell), always clamped, zero consequences — but the source of
  the leak is still upstream, unfixed.

---

## 18. The first wall, pushed — CUDA graphs for the speech stage

*2026-08-03, config experiment following commit `1210772d`; a one-line change.*

Section 17's diagnosis said the first wall for audio-only is scheduling — the
talker launches hundreds of small kernels one by one per step
(`enforce_eager`), and has kernels running only 29% of the time. A diagnosis
this specific can be tested: turn eager off for stage 1 so vLLM records the
talker's step as one pre-built graph, launched once per step
(`deploy_mu_fp8_s32_graph.yaml`, differing from the 32-slot config by that one
line). code2wav was **deliberately left alone**: one variable at a time.

**Correctness before speed.** Text, audio durations, chunk cadence, all probes
normal; the stage-1 KV pool did not lose a single token (123,040); graphs cost
~200 MiB; boot takes 55 s longer (recording is a one-time cost). Four speech
samples (counting, a tongue twister, …) are saved under
`/data/zx/results/graph_ab/` — **the final verdict belongs to human ears**;
text gates cannot see speech-side defects.

**Results (same protocol as the eager cells):**

| audio-only | eager | CUDA graph |
|---|---|---|
| solo price (1 user p50) | 317 ms | **145** |
| ensemble price (32 users p50) | 702 | **367** |
| 32-user p95 | 798 | **497** |
| speech share p50 (32 users) | 652 | **303** |
| rtf p50 / p10 (32 users) | 1.42 / 1.23 | **3.62 / 2.65** |
| playback starvation p95 (32 users) | 268 ms | **19** |

| video, 13 users | eager | CUDA graph |
|---|---|---|
| near-still: p50 / over 1 s | 758 / 4.0% | **367 / 0.2%** |
| high-motion: p50 / over 1 s / over 2 s | 1,168 / 63.6% / 18.1% | **749 / 31.1% / 11.9%** |
| high-motion: p95 | 3,535 | 3,986 (not improved) |

**Four readings.**

1. **Both prices halved.** Thirty-two people in ensemble (367) now cost 50 ms
   more than one person solo used to (317); the speech side's fixed startup
   fell from ~600 ms to ~300.
2. **The 50–70-user scheduling wall extrapolated in section 17 has been pushed
   out of measurable range**: the rtf margin at 32 users rose from 1.23 to
   2.65. Finding the new wall requires more slots first — next time's work.
3. **The load intensifies itself — a property of closed loops.** Halve the
   latency and each turn cycle shortens, so the same 32 users produce 26% more
   turns per second; utilization therefore rose across the board (sm 58→93,
   DRAMA 0.17→0.27, power 184→226 W): the card is doing more work, not working
   harder per unit. One mechanism worth recording: faster steps → fewer streams
   speaking at once → thinner co-batches → more weight-read bytes per token —
   **trading bandwidth for latency**, an excellent trade on a card using a
   third of its bandwidth.
4. **The high-motion tail did not move** (p95 3,986), because it never lived on
   the speech side: it is the thinker's frame prefill and swap windows fighting
   for the card — the territory of the next wall section 17 queued up
   (priority scheduling). The median improvement (−36%) is entirely the speech
   half.

### The new wall shows itself: 48 users clean, 64 hit the speech stage's KV pool

With the wall pushed, slots went to 64 (`deploy_mu_fp8_s64_graph.yaml`, three
`max_num_seqs` lines changed and nothing else) and the audio ladder continued:

| audio-only (graph, 64 slots) | 48 users | 64 users |
|---|---|---|
| completed | **2,304 / 2,304** | 2,585, +15 timeouts, 472 given up, 112 connection errors |
| p50 / p95 | 484 / 644 | 596 / 830, **p99 = 57,028** |
| rtf margin p10 | 1.97 | 1.46 |
| zero-output wedges | 0 | 125 lines |
| nonzero preemption counters | 0 | **exactly 64, all on stage 1, one per session** |
| DRAMA / SMACT / power | 0.31 / 0.45 / 241 W | 0.31 / 0.44 / 236 W |

(The 26 over-1 s turns at 48 users are all turn 1 after the restart — cold
start again.)

The cause reads off in one line: at 64 users **bandwidth and compute are more
idle than at 48**, yet each of the 64 talker sessions got preempted exactly
once — the speech stage's own KV pool (116,384 tokens, one sixth of the
thinker's) ran out, preempted requests recompute their whole array, every step
lags the next, and the server finally kicked 56 users on "Idle timeout".
**A capacity wall again, and from the same family as section 17's**: the
thinker's pool has the compression trigger standing guard (the fix is a
pool-aware trigger), the speech pool had nobody guarding it until today.

So the audio-only capacity conclusion is revised: **this card, this config,
~48 users clean; between 48 and 64 the speech-side KV pool bites.** To go
higher, options by cost: give stage 1 a larger memory share (0.10 → 0.15,
~5.5 GB is free on the card), pool-aware admission control (same family as
the trigger formula), or FP8 KV for the talker (quality-sensitive — ears
before speed).

### More memory, plus a guard — what each fix bought

Both fixes, as decided: the memory split moved toward the speech stage
(stage-1 0.10 → 0.15, pool 116,384 → 365,344, ×3.1; the thinker gave up 8%),
and the speech stage got a guard from the same family as the compression
trigger — each turn the roll threshold is lowered to 0.75 × pool ÷ active
sessions, and a new session whose share would fall below the roll floor
(2,048) is refused at the door (`stage1_kv_pool_tokens`, off by default).
The session counter lives in exactly one wrapper function — section 13's
leaked-counter lesson.

64 users × 48 turns, three cells:

| | A memory only | B memory + guard (real pool) | C guard on, pool lied down to 116k |
|---|---|---|---|
| completed | **3,072/3,072** | **3,072/3,072** | 2,016 (= 42 users, clean) + 22 refused at the door |
| p50 / p95 / p99 | 614 / 804 / 966 | 614 / 921 / 1,951 | 452 / 1,449 / 1,607 |
| rolls | 0 | 104 | 140 |
| preempt / wedge / timeout | 0 / 0 / 0 | 0 / 0 / 0 | **0 / 0 / 0** |

Three sentences:

1. **Memory alone rescued 64 users within the 48-turn horizon** — and refuted
   the pre-run arithmetic: total demand ~512k exceeds 365k if every session's
   array stays resident, yet nothing burst, so the speech stage's residency is
   much looser than full-array. A wrong account, recorded: **the pool's true
   residency model is unmeasured, so the guard's 0.75 factor is conservative.**
2. **The guard buys expensive insurance when the pool is not actually tight**
   (B's p99 doubles): under its pessimistic model it starts rolling at ~turn
   26. For bounded horizons, admission-only is enough; the full guard earns
   its keep on unbounded sessions.
3. **Under real scarcity the guard wins outright** (C versus yesterday's death
   under identical conditions): 64 users against a 116k pool used to mean
   every session preempted, p99 57 s, 56 users kicked; with the guard it means
   42 users served cleanly (p99 1,607, over-2 s 0%), 22 refused at the door,
   zero preemptions, zero wedges. **The crash became a capacity boundary.**

### Pushing to 128: the wall is at ~90 users, and it is still the timeline

Guard off, 128 slots, audio-only ladder 80 → 96 → 112 → 128 (48 turns each):

| users | p50 | rtf margin p50 / p10 | starvation p95 | timeouts/preempts/wedges |
|---|---|---|---|---|
| 80 | 756 | 1.45 / 1.16 | 353 | 1 / 0 / 1 |
| 96 | 909 | 1.17 / **0.88** | 747 | 0 / 0 / 0 |
| 112 | 1,130 | **0.91** / 0.76 | 1,175 | 0 / 0 / 0 |
| 128 | 1,342 | 0.75 / 0.63 | **1,805 (max 8.6 s)** | 0 / 0 / 0 |

Resource fingerprint at the break (96–128 users): sm% pinned at 95–98, SMACT
only 0.54–0.59, DRAMA 0.39–0.43, power 276–294 of 600 W.

Three conclusions:

1. **It breaks by running out of breath mid-sentence, not by answering late.**
   TTFA's p99 is still 1.7 s at 128 users; what blows up is rtf — at ~90 users
   generation falls below playback speed and utterances stall for 1–2 s in the
   middle. The prebuffer protects the first sound, not the rest of it.
2. **What pins it is still the timeline, not the silicon.** With the time axis
   95% occupied, compute sits 40% idle, bandwidth 60% idle, power at half —
   the same disease as before the CUDA graphs, with the wall moved from
   ~50–70 users to ~90. The remaining organisation overheads, by name:
   code2wav still eager (the talker's old ailment), stage-1 async_scheduling
   still off, three stages handing the card back and forth.
3. **Both memory pools spectated** (zero preemptions, wedges, timeouts; the
   thinker pool at 128 × ~5k ≈ its ceiling never bit) — 128 users degrade
   smoothly into breathlessness rather than crashing, which is exactly the
   failure shape one wants.

The final audio-only account (this card, this config): **~48 users with
headroom, ~80 usable, ~90 starts gasping, 128 everyone gasps — and not one
crash on the way.**

---

## 19. Three locked doors, one key each — the video path gets its optimisations back

*2026-08-08, commits `d8ccfd48`, `baf0e711`, `9a6a0908` (plus `eea63d41` from the
colocation project). Section 18 halved the speech prices with CUDA graphs, and a
later experiment (async scheduling) bought another +51% — but only audio could
collect. Three defects kept every one of those wins away from video. Tonight all
three fell.*

**Door 1: vocoder CUDA graphs crashed on video.** The crash
(`split_with_sizes expects sum 88, got 81`) looked like the vocoder's batch
composition mishandling mixed chunk sizes. The real culprit was an accounting
mismatch two layers up: with graphs on, the runner pads the flat token stream to
the capture bucket (81 → 88, always to a multiple of 8) while the per-request
counts stay unpadded. Audio never crashed because uniform 16-token chunks land
on bucket sizes by luck. Fix: slice the stream to the declared total before
splitting, and complain loudly if it is ever SHORTER than declared. Verified on
the exact cell that used to kill the engine: 24 video users, 96/96 turns.

**Door 2: async scheduling killed the talker one second into any browser video
session.** With async on, the engine schedules step N+1 before step N's finishes
are known, leaving a −1 placeholder where the previous sampled token will be
backfilled. Video sessions produce prefill-only appends (frames, no query),
whose talker segments are boundary-capped: they finish on their FIRST sample, so
nothing ever backfills the placeholder — and the −1 walks into the codec
embedding (vocabulary 3,072) and dies with a device-side assert. Audio never
makes prefill-only segments, which is why the audio ladder stayed green for a
whole night while one real browser call died in a second. A diagnostic guard
caught the value red-handed (`values=[-1]`), and clamping at the embedding alone
was not enough — the MTP code predictor consumes the same ids. Fix: clamp
negatives once at the runner's input chokepoint (a no-op for text stages), so
every consumer is covered. Verified with the browser-recipe driver: engine dead
after turn 1 → 6/6 turns, health 200.

**Door 3: the colocated speech pair ran video glacially** (13 users, fifteen
minutes, zero turns finished, no errors). Not re-diagnosed — re-TESTED on
tonight's tree: 52/52 turns, p50 483 ms. Cured by one of the intervening fixes
(most plausibly door 2's clamp removing the boundary-segment pathology, or the
MoE-workspace guard); attribution is honestly unknown, the reproduction is
honestly gone.

**What the unlocked configuration measures** (async + talker graphs + vocoder
graphs, the first time video runs the full stack):

| cell | best before | tonight | change |
|---|---|---|---|
| video 13 users, p50 | 749 ms | **457 ms** | −39% |
| video 13 users, over 1 s | 31.1% | **11.5%** | tail cut by ⅔ |
| video 13 users, p95 | ~3,986 ms | **2,200 ms** | −45% |
| audio 32 users, p50 | 307 ms | **297 ms** | best yet |
| browser video session on async | dead in ~1 s | 4/4 turns | fixed |

180 turns, zero timeouts, all five quality probes zero, on both the separate-
process and the colocated boot.

**Still open, found tonight:** stage 1 deterministically ships some segments
with a 16k+8-token tail (half a codec frame; 44 occurrences per benchmark run,
identical across boots since at least 08-04). Those chunks decode scrambled for
their tail frames. Pre-existing, bounded, now loudly logged — the producer hunt
is queued.

---

## 20. Three engines, one process — the platform lands, the GIL keeps the crown

*2026-08-08, commits `f2b1edeb`, `5aa565df`, `6221aa51`; design in
`benchmarks/live_agent/design/TRI_COLOCATION.md`.*

Section 19 cleared the video path; this section merges the THINKER into the
process that already housed the talker and the vocoder, so all three stages
share one CUDA context — the physical prerequisite for a single scheduler that
owns the whole timeline (the mixed-tenancy chapter's platform).

**Two things had to be true first.** (1) Memory accounting: the per-PID NVML
diff charges every engine for its housemates — the second KV-bearing engine's
pool computes to ~zero. Colocated mode now uses the per-engine profiling
fallback; measured pools in one process: talker 392,720 tokens, thinker
681,968 — both honest. (2) The MoE scratch arena: vllm's WorkspaceManager
hands the SAME bytes to every caller, safe with one MoE engine, silent
corruption with two. Arenas are now keyed by each engine's private CUDA
stream (four observed, the thinker's at 512 MB).

**It booted on the first try** — 155 s, one process at 93.9 GB, both guests
handshaking in order (the parent awaits guests in exactly the order the child
builds them; a mismatch is a silent hang, so both sides derive from one list).

**The scoreboard is honest and mixed.** Correctness: every gate green (audio
32 users, handheld 13 users, browser-recipe driver, five probes at zero,
throughout). Performance vs the separate-process config: audio p50 349 vs
297 ms with a 3× worse tail; video p50 648 vs 457, though shipping payload
snapshots as on-device clones (instead of pinned-CPU copies) bought the video
tail back — over-1-second turns 40% → 17.3%, p95 1702 → 1363. The famous
291 MB copy turned out to be paid twice upstream of where the study pointed:
once in the async output snapshot (now removed under full colocation) and once
in a forward-time staging that the prefix-cache pooler REQUIRES on CPU — that
redesign is the next dig site.

**Verdict.** For serving today, the separate-process async config keeps the
crown; the residual gap is the GIL (three Python busy loops, ~20 threads, one
interpreter) — the same structural ceiling the speech pair hit, and the trio
already carries every remedy that saved the pair. Moving the engine loop off
Python is thesis-scale work. What tri-colocation delivers TODAY is the
research platform: one process, unified visibility, partitioned arenas,
honest per-engine budgets — the room where the mixed-tenancy experiments can
now be run.

---

## 21. Where the multi-user p99 comes from — the speech stage, not the thinker

*2026-08-08, commit `ef30abf9`; data under `/data/zx/results/p99aud_none_u*`,
analysis in `benchmarks/live_agent/analysis/p99_attribution.py`.*

The question: audio-only users, each holding a long session. As the user
count grows, the worst turns (p99 — the latency of the slowest 1 in 100
turns) get slower. Slower WHERE?

**The setup.** Six cells: 1, 4, 8, 16, 24, 32 users, each user one 30-turn
session-scoped conversation, fresh engine per cell, two-process default
config. New instrument: the bench now stamps every turn with absolute times,
so for any turn we can count how many OTHER turns were still waiting for
their first sound, and how many were mid-playback, at the moment it arrived.
That splits "I queued behind others" from "everything got slower" — the two
stories a percentile alone cannot tell apart.

**First finding: the scary p99 was a boot artifact.** Every turn slower than
800 ms in the whole ladder — all 32 of them across six cells — was a FIRST
turn, arriving within ~2.6 s of the first query after an engine boot. The
first query on a cold engine pays ~3 s on the thinker side; everyone who
arrives behind it queues, and the delay drains linearly (at 32 users: the
t+0 arrival pays 2,961 ms of thinker time, the t+2.3 s arrival pays 682 ms).
The second session on the same engine pays nothing. This is a once-per-boot
transient, not a scaling law — and if left in the data it makes p99
NON-monotonic in user count (24 users "beat" 16), which is how it gave
itself away. A warm-up query at boot would erase it.

**Steady state (turn ≥ 2): p99 grows 3× from 1 to 32 users, and ~85% of
the growth is the speech side.** Per turn, thinker side = query → first
text token; speech side = first text → first audio (the talker's first
codec batch plus the vocoder's first chunk). The tail's excess over the
cell median, split within each turn and then averaged:

| users | turns | TTFA p50 | p95 | p99 | thinker p50/p99 | speech p50/p99 | speech share of tail excess |
|------:|------:|---------:|----:|----:|----------------:|---------------:|---------------------------:|
| 1     | 58    | 130      | 140 | 146 | 30 / 33         | 99 / 115       | — (no real tail)            |
| 4     | 116   | 134      | 233 | 248 | 30 / 75         | 105 / 187      | 67%                         |
| 8     | 232   | 142      | 258 | 291 | 32 / 71         | 110 / 233      | 81%                         |
| 16    | 464   | 220      | 321 | 362 | 51 / 85         | 168 / 288      | 85%                         |
| 24    | 696   | 262      | 365 | 403 | 55 / 95         | 203 / 327      | 88%                         |
| 32    | 928   | 296      | 400 | 435 | 61 / 100        | 235 / 360      | 84%                         |

The thinker never becomes the problem: its p99 stays at or under 100 ms all
the way to 32 users. The speech side triples.

**Why the speech side: it is serving everyone else's playback while you
wait for your first sound.** At 32 users, a turn that arrives when nobody
is mid-playback pays 158 ms of speech-side time; with one concurrent stream
221 ms; with four or more, ~260 ms, and it saturates there — the batching
is sublinear, there is no hard serialization. Tail turns are exactly the
unlucky arrivals: the average tail-turn arrives with twice as many
concurrent waiters and streams as the average turn.

**Two exonerations.** (1) Context growth is innocent: within a cell, turns
2–10, 11–20 and 21–30 have flat p50 AND flat p99 (32 users: 296/307/289 ms
p50) — the long session itself costs nothing, consistent with §13. (2) The
run was clean: zero timeouts, zero session rolls, zero wedges across 2,532
turns; the parked-counter guard clamped 2–6 leaks per cell, which is the
known §12 leak being contained, not a new fault.

**What this buys.** The lever for multi-user p99 is not the thinker and not
memory — it is priority inside the speech stage: a turn's FIRST audio chunk
competing against other users' mid-stream chunks. Mid-stream work has slack
(delivery runs 5–10× faster than playback even at 32 users); first-chunk
work is the user-visible pause. A first-chunk-first scheduling policy in
stages 1–2 is the direct attack, and it is the same shape as the
speech-over-text priority item already on the list.

---

## 22. A 4k sliding window on the talker — the 8-user memory wall was self-inflicted

*2026-08-08; config `benchmarks/live_agent/web_client/deploy_mu_fp8_s128_async_sw4k.yaml`,
data under `/data/zx/results/sw4k_verify_u{1,8}`, listening samples in
`/data/zx/results/sw4k_verify_u1/wavs/`.*

The question from the capacity ledger: at the 45k roll line, one audio user
parks 45,000 × 20 KB ≈ 0.88 GiB in the talker's KV pool, so the 7.49 GiB
pool holds 8 users. But the talker does not NEED 45k of history — its job
is to continue the voice of the last few sentences, and every session roll
already proves it (we throw the whole speech history away and reseed with 8
turns of text, and nobody hears it). Can the talker keep only a 4,096-token
attention window while the session, the roll line, and the user-visible
context stay exactly as they are?

**Yes, and it is one yaml stanza.** No model code changes: vLLM's Attention
constructor takes a model-level window from the engine's cache config, and
the V1 KV manager frees out-of-window blocks WHILE a request decodes —
block tables stay position-indexed (freed slots become null blocks), so
positions keep growing and M-RoPE is untouched. The stanza, on stage 1
only:

```yaml
hf_overrides:
  talker_config:
    text_config:
      sliding_window: 4096
```

**The trap that makes this worth writing down:** a FLAT
`hf_overrides: {sliding_window: 4096}` lands on the top-level Omni wrapper
config and is **silently ignored** — no error, no warning, no savings. The
override must be nested to `talker_config.text_config`, because that is
what stage 1 resolves as its text config. We verified the landing offline
(config parse only) before burning a boot.

**Boot receipts (same 7.49 GiB pool, same everything else):**

| stage-1 boot line                        | full attention | 4k window |
|------------------------------------------|---------------:|----------:|
| Maximum concurrency for 65,536 tokens    | 5.99×          | **19.16×** |
| GPU KV cache size (tokens)               | 392,720        | 1,255,722 |
| thinker pool (untouched)                 | 677,968        | 677,968   |

Two reading rules. First, 19.16× is the GUARANTEED floor: the admission
bound per request is window 4,095 + 16,384 of scheduler in-flight allowance
(2 × 8,192 batched tokens) ≈ 20,479 tokens — the pool admits 19 users even
if every one of them is mid-prefill of a maximal batch simultaneously. In
steady state a session holds only ~window, so the realistic ceiling is
392,720 physical tokens / ~4k ≈ **~90 users**, up from 8. Second, the "GPU
KV cache size: 1,255,722 tokens" line is EQUIVALENT capacity under the
sliding window, not physical bytes — the pool is still 7.49 GiB at 20
KB/token; dividing those and getting 6.4 KB/token means you misread the
line, not that KV got cheaper.

**Mechanics verified across the boundary.** One user, 40 turns, one
session; cumulative model speech crosses 4,096 talker tokens at turn 25:

- TTFA p50 129 ms on turns 2–20 (before the window fills), 132 ms on turns
  25–40 (window sliding every step). Flat.
- Text stays correct and non-repetitive through turn 40; audio seconds per
  turn flat (1.85 s before / 2.02 s after); rtf 9.1–11.7 throughout.
- Engine probes all zero; the only >1 s turn was the once-per-boot cold
  start (§21).

Eight users co-batching (the interaction sliding window × CUDA graphs ×
async scheduling, which one user cannot exercise): 80/80 turns, p50
138.5 ms / p95 276.2 ms vs the full-attention baseline's 145.8 / 275.9 —
the same numbers. `counter_leak_clamped: 4` is the §12 guard containing
the known leak, in its normal 2–6 band.

**What is NOT settled: how it sounds.** The checkpoint was trained
full-attention; once the window slides, the earliest tokens — including
whatever anchoring role they play — are gone, and text LLMs have a history
of degrading exactly there (the attention-sink lesson). The numbers above
say mechanics and latency are clean; only ears can pass the audio. The 40
per-turn wavs are on disk (the boundary is t25; compare t02–t05 against
t25–t40), and the sliding-window engine is left running for live listening.
If 4k fails the ear test, 8k/16k still buy 2.5–6× capacity.

**What this does not change:** the 65,536 stage-1 array wall and the 45k
roll cadence. Positions keep growing; only the KV residency plateaus. The
sliding window converts the talker pool from the binding capacity wall
into a bystander — after this, the walls that remain are slots
(max_num_seqs 128) and the speech-stage latency slope of §21.

**Addendum, same evening — talker KV to FP8, and the share moves to the
thinker** (`deploy_mu_sw4k_kvfp8.yaml`, data
`/data/zx/results/sw4k_kvfp8_verify_u{1,8}`). Two more ledger moves, no
compute touched: stage-1 `kv_cache_dtype: fp8` (weights stay bf16 — the
talker feeds the vocoder, and the FP8-weights trap of the config header
was the weights path), and the freed share to the thinker: 0.67/0.15/0.08
→ **0.72/0.10/0.08**. Xiao asked for thinker 0.80; it does not fit inside
the 0.90 shared-card envelope: the talker process carries ~7 GB of fixed
overhead, so share cuts come straight out of its pool — at 0.08 the pool
is ~0.6 GiB (~16 users even with window+FP8), worse than before §22.
0.72 is the thinker maximum with a functional talker.

Boot receipts: thinker pool 31.04 → **35.78 GiB** (677,968 → 781,696
tokens, +15%); talker physical pool 2.74 GiB with guaranteed concurrency
**14.03×** — and that line is itself the FP8 receipt: at bf16 a 2.74 GiB
pool could only guarantee ~7×; 14.03× is only arithmetic at 10 KB/token
(287k physical tokens → ~71 users at the 4k steady-state window).
Functional, same protocol as §22: u1 × 40 crossing the boundary at t25 —
TTFA p50 128 pre / 131 ms post, audio seconds 1.96/1.95, no repetition
through t40, probes zero; u8 co-batch 80/80, p50 161.9 / p95 268.8 (p95
the best of the three arms; the p50 sits inside the baseline's per-user
spread of 137–195, one cell cannot separate it from noise). The ear test
now covers TWO stacked variables: `sw4k_verify_u1/wavs` is the
bf16+window arm, `sw4k_kvfp8_verify_u1/wavs` the FP8+window arm — listen
to both against full-attention memory before either becomes the default.

---

## 23. Where audio bursts past one second — the tail at 96 users, the median at 128, all of it the speech stage

*2026-08-08; ladder config `deploy_mu_sw4k_kvfp8_t15.yaml` (talker share
back to 0.15 so the pool CANNOT be the culprit), data
`/data/zx/results/burst_none_u{32,48,64,96,128}`, full detail
`/data/zx/results/burst_attribution.json`.*

The question: on the new default stack (talker 4k window + FP8 KV), at how
many audio users does latency burst past 1 s, and what is the mechanism?
Design choice up front: the ladder runs with the talker share at 0.15 —
785k FP8 tokens, 6.1k per user at 128 users, ~30% above even uncapped
30-turn growth — so pool exhaustion is excluded BY CONSTRUCTION and
whatever bursts is compute or scheduling.

**The curve (steady state, turn ≥ 2; each cell 30-turn sessions, fresh
engine):**

| users | turns | TTFA p50 | p95 | p99 | >1s % | thinker p50/p99 | speech p50/p99 | rtf p50/p10 | stalls >0.3s |
|------:|------:|---------:|----:|----:|------:|----------------:|---------------:|------------:|-------------:|
| 32    | 928   | 296      | 404 | 480 | 0     | 60 / 97         | 234 / 402      | 5.27 / 3.72 | 0%           |
| 48    | 1,392 | 375      | 486 | 536 | 0     | 69 / 107        | 304 / 470      | 3.79 / 2.86 | 0%           |
| 64    | 1,856 | 477      | 610 | 668 | 0     | 79 / 120        | 396 / 575      | 2.88 / 2.23 | 0%           |
| 96    | 2,784 | 711      | 920 | **1,064** | 2.5 | 90 / 139   | 618 / 970      | 1.75 / 1.40 | 1.6%         |
| 128   | 3,712 | 971      | 1,265 | 1,523 | **42.1** | 100 / 160 | 870 / 1,427   | 1.19 / 0.96 | 29.2%        |

**Answer one — the burst line.** The tail first crosses 1 s at **96
users** (p99 1,064 ms, 2.5% of turns over). At **128** it is no longer a
tail: 42% of ALL turns exceed 1 s and the median itself sits at 971 ms.
64 users is the last cleanly healthy cell (p99 668 ms, zero turns over
1 s). Hygiene: zero timeouts, zero rolls, zero preemptions across 10,752
turns — even the 128-user cell completes 3,840/3,840. The pre-FP8 stack
died logically at 8 users; this one degrades smoothly to the slot cap.

**Answer two — where.** Entirely the speech stage. The >1 s turns'
excess over the cell median splits thinker/speech = -0.01/1.01 at 96 and
0.02/0.99 at 128; the thinker's p99 stays at or under 160 ms even at 128
users. (The raw table without the turn-1 cut shows thinker p99 2,389 ms
at 128 — that is the §21 once-per-boot cold-start drain again, 128
turn-1s deep. Excluded, as always.)

**Answer three — the mechanism is throughput exhaustion, not an event.**
Real-time factor (delivered audio seconds per wall second) decays
smoothly: 5.27 → 3.79 → 2.88 → 1.75 → 1.19 at the median. At 128 users
the p10 dips under 1.0 — one turn in ten delivers SLOWER than playback,
and 29% of turns stall mid-playback for >0.3 s. Nothing broke; the two
speech stages simply ran out of compute, and the first audio chunk of
each turn queues behind everyone else's mid-stream chunks (a tail
arrival at 128 users finds ~37 other turns mid-TTFA). Per added user the
median speech stage pays ~6.6 ms; the thinker pays 0.4 ms.

**What this fixes in the ledger.** The capacity story for audio on this
card is now measured end to end: memory stopped being the wall in §22,
and the wall that remains is speech-stage compute at ~96 users (tail) /
~128 (median). The §21 lever — first-chunk-first priority in stages 1–2
— has real room at 96 (rtf margin 1.75 = 43% of delivery time is slack
that mid-stream chunks can donate) and none at 128 (p10 rtf < 1: there
is nothing left to reshuffle; only more throughput helps — co-batch
efficiency, vocoder batching, the engine-loop/GIL dig sites).

**Addendum — "is the SM exploding?" No. The 128-user roof is software,
not silicon.** Xiao asked whether the saturation is actual SM
saturation. nvidia-smi's utilization cannot answer (it reads ~96% from
32 users onward — a kernel is almost always running SOMEWHERE — while
rtf collapsed 5.27 → 1.19 underneath it). DCGM on a live re-run of both
rungs, same boot, jkim idle (device SMACT 0.00 before the probe):

| window (10 turns) | TTFA p50 | rtf p50 | SMACT | SMOCC | TENSO | DRAMA |
|---|---:|---:|---:|---:|---:|---:|
| 32 users  | 295 ms | 5.03 | 0.42 | 0.05 | 0.05 | 0.29 |
| 128 users | 964 ms | 1.20 | **0.65** | **0.16** | 0.13 | **0.46** |

At the exact operating point where latency bursts, the SMs sit idle 35%
of the time, warp occupancy is 16%, and the memory bus is half used.
Users went ×4 (aggregate audio production ~×4) while SMACT rose only
×1.55 — co-batching amortizes beautifully; the hardware is nowhere near
its roof. The service-rate ceiling that pins rtf at ~1.19 is therefore
the SOFTWARE path between kernels: Python engine loops and the GIL
shared by the colocated speech pair, per-step launch/organisation gaps,
connector handoffs. Same diagnosis as the pre-graphs talker (SMACT 0.23
at 32 users) one tier up. Consequence: capacity beyond 128 audio users
does not require more FLOPs — it requires the engine-loop/GIL dig sites;
the silicon is already paid for and idle.

**Second addendum — inside the speech side: talker vs code2wav now split
the bill evenly, and a natural experiment fingers the handoffs.** Xiao
asked which half of the speech side binds: token generation (talker) or
token→waveform (code2wav). py-spy is blocked on this host
(ptrace_scope=1, no sudo) and both stages share one process in pair
mode, so the split was measured the §19 way: boot the SAME t15 yaml with
`VLLM_OMNI_COLOCATE_STAGES=""` (three processes) and read per-process
sm% under a 128-user probe:

| stage (128 users, 3-proc) | sm% p50 | sm% p90 | mem% p50 |
|---|---:|---:|---:|
| thinker  | 37 | 58 | 16 |
| talker   | 27 | 48 | 12 |
| code2wav | 27 | 53 | 12 |

Two findings. First, **code2wav is no longer the 4% bystander of §19**
(32 users, pre-graphs): its work scales linearly with audio seconds
produced (~×4 from 32→128 users) while graphs made the talker cheaper,
and at 128 users the two speech stages cost the same GPU time. Neither
saturates anything (device SMACT 0.65 stands). Second, the probe itself
is a natural experiment: the SAME work, re-arranged from pair-colocated
into three processes, drops rtf 1.06 → 0.85 and pushes >1s turns 65% →
90% — re-plumbing WHO HANDS WHAT TO WHOM costs 20% of throughput while
the GPU math is untouched. The handoff/organisation layer is not just
where the slack lives; it is itself a first-order term. (Caveat: probe
cells ran hotter than the ladder — pmon+observer load — compare shapes,
not absolute p50s. The thinker's GPU share also keeps growing with
users, 6% → 15% → 37% at 16/32/128, but its latency contribution stays
≤160 ms p99 — it co-batches well.) For the first-audio path both stages
sit in series: first-chunk-first priority (#19) must cover stages 1 AND
2 to move TTFA.

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
