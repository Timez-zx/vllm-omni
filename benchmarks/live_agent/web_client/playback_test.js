#!/usr/bin/env node
// Regression test for the bug that made every turn after the first one silent.
//
//   node benchmarks/live_agent/web_client/playback_test.js
//
// The failure was not audible as a failure: the first reply made a fraction of a
// second of sound and later replies produced nothing, with no error anywhere.
// Two causes in playback_worklet.js, and section 2 below is the record of finding
// out that NEITHER IS SUFFICIENT ALONE -- each was first asserted to be, and the
// assertions failed:
//
//   1. the prebuffer threshold (250 ms) was LARGER than the delivery it had to start
//      on. Alone this only delays the start: the audio stays queued and the next
//      delivery pushes the total over the threshold, so playback runs one behind.
//   2. an underrun re-armed the prebuffer. Alone, at a threshold below one delivery,
//      this is harmless -- every delivery clears it again.
//
// Together they are the bug: once the threshold exceeds one delivery, re-arming means
// each delivery must be paid for out of the NEXT one, so deliveries fall silent in
// turn and what plays is the previous one. Fixing one and not the other would have
// looked like a fix and left it broken, which is the reason this file exists.
//
// The arrival pattern here -- one small delivery, then a long gap -- is not a museum
// piece. The server sends a deliberately small first granule so speech starts early
// (measured 0.217 s, from `initial_codec_chunk_frames: 4`) and the 2 s granules after
// it can be a moment behind, so the first delta of every turn still lands in exactly
// this shape. What HAS changed since these tests were written is the rest of the turn:
// a truncation bug capped each reply at that first granule, and now a turn delivers
// its whole reply (measured 13.58 s of 13.66 s produced), in about 8 deltas.
//
// Neither is visible to selftest.py, which checks the protocol, nor to probe.py,
// which never plays anything. A browser was the only thing that could catch it,
// and a browser is exactly what neither of those has. So run the real worklet
// under Node with the two AudioWorklet globals stubbed.

'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const RATE = 24000;                     // the server's audio.delta rate
const BLOCK = 128;                      // AudioWorklet render quantum
const TURN_MS = 220;                    // ~ the small first granule (0.217 s measured)
const GAP_MS = 3000;                    // a hostile gap before the next delivery
const TURNS = 3;

// ---------------------------------------------------------------- the sandbox
function loadProcessor(file) {
  let registered = null;
  const sandbox = {
    AudioWorkletProcessor: class {
      constructor() {
        this.port = {
          messages: [],
          onmessage: null,
          postMessage(msg) { this.messages.push(msg); },
        };
      }
    },
    registerProcessor: (name, cls) => { registered = { name, cls }; },
    sampleRate: RATE,
    currentFrame: 0,
    Float32Array, Math, console,
  };
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(file, 'utf8'), sandbox, { filename: file });
  if (!registered) throw new Error(`${file} never called registerProcessor`);
  return registered;
}

// ------------------------------------------------------------------ the drive
function runSession(prebufferMs, reArmOnUnderrun, turnMsList) {
  const turns = turnMsList || new Array(TURNS).fill(TURN_MS);
  const file = path.join(__dirname, 'app', 'static', 'playback_worklet.js');
  const { cls } = loadProcessor(file);
  const node = new cls({ processorOptions: { prebufferFrames: Math.round(RATE * prebufferMs / 1000) } });

  // Reproduce the historical bug on demand, to prove the test can fail.
  if (reArmOnUnderrun) {
    const original = node.process.bind(node);
    node.process = (inputs, outputs) => {
      const r = original(inputs, outputs);
      const dry = node.queue.length === 0 && node.buffered === 0;
      if (dry && node.started) node.started = false;
      return r;
    };
  }

  // outputs[bus][channel] -- one bus, one channel. Nesting this one level too
  // shallow makes `outputs[0][0]` a number, every write a silent no-op, and every
  // assertion below fail for a reason that has nothing to do with the worklet.
  const outputs = [[new Float32Array(BLOCK)]];
  const out = outputs[0];
  const perTurn = [];
  const send = (frames) => {
    // app.js posts one message per response.audio.delta.
    const pcm = new Float32Array(frames);
    for (let i = 0; i < frames; i += 1) pcm[i] = 0.5;   // non-zero: audible
    node.port.onmessage({ data: { type: 'samples', pcm: pcm.buffer } });
  };
  const render = (ms) => {
    let nonZero = 0;
    const blocks = Math.round(RATE * ms / 1000 / BLOCK);
    for (let b = 0; b < blocks; b += 1) {
      out[0].fill(0);
      node.process([], outputs);
      for (let i = 0; i < BLOCK; i += 1) if (out[0][i] !== 0) nonZero += 1;
    }
    return nonZero;
  };

  for (const turnMs of turns) {
    send(Math.round(RATE * turnMs / 1000));
    // Render generously past the turn so nothing is blamed on a short window.
    let played = render(turnMs * 2);
    played += render(GAP_MS);                      // the silent gap between turns
    perTurn.push(played);
  }
  return { perTurn, underruns: node.underruns };
}

// ------------------------------------------------------------------- checking
let failures = 0;
function check(name, ok, detail) {
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  -- ${detail}` : ''}`);
  if (!ok) failures += 1;
}

const expected = Math.round(RATE * TURN_MS / 1000);

console.log('1. the shipped worklet, at the shipped prebuffer');
const live = runSession(60, false);
live.perTurn.forEach((played, t) => {
  check(`turn ${t} played audio`, played >= expected * 0.99,
        `${played}/${expected} frames`);
});
check('the gap between turns is not treated as fatal',
      live.perTurn[TURNS - 1] >= expected * 0.99,
      `underruns during the run: ${live.underruns} (expected: many, they are normal)`);

// This section is what corrected the diagnosis. Each cause was first asserted to
// be sufficient on its own; neither was. Running them separately is what showed
// that -- and the reported symptom needs both, which is why the fix had to be both.
console.log('\n2. the historical causes, separated');

const tooBig = runSession(250, false);
check('an oversized prebuffer alone loses only the FIRST turn, then runs a turn behind',
      tooBig.perTurn[0] === 0 && tooBig.perTurn[1] > expected,
      `frames per turn: [${tooBig.perTurn}] -- turn 0 stays queued below the 250 ms `
      + `threshold, then turn 1 pushes the total over it and both play at once`);

const reArm = runSession(60, true);
check('re-arming alone is harmless while the prebuffer is smaller than a turn',
      reArm.perTurn.every((p) => p >= expected * 0.99),
      `frames per turn: [${reArm.perTurn}] -- re-armed at 60 ms, and every turn clears 60 ms`);

// The reported shape: "first answer had some sound, then no response at all".
// It needs the oversized prebuffer AND the re-arm, and a first turn long enough
// to clear the threshold on its own -- which the server log showed, turn 0 having
// arrived as 3 chunks where later turns were 2.
const reported = runSession(250, true, [330, TURN_MS, TURN_MS, TURN_MS, TURN_MS]);
const later = reported.perTurn.slice(1);
check('both together reproduce what was reported',
      reported.perTurn[0] > 0
      && later.some((p) => p === 0)
      && !later.every((p) => Math.abs(p - expected) < BLOCK),
      `frames per turn: [${reported.perTurn}] -- the long first turn clears 250 ms; the `
      + `re-arm then demands 250 ms of every 220 ms delivery, so deliveries fall silent and what `
      + `does play is the PREVIOUS turn's reply, two turns' worth at a time`);



// ---------------------------------------------------------------------------
// 3. the startup stall, replayed from the MEASURED arrival schedule
//
// The tests above deliver a chunk and then render; they cannot express "the next
// chunk arrives 1.37 s from now", which is the whole of what a listener complained
// about. This drives the worklet against wall-clock arrivals instead, taken from
// audio_timeline.py against the live server:
//
//     delta 0   0.35 s   0.217 s of audio
//     delta 1   1.70 s   2.000 s
//     delta 2+  every ~1.28 s, 2.000 s each
//
// 0.217 s of audio cannot cover the 1.34 s the 2 s granule takes to make, so a
// start on delta 0 runs dry ~1.1 s -- one word in, every single turn.
const MEASURED = [
  { at: 350, audio: 217 },
  { at: 1700, audio: 2000 },
  { at: 2980, audio: 2000 },
  { at: 4260, audio: 2000 },
  { at: 5540, audio: 2000 },
];

function runSchedule(prebufferMs, schedule, { startNowAtEnd = false } = {}) {
  const file = path.join(__dirname, 'app', 'static', 'playback_worklet.js');
  const { cls } = loadProcessor(file);
  const node = new cls({ processorOptions: { prebufferFrames: Math.round(RATE * prebufferMs / 1000) } });
  const outputs = [[new Float32Array(BLOCK)]];
  const out = outputs[0];

  const lastAt = schedule[schedule.length - 1].at + schedule[schedule.length - 1].audio;
  const totalBlocks = Math.ceil(RATE * (lastAt + 2000) / 1000 / BLOCK);
  let delivered = 0, startedAtMs = null, playing = 0;
  const gaps = [];               // [{ atMs, durMs }] silence AFTER playback began

  for (let b = 0; b < totalBlocks; b += 1) {
    const nowMs = (b * BLOCK / RATE) * 1000;
    while (delivered < schedule.length && schedule[delivered].at <= nowMs) {
      const frames = Math.round(RATE * schedule[delivered].audio / 1000);
      const pcm = new Float32Array(frames).fill(0.5);
      node.port.onmessage({ data: { type: 'samples', pcm: pcm.buffer } });
      delivered += 1;
      if (startNowAtEnd && delivered === schedule.length) {
        node.port.onmessage({ data: { type: 'start_now' } });
      }
    }
    out[0].fill(0);
    node.process([], outputs);
    let nonZero = 0;
    for (let i = 0; i < BLOCK; i += 1) if (out[0][i] !== 0) nonZero += 1;
    if (nonZero > 0) {
      if (startedAtMs === null) startedAtMs = nowMs;
      playing += 1;
    } else if (startedAtMs !== null && delivered < schedule.length) {
      // Silence while more audio is still to come is a stall a listener hears.
      // Trailing silence after the last chunk is just the end of the reply.
      const blockMs = (BLOCK / RATE) * 1000;
      const prev = gaps[gaps.length - 1];
      if (prev && Math.abs(prev.atMs + prev.durMs - nowMs) < blockMs * 1.5) prev.durMs += blockMs;
      else gaps.push({ atMs: nowMs, durMs: blockMs });
    }
  }
  return { startedAtMs, gaps, playedMs: (playing * BLOCK / RATE) * 1000 };
}

console.log('\n3. the startup stall, on the measured arrival schedule');

const fast = runSchedule(60, MEASURED);
const worst = fast.gaps.reduce((m, g) => Math.max(m, g.durMs), 0);
check('starting on delta 0 stalls about a second, one word in',
      fast.startedAtMs < 500 && worst > 800,
      `starts ${fast.startedAtMs.toFixed(0)} ms, worst stall ${worst.toFixed(0)} ms `
      + `at ${(fast.gaps[0] ? fast.gaps[0].atMs : 0).toFixed(0)} ms -- this is what was reported`);

const smooth = runSchedule(1400, MEASURED);
check('waiting for delta 1 removes the stall entirely',
      smooth.gaps.length === 0 && smooth.startedAtMs >= 1700,
      `starts ${smooth.startedAtMs.toFixed(0)} ms, ${smooth.gaps.length} stall(s) -- `
      + `the cost is ${(smooth.startedAtMs - fast.startedAtMs).toFixed(0)} ms of start latency`);

// A reply shorter than the smooth target must still play. Without start_now it would
// sit in the buffer forever, which would be a worse bug than the stutter.
const shortReply = [{ at: 350, audio: 217 }, { at: 1700, audio: 400 }];
const stuck = runSchedule(1400, shortReply);
check('a reply shorter than the target would never play without start_now',
      stuck.startedAtMs === null,
      'confirms the escape hatch is load-bearing, not decorative');
const released = runSchedule(1400, shortReply, { startNowAtEnd: true });
check('response.audio.done releases it', released.startedAtMs !== null
      && released.playedMs > 500,
      `starts ${released.startedAtMs === null ? 'never' : released.startedAtMs.toFixed(0) + ' ms'}, `
      + `plays ${released.playedMs.toFixed(0)} ms of the 617 ms delivered`);


// ---------------------------------------------------------------------------
// 4. MULTIPLE turns in one call, which is where the smooth start actually broke
//
// Sections 1-3 and audio_timeline.py all drive a single turn, and every one of them
// passed while a listener heard a stutter on every reply after the first. Two faults
// compounded, and each alone was enough:
//
//   * `start_now` at the end of a turn zeroed the threshold PERMANENTLY -- the worklet
//     node is created once per CALL, not per turn;
//   * `started` stays true once playback has begun (correctly -- re-arming it on
//     underrun is what latched the whole session silent earlier), so the gate is not
//     consulted again at all.
//
// So the fix needs an explicit per-turn re-arm, and this section is the only thing
// here that can tell whether it works: it replays turn boundaries, not just chunks.
function runTurns(prebufferMs, turnCount) {
  const file = path.join(__dirname, 'app', 'static', 'playback_worklet.js');
  const { cls } = loadProcessor(file);
  const node = new cls({ processorOptions: { prebufferFrames: Math.round(RATE * prebufferMs / 1000) } });
  const outputs = [[new Float32Array(BLOCK)]];
  const out = outputs[0];
  const prebufFrames = Math.round(RATE * prebufferMs / 1000);
  const blockMs = (BLOCK / RATE) * 1000;
  const perTurn = [];

  const render = () => {
    out[0].fill(0);
    node.process([], outputs);
    for (let i = 0; i < BLOCK; i += 1) if (out[0][i] !== 0) return true;
    return false;
  };

  for (let t = 0; t < turnCount; t += 1) {
    // app.js posts this on response.start.
    node.port.onmessage({ data: { type: 'rearm', frames: prebufFrames } });

    let nowMs = 0, startedAt = null, stalls = 0, dry = 0, delivered = 0;
    // Run until the whole reply has been delivered AND played out, so the next turn
    // begins from silence -- which is what actually happens, since the user cannot
    // trigger the next turn until this reply has finished.
    while (delivered < MEASURED.length || node.buffered > 0) {
      while (delivered < MEASURED.length && MEASURED[delivered].at <= nowMs) {
        const frames = Math.round(RATE * MEASURED[delivered].audio / 1000);
        node.port.onmessage({ data: { type: 'samples', pcm: new Float32Array(frames).fill(0.5).buffer } });
        delivered += 1;
        // ...and this on response.audio.done.
        if (delivered === MEASURED.length) node.port.onmessage({ data: { type: 'start_now' } });
      }
      const sounding = render();
      if (sounding) {
        if (startedAt === null) startedAt = nowMs;
        dry = 0;
      } else if (startedAt !== null && delivered < MEASURED.length) {
        dry += 1;
        if (dry === 3) stalls += 1;      // 3 blocks = 16 ms, past rounding
      }
      nowMs += blockMs;
      if (nowMs > 60000) break;          // never hang the test on a logic error
    }
    perTurn.push({ startedAt, stalls });
  }
  return perTurn;
}

// The other order: the next turn is announced while the previous reply is still
// sounding. The re-arm must not cut that tail off mid-word, which is why it is
// deferred rather than applied on arrival.
function tailSurvivesRearm() {
  const file = path.join(__dirname, 'app', 'static', 'playback_worklet.js');
  const { cls } = loadProcessor(file);
  const node = new cls({ processorOptions: { prebufferFrames: 0 } });
  const outputs = [[new Float32Array(BLOCK)]];
  const out = outputs[0];
  const frames = Math.round(RATE * 1.0);            // 1 s of tail still to play
  node.port.onmessage({ data: { type: 'samples', pcm: new Float32Array(frames).fill(0.5).buffer } });
  let played = 0;
  for (let b = 0; b < 40; b += 1) {                 // ~0.2 s in
    out[0].fill(0); node.process([], outputs);
    for (let i = 0; i < BLOCK; i += 1) if (out[0][i] !== 0) played += 1;
  }
  node.port.onmessage({ data: { type: 'rearm', frames: Math.round(RATE * 1.4) } });
  for (let b = 0; b < 300; b += 1) {                // play the rest out
    out[0].fill(0); node.process([], outputs);
    for (let i = 0; i < BLOCK; i += 1) if (out[0][i] !== 0) played += 1;
  }
  return { played, frames };
}

console.log('\n4. several turns in one call');

const smoothTurns = runTurns(1400, 3);
check('the smooth start applies to EVERY turn, not just the first',
      smoothTurns.every((t) => t.stalls === 0 && t.startedAt >= 1700),
      smoothTurns.map((t, i) => `turn ${i}: start ${t.startedAt === null ? 'never' : t.startedAt.toFixed(0)} ms, `
        + `${t.stalls} stall(s)`).join('; '));

const fastTurns = runTurns(60, 3);
check('the early start still stalls on every turn, as designed',
      fastTurns.every((t) => t.stalls >= 1 && t.startedAt < 500),
      fastTurns.map((t, i) => `turn ${i}: start ${t.startedAt.toFixed(0)} ms, ${t.stalls} stall(s)`).join('; '));

const tail = tailSurvivesRearm();
check('a re-arm while the previous reply is still sounding does not cut it off',
      tail.played >= tail.frames * 0.99,
      `played ${tail.played}/${tail.frames} frames of the tail -- the deferred re-arm `
      + `waits for the queue to drain instead of truncating a word`);

// ---------------------------------------------------------------------------
// 5. the SHIPPED schedule (codec_chunk_frames: 4), where early is also smooth
//
// The schedule in section 3 is what codec_chunk_frames: 25 produces, and the whole
// prebuffer trade only exists because of it. At 4 the granules are 0.32 s and arrive
// every 0.213 s, so delivery outruns playback and there is nothing to trade.
//
// The first boundary is still tight and that is worth pinning: 0.217 s of playable
// audio against 0.213 s to make the next granule is a 4 ms margin, and one measured
// turn in nine showed exactly a 4 ms stall. Inaudible, but it is why the prebuffer
// should not be shaved below 60 ms and why widening it belongs server-side, in
// initial_codec_chunk_frames.
const SHIPPED = [{ at: 350, audio: 217 }].concat(
  Array.from({ length: 12 }, (_, i) => ({ at: 350 + 213 * (i + 1), audio: 320 })));

console.log('\n5. the shipped schedule (codec_chunk_frames: 4)');

const shipped = runSchedule(60, SHIPPED);
const shippedWorst = shipped.gaps.reduce((m, g) => Math.max(m, g.durMs), 0);
check('at codec_chunk_frames 4 the earliest start is already gap-free',
      shipped.startedAtMs < 400 && shippedWorst < 20,
      `starts ${shipped.startedAtMs.toFixed(0)} ms, worst gap ${shippedWorst.toFixed(0)} ms `
      + `-- the 4 ms first-boundary margin is the only slack in it`);

console.log(failures === 0 ? '\nAll checks passed.' : `\n${failures} check(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
