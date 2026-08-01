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
//   1. the prebuffer threshold (250 ms) was LARGER than one turn's delivered
//      audio (~220 ms). Alone this only loses the first turn: the audio stays
//      queued, the next turn pushes the total over the threshold, and playback
//      simply runs one turn behind.
//   2. an underrun re-armed the prebuffer. Alone, at a prebuffer smaller than a
//      turn, this is harmless -- every turn clears the threshold again.
//
// Together they are the bug: once the threshold exceeds one turn, re-arming means
// each turn must be paid for out of the NEXT one, so turns fall silent in turn and
// what does play is the previous reply. Fixing one and not the other would have
// looked like a fix and left it broken, which is the reason this file exists.
//
// Neither is visible to selftest.py, which checks the protocol, nor to probe.py,
// which never plays anything. A browser was the only thing that could catch it,
// and a browser is exactly what neither of those has. So run the real worklet
// under Node with the two AudioWorklet globals stubbed, and drive it with the
// arrival pattern the server actually produces: a short burst, a long gap, the
// next burst.

'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const RATE = 24000;                     // the server's audio.delta rate
const BLOCK = 128;                      // AudioWorklet render quantum
const TURN_MS = 220;                    // what a turn currently delivers
const GAP_MS = 3000;                    // thinking time between turns
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
      + `re-arm then demands 250 ms of every 220 ms turn, so turns fall silent and what `
      + `does play is the PREVIOUS turn's reply, two turns' worth at a time`);

console.log(failures === 0 ? '\nAll checks passed.' : `\n${failures} check(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
