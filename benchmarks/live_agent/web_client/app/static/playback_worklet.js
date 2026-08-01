// Gapless playback of PCM that arrives irregularly over a network.
//
// Adapted from the upstream MiniCPM realtime_web playback worklet. The change
// that matters is upstream of here: this server sends each audio delta as a
// COMPLETE WAV FILE, so app.js strips the 44-byte header per chunk and posts
// only samples. Feeding header bytes to a vocoder-rate stream is audible as a
// click on every chunk boundary, which is easy to mistake for a model fault.
//
// Two behaviours are deliberate:
//
//   * A prebuffer before the first sample plays, and it must be well under the
//     FIRST delta -- not merely enough for network jitter. The server sends a small
//     first granule on purpose so speech starts early (measured 0.217 s), then 2 s
//     granules. A 250 ms threshold sat above that first delta, so playback never
//     started on it.
//   * On underrun the processor emits SILENCE and keeps running. Returning
//     false, or throwing, permanently kills the node -- and a dead node is
//     silent for the rest of the session with nothing in the console.
//   * An underrun does NOT re-arm the prebuffer, because running dry between turns
//     is normal, not a fault.
//
// The last two interact, and that is the whole bug: an oversized threshold alone
// only delays the start (the audio waits, the next delta tops it up), and re-arming
// alone is harmless while the threshold is under one delta. Together, whatever
// arrives below the threshold is held hostage by the next arrival.
// playback_test.js drives the real class under Node and asserts all three cases,
// because neither selftest.py nor probe.py plays a sample.

class LiveAgentPlayback extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this.prebufferFrames = Math.max(0, opts.prebufferFrames || 0);
    this.queue = [];          // Float32Array chunks
    this.offset = 0;          // read offset into queue[0]
    this.buffered = 0;        // frames currently queued
    this.started = false;
    this.underruns = 0;
    this.playedFrames = 0;

    this.port.onmessage = (event) => {
      const msg = event.data || {};
      if (msg.type === 'samples' && msg.pcm) {
        const f32 = new Float32Array(msg.pcm);
        this.queue.push(f32);
        this.buffered += f32.length;
      } else if (msg.type === 'flush') {
        // Barge-in / turn abandoned: drop everything not yet played so the
        // assistant goes quiet immediately rather than finishing a stale reply.
        this.queue = [];
        this.offset = 0;
        this.buffered = 0;
        this.started = false;
      } else if (msg.type === 'prebuffer') {
        // Live retune, so the start-latency/smoothness toggle takes effect on the
        // current turn rather than the next call.
        this.prebufferFrames = Math.max(0, msg.frames || 0);
      } else if (msg.type === 'start_now') {
        // The turn is over: whatever is queued is the whole rest of the reply, so
        // waiting for a threshold it may never reach would swallow it. Short replies
        // total less than the smooth-start target and would otherwise never play.
        this.prebufferFrames = 0;
      } else if (msg.type === 'stats') {
        this.port.postMessage({
          type: 'stats',
          buffered: this.buffered,
          underruns: this.underruns,
          playedFrames: this.playedFrames,
        });
      }
    };
  }

  process(_inputs, outputs) {
    const out = outputs[0][0];
    if (!out) return true;

    if (!this.started) {
      if (this.buffered < this.prebufferFrames) {
        out.fill(0);
        return true;
      }
      this.started = true;
      this.port.postMessage({ type: 'started' });
    }

    let written = 0;
    while (written < out.length && this.queue.length > 0) {
      const head = this.queue[0];
      const avail = head.length - this.offset;
      const take = Math.min(avail, out.length - written);
      out.set(head.subarray(this.offset, this.offset + take), written);
      written += take;
      this.offset += take;
      this.buffered -= take;
      if (this.offset >= head.length) {
        this.queue.shift();
        this.offset = 0;
      }
    }
    this.playedFrames += written;

    if (written < out.length) {
      out.fill(0, written);
      if (this.started) {
        this.underruns += 1;
        // Stay started. Re-arming the prebuffer here is what made every turn
        // after the first one silent: the threshold was larger than a turn's
        // audio, so once re-armed it was never met again. Running dry between
        // turns is normal and must not latch.
        this.port.postMessage({ type: 'underrun', underruns: this.underruns });
      }
    }
    return true;
  }
}

registerProcessor('live-agent-playback', LiveAgentPlayback);
