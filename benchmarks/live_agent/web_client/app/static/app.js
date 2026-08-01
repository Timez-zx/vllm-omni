// Browser client for the Qwen3-Omni live session on /v1/video/chat/stream.
//
// Adapted from upstream's MiniCPM realtime_web client. The capture worklet and
// the stylesheet are reused verbatim; the playback worklet is adapted; this
// file -- the transport -- is rewritten, because the two servers speak
// different protocols:
//
//   upstream (MiniCPM, OpenAI Realtime)      this server (Qwen live session)
//   -------------------------------------    ---------------------------------
//   session.update {session:{...}}           session.config {...}  (flat)
//   input_audio_buffer.append                audio.chunk  +  video.frame
//     with video_frames riding along           as TWO separate messages
//   (never commits -- model decides)         video.query   <- REQUIRED
//   response.audio.delta = raw PCM           response.audio.delta = a COMPLETE
//                                              WAV FILE, 24 kHz, per chunk
//
// Three of those are easy to get wrong and fail quietly:
//
//   * Uplink audio is RAW PCM16 at 16 kHz. Downlink is a whole WAV per chunk at
//     24 kHz. Concatenating downlink payloads verbatim embeds a 44-byte header
//     every chunk, which is a click each time -- so each one is parsed here and
//     only its samples are queued.
//   * Frames are their own message type, not a field on the audio append.
//   * THE TURN TRIGGER LIVES HERE, and that is a real difference from the
//     MiniCPM client rather than a shortcut. Qwen3-Omni has no listen/speak
//     token in its vocabulary, and this server's should_trigger_turn() returns
//     False unconditionally, so nothing on the server side will ever decide the
//     user has finished. The UI says so, because a user comparing the two
//     clients deserves to know which one the model is actually driving.

(() => {
  'use strict';

  const config = window.LIVE_AGENT_CONFIG || {};

  // addModule() caches aggressively and a reload does not reliably replace it, so a
  // stale playback worklet can make a fixed audio bug look unfixed. The server stamps
  // every asset; carry it onto the worklet URL too.
  const versioned = (url) => (config.assetVersion ? `${url}?v=${config.assetVersion}` : url);

  const el = (id) => document.getElementById(id);
  const callButton = el('callButton');
  const muteButton = el('muteButton');
  const cameraButton = el('cameraButton');
  const cameraPreview = el('cameraPreview');
  const talkButton = el('talkButton');
  const triggerMode = el('triggerMode');
  const playbackMode = el('playbackMode');
  const systemPromptInput = el('systemPrompt');
  const connectionState = el('connectionState');
  const modelState = el('modelState');
  const playbackState = el('playbackState');
  const sessionTimer = el('sessionTimer');
  const meterFill = el('meterFill');
  const conversation = el('conversation');
  const emptyConversation = el('emptyConversation');
  const eventLog = el('eventLog');
  const eventCount = el('eventCount');
  const runtimeDetail = el('runtimeDetail');
  const clearLogButton = el('clearLogButton');

  const INPUT_RATE = 16000;          // this server's audio.chunk contract
  const SEND_INTERVAL_MS = 200;      // how often queued mic PCM goes up
  const FRAME_INTERVAL_MS = 500;     // ~2 fps; frames are their own message
  // How much audio to hold before the first sample plays. Measured, not guessed --
  // benchmarks/live_agent/web_client/audio_timeline.py, and the right value depends
  // entirely on the server's `codec_chunk_frames`.
  //
  // With the shipped 4 (0.32 s granules, one every ~0.213 s):
  //   delta 0  0.35 s  0.217 s of audio      delta 1+  every 0.213 s, 0.320 s each
  //   Delivery outruns playback from the first boundary, so 60 ms is smooth AND early.
  //
  // With 25 (2.0 s granules, one every ~1.28 s) it is not: 0.217 s of audio cannot
  // cover the 1.34 s the next granule takes, so an early start stalls ~1.1 s one word
  // in and the only alternative is to wait for delta 1 at 1.70 s. That is what `smooth`
  // is for, and why the choice is still on the page -- a page cannot see the server's
  // chunk size, so if speech ever stutters one word in, this is the switch.
  //
  // The first boundary is TIGHT even at 4: 0.217 s of playable audio against 0.213 s to
  // produce the next granule is a 4 ms margin, and one turn in nine showed exactly a
  // 4 ms stall. Inaudible, but it is the reason not to shave this further. To widen it,
  // raise `initial_codec_chunk_frames` server-side rather than this.
  const PLAYBACK_PREBUFFER_MS = { fast: 60, smooth: 1400 };
  const ECHO_GUARD_MS = 300;         // keep uploading this long after playback

  // Silence detection, used only when the trigger mode is 'auto'. These are
  // deliberately conservative: a false trigger interrupts the user mid-sentence,
  // which is far more annoying than a late one.
  const SILENCE_RMS = 0.012;
  const SILENCE_HANG_MS = 700;
  const MIN_SPEECH_MS = 400;         // ignore coughs and door slams

  const DEFAULT_SYSTEM_PROMPT =
    'You are a friendly voice assistant in a live video call. You can see the camera and hear '
    + 'the user. Reply out loud in one or two short sentences, conversationally. Never describe '
    + 'yourself as an AI model. Always answer with both text and speech.';

  let socket = null;
  let mediaStream = null;
  let captureContext = null;
  let captureNode = null;
  let playbackContext = null;
  let playbackNode = null;
  let sendTimer = null;
  let clockTimer = null;
  let startedAt = 0;
  let running = false;
  let muted = false;
  let captureRate = INPUT_RATE;
  let pendingCapture = [];
  let cameraStream = null;
  let cameraTimer = null;
  let cameraPendingFrame = null;
  const cameraCanvas = document.createElement('canvas');

  let assistantSpeaking = false;
  let lastAudioAt = 0;
  let turnInFlight = false;
  let turnWatchdog = null;
  let events = 0;

  // silence-detector state
  let speechMs = 0;
  let silenceMs = 0;
  let sawSpeech = false;

  // ---------------------------------------------------------------- helpers
  function log(message, isError) {
    events += 1;
    if (eventCount) eventCount.textContent = String(events);
    if (!eventLog) return;
    const line = document.createElement('div');
    line.textContent = `${new Date().toLocaleTimeString()}  ${message}`;
    if (isError) line.style.color = 'var(--bad, #c0392b)';
    eventLog.appendChild(line);
    eventLog.scrollTop = eventLog.scrollHeight;
  }

  // The stylesheet colours these three readouts as pills off `data-state`, so the
  // one that matters -- Speaking -- is readable without reading. Deriving the
  // state from the label keeps every call site a plain string and means a new
  // label degrades to the neutral pill rather than breaking.
  const STATE_CLASS = {
    Idle: 'idle', Connecting: 'busy', Connected: 'ok', Listening: 'ok',
    Thinking: 'busy', Speaking: 'speaking', Playing: 'speaking',
    Draining: 'busy', Error: 'bad', Disconnected: 'bad',
  };

  function setState(node, text) {
    if (!node) return;
    node.textContent = text;
    // Underrun carries a count ("Underrun x3"), so match the first word.
    const key = text.split(' ')[0];
    node.dataset.state = STATE_CLASS[key] || (key === 'Underrun' ? 'bad' : 'idle');
  }

  const setConnection = (t) => setState(connectionState, t);
  const setModel = (t) => setState(modelState, t);
  const setPlayback = (t) => setState(playbackState, t);

  function addTranscript(role, text) {
    if (!conversation || !text) return;
    if (emptyConversation) emptyConversation.style.display = 'none';
    let last = conversation.lastElementChild;
    if (!last || last.dataset.role !== role || last.dataset.done === '1') {
      last = document.createElement('div');
      last.dataset.role = role;
      last.className = `bubble ${role}`;
      last.textContent = role === 'user' ? 'you: ' : '';
      conversation.appendChild(last);
    }
    last.textContent += text;
    conversation.scrollTop = conversation.scrollHeight;
  }

  function finishTranscript(role) {
    const last = conversation && conversation.lastElementChild;
    if (last && last.dataset.role === role) last.dataset.done = '1';
  }

  function int16ToBase64(pcm) {
    const bytes = new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength);
    let binary = '';
    const step = 0x8000;
    for (let i = 0; i < bytes.length; i += step) {
      binary += String.fromCharCode.apply(null, bytes.subarray(i, i + step));
    }
    return btoa(binary);
  }

  function base64ToBytes(b64) {
    const binary = atob(b64);
    const out = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) out[i] = binary.charCodeAt(i);
    return out;
  }

  // Linear resample. The mic is usually 48 kHz on a Mac and the server contract
  // is 16 kHz; CoreAudio is not asked to convert because AudioContext sample
  // rates are not reliably settable across browsers.
  function resampleInt16(input, fromRate, toRate) {
    if (fromRate === toRate) return input;
    const ratio = fromRate / toRate;
    const outLength = Math.floor(input.length / ratio);
    const out = new Int16Array(outLength);
    for (let i = 0; i < outLength; i += 1) {
      const pos = i * ratio;
      const idx = Math.floor(pos);
      const frac = pos - idx;
      const a = input[idx] || 0;
      const b = idx + 1 < input.length ? input[idx + 1] : a;
      out[i] = a + (b - a) * frac;
    }
    return out;
  }

  // Parse one complete WAV file and return {pcm: Float32Array, rate}. The
  // header is walked rather than assumed to be 44 bytes: a writer that inserts
  // any extra chunk would otherwise put header bytes into the audio, which is
  // audible but easy to blame on the model.
  function decodeWav(bytes) {
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if (dv.byteLength < 12) throw new Error('wav too short');
    if (String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]) !== 'RIFF') {
      throw new Error('not RIFF');
    }
    let offset = 12;
    let rate = 24000;
    let bits = 16;
    let channels = 1;
    let dataStart = -1;
    let dataLen = 0;
    while (offset + 8 <= dv.byteLength) {
      const id = String.fromCharCode(bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]);
      const size = dv.getUint32(offset + 4, true);
      const body = offset + 8;
      if (id === 'fmt ') {
        channels = dv.getUint16(body + 2, true);
        rate = dv.getUint32(body + 4, true);
        bits = dv.getUint16(body + 14, true);
      } else if (id === 'data') {
        dataStart = body;
        dataLen = Math.min(size, dv.byteLength - body);
        break;
      }
      offset = body + size + (size % 2);
    }
    if (dataStart < 0) throw new Error('no data chunk');
    if (bits !== 16) throw new Error(`unsupported bit depth ${bits}`);
    const count = Math.floor(dataLen / 2);
    const pcm = new Float32Array(Math.floor(count / channels));
    for (let i = 0; i < pcm.length; i += 1) {
      // Downmix defensively; this server sends mono, but a stereo surprise
      // played as mono would sound like double-speed rather than fail.
      let acc = 0;
      for (let c = 0; c < channels; c += 1) {
        acc += dv.getInt16(dataStart + (i * channels + c) * 2, true);
      }
      pcm[i] = (acc / channels) / 32768;
    }
    return { pcm, rate };
  }

  // ------------------------------------------------------------- the session
  function buildSessionConfig() {
    const instructions = (systemPromptInput && systemPromptInput.value.trim()) || DEFAULT_SYSTEM_PROMPT;
    return {
      type: 'session.config',
      system_prompt: instructions,
      modalities: ['text', 'audio'],
      // Frames the server samples per turn, and how many it keeps buffered.
      num_frames: 16,
      max_frames: 256,
      // Shrink on arrival: one 1280x720 frame is 880 tokens against 220 at
      // 640x352, and that cost lands on every turn's latency.
      max_frame_width: 640,
      max_frame_height: 352,
      frame_jpeg_quality: 90,
      // Without the filter a moving camera fills the prompt in seconds; the two
      // gaps bound it from both sides so a still scene still yields a frame.
      enable_frame_filter: true,
      frame_filter_threshold: 0.95,
      frame_filter_min_gap: 8,
      frame_filter_max_gap: 16,
      use_audio_in_video: true,
      // One engine request per conversation, so turn 20 does not re-read turns
      // 1..19, and an automatic roll before the speech stage's limit so the
      // conversation can run indefinitely.
      session_scoped_request: true,
      // Turn each retained frame into tokens as it ARRIVES, so its prefill happens while
      // you are still speaking instead of after you stop. Session mode already only ever
      // submits new frames; this changes when that work runs, not how much of it there is.
      // Turn each retained frame into tokens as it ARRIVES. Session mode already submits
      // only new frames, so this moves WHEN that work happens, not how much there is.
      // Measured, 6 turns per arm, 6 s of streaming before each query:
      //   OFF  median 355.4 ms   range 344.2-379.3  (spread 35.1)
      //   ON   median 345.5 ms   range 343.0-348.6  (spread  5.6)
      // The median gain is small and matches the token arithmetic (232 tokens moved off the
      // critical path). The SPREAD is the real result: 6x tighter, because frame prefill is
      // no longer racing the query. An earlier measurement claimed -73%; that was audio
      // landing against the wrong turn, and it went away when the marker was fixed.
      // OFF: it CRASHES THE ENGINE in real browser use. A stage-1 CUDA device-side assert
      // (torch.AcceleratorError / cudaErrorAssert) killed the engine core about 10 s into a
      // hand-driven session. My own A/B survived because it retains ~1-3 frames per turn;
      // the browser streams continuously, so appends accumulate -- and each append adds a
      // spurious `<|im_start|>assistant` header plus its one sampled token to the thinker's
      // context, which is the leading suspect for desynchronising the talker's span
      // accounting. A latency win is not worth a dead engine.
      prefill_frames_on_arrival: false,
      session_roll_at_talker_tokens: 45000,
      session_roll_history_turns: 8,
    };
  }

  function realtimeUrl() {
    if (config.wsUrl) return config.wsUrl;
    const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws';
    return `${scheme}://${window.location.host}/ws`;
  }

  function handleEvent(event) {
    const type = event.type;
    switch (type) {
      case 'response.start':
        turnInFlight = true;
        // Re-apply the prebuffer for this turn. The worklet node lives for the whole
        // call, so without this the smooth start applies to the first reply only.
        if (playbackNode) {
          playbackNode.port.postMessage({ type: 'rearm', frames: prebufferFrames() });
        }
        setModel('Thinking');
        log('turn started');
        break;
      case 'response.text.delta':
        addTranscript('assistant', event.delta || '');
        break;
      case 'response.text.done':
        finishTranscript('assistant');
        break;
      case 'response.audio.delta': {
        assistantSpeaking = true;
        lastAudioAt = performance.now();
        setModel('Speaking');
        try {
          const { pcm, rate } = decodeWav(base64ToBytes(event.data || ''));
          feedPlayback(pcm, rate);
        } catch (error) {
          log(`audio decode failed: ${error.message}`, true);
        }
        break;
      }
      case 'response.audio.done':
        // Release the smooth-start threshold: no more audio is coming, so whatever is
        // queued is the whole remainder of the reply. A reply shorter than the target
        // would otherwise sit in the buffer and never play.
        if (playbackNode) playbackNode.port.postMessage({ type: 'start_now' });
        setPlayback('Draining');
        endTurn(null);
        log('turn done');
        break;
      case 'session.done':
        endTurn('server closed the session');
        break;
      case 'session.rolled':
        log(`session rolled (#${event.rolls}, carried ${event.carried_messages} messages)`);
        break;
      case 'error':
        log(`server error: ${event.message}`, true);
        setModel('Error');
        break;
      default:
        log(`event ${type}`);
    }
  }

  function feedPlayback(pcm, rate) {
    if (!playbackNode) return;
    // The worklet runs at the AudioContext rate; resample if the server's rate
    // differs rather than letting it play at the wrong speed.
    let samples = pcm;
    if (rate !== playbackContext.sampleRate) {
      const ratio = rate / playbackContext.sampleRate;
      const out = new Float32Array(Math.floor(pcm.length / ratio));
      for (let i = 0; i < out.length; i += 1) {
        const pos = i * ratio;
        const idx = Math.floor(pos);
        const frac = pos - idx;
        const a = pcm[idx] || 0;
        const b = idx + 1 < pcm.length ? pcm[idx + 1] : a;
        out[i] = a + (b - a) * frac;
      }
      samples = out;
    }
    const buf = samples.buffer.slice(samples.byteOffset, samples.byteOffset + samples.byteLength);
    playbackNode.port.postMessage({ type: 'samples', pcm: buf }, [buf]);
    setPlayback('Playing');
  }

  // ------------------------------------------------------------ the trigger
  //
  // Client-side by necessity, not by choice. See the header comment.
  function endTurn(why) {
    if (!turnInFlight) return;
    turnInFlight = false;
    assistantSpeaking = false;
    if (turnWatchdog !== null) { window.clearTimeout(turnWatchdog); turnWatchdog = null; }
    setModel('Listening');
    if (why) log(`turn ended (${why})`);
  }

  function sendQuery(reason) {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    if (turnInFlight) {
      log(`turn trigger suppressed (${reason}): one is still in flight`);
      return;
    }
    // Empty text: the question is in the audio, which is the point of a voice
    // interface. The server treats video.query purely as a cut marker.
    socket.send(JSON.stringify({ type: 'video.query', text: '' }));
    turnInFlight = true;
    sawSpeech = false;
    speechMs = 0;
    silenceMs = 0;
    setModel('Thinking');
    log(`turn trigger sent (${reason})`);
    // Without this, one missing response.audio.done wedges the client for the
    // rest of the session: turnInFlight gates both the trigger and the mic
    // upload, so nothing would ever recover it.
    if (turnWatchdog !== null) window.clearTimeout(turnWatchdog);
    turnWatchdog = window.setTimeout(() => endTurn('watchdog: no completion in 45 s'), 45000);
  }

  function updateSilenceDetector(rms, elapsedMs) {
    if (triggerMode && triggerMode.value !== 'auto') return;
    if (assistantSpeaking || turnInFlight) { sawSpeech = false; speechMs = 0; silenceMs = 0; return; }
    if (rms >= SILENCE_RMS) {
      speechMs += elapsedMs;
      silenceMs = 0;
      if (speechMs >= MIN_SPEECH_MS) sawSpeech = true;
    } else if (sawSpeech) {
      silenceMs += elapsedMs;
      if (silenceMs >= SILENCE_HANG_MS) sendQuery('silence detected');
    }
  }

  function microphoneUploadEnabled() {
    if (muted) return false;
    // Keep uploading during and just after playback so the stream is genuinely
    // continuous -- audio arriving mid-turn is buffered by the server for the
    // NEXT turn, so nothing is lost. The guard only suppresses the window where
    // the speakers would feed the model its own voice.
    if (assistantSpeaking) return false;
    if (performance.now() - lastAudioAt < ECHO_GUARD_MS) return false;
    return true;
  }

  // --------------------------------------------------------------- plumbing
  function flushCapture() {
    if (!socket || socket.readyState !== WebSocket.OPEN) { pendingCapture = []; return; }
    if (pendingCapture.length === 0) return;
    if (!microphoneUploadEnabled()) { pendingCapture = []; return; }
    let length = 0;
    for (const chunk of pendingCapture) length += chunk.length;
    const merged = new Int16Array(length);
    let offset = 0;
    for (const chunk of pendingCapture) { merged.set(chunk, offset); offset += chunk.length; }
    pendingCapture = [];
    const pcm = resampleInt16(merged, captureRate, INPUT_RATE);
    socket.send(JSON.stringify({ type: 'audio.chunk', data: int16ToBase64(pcm) }));
  }

  function updateMeter(pcm) {
    let sum = 0;
    for (let i = 0; i < pcm.length; i += 1) { const v = pcm[i] / 32768; sum += v * v; }
    const rms = Math.sqrt(sum / Math.max(1, pcm.length));
    if (meterFill) meterFill.style.width = `${Math.min(100, rms * 400).toFixed(1)}%`;
    updateSilenceDetector(rms, (pcm.length / captureRate) * 1000);
  }

  async function startCapture() {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    captureContext = new AudioContext();
    captureRate = captureContext.sampleRate;
    await captureContext.audioWorklet.addModule('static/pcm_worklet.js');
    const source = captureContext.createMediaStreamSource(mediaStream);
    captureNode = new AudioWorkletNode(captureContext, 'fullduplex-pcm-capture');
    captureNode.port.onmessage = (message) => {
      const pcm = new Int16Array(message.data);
      updateMeter(pcm);
      if (microphoneUploadEnabled()) pendingCapture.push(pcm);
    };
    const sink = captureContext.createGain();
    sink.gain.value = 0;
    source.connect(captureNode);
    captureNode.connect(sink).connect(captureContext.destination);
    await captureContext.resume();
  }

  function prebufferFrames() {
    const mode = (playbackMode && playbackMode.value) === 'smooth' ? 'smooth' : 'fast';
    const rate = playbackContext ? playbackContext.sampleRate : 24000;
    return Math.floor(rate * PLAYBACK_PREBUFFER_MS[mode] / 1000);
  }

  async function startPlayback() {
    playbackContext = new AudioContext();
    await playbackContext.audioWorklet.addModule(versioned('static/playback_worklet.js'));
    playbackNode = new AudioWorkletNode(playbackContext, 'live-agent-playback', {
      processorOptions: { prebufferFrames: prebufferFrames() },
    });
    playbackNode.port.onmessage = (message) => {
      const msg = message.data || {};
      if (msg.type === 'underrun') {
        setPlayback(`Underrun x${msg.underruns}`);
      } else if (msg.type === 'started') {
        setPlayback('Playing');
      }
    };
    playbackNode.connect(playbackContext.destination);
    await playbackContext.resume();
  }

  function startCamera() {
    if (cameraStream) return;
    navigator.mediaDevices.getUserMedia({ video: true, audio: false }).then(async (stream) => {
      cameraStream = stream;
      cameraPreview.srcObject = stream;
      cameraPreview.style.display = '';
      await cameraPreview.play().catch(() => {});
      cameraTimer = window.setInterval(() => {
        if (!cameraStream || cameraPreview.videoWidth === 0) return;
        cameraCanvas.width = cameraPreview.videoWidth;
        cameraCanvas.height = cameraPreview.videoHeight;
        cameraCanvas.getContext('2d').drawImage(cameraPreview, 0, 0);
        cameraPendingFrame = cameraCanvas.toDataURL('image/jpeg', 0.8).split(',')[1];
        // Frames are their own message on this protocol, and the server shrinks
        // and filters them on arrival, so they are sent as captured.
        if (socket && socket.readyState === WebSocket.OPEN && cameraPendingFrame) {
          socket.send(JSON.stringify({ type: 'video.frame', data: cameraPendingFrame }));
          cameraPendingFrame = null;
        }
      }, FRAME_INTERVAL_MS);
      cameraButton.textContent = 'Camera off';
      cameraButton.classList.add('is-active');
      log(`camera on (~${(1000 / FRAME_INTERVAL_MS).toFixed(1)} fps)`);
    }).catch((error) => {
      log(`camera failed: ${error.message}`, true);
    });
  }

  function stopCamera() {
    if (cameraTimer !== null) window.clearInterval(cameraTimer);
    cameraTimer = null;
    if (cameraStream) for (const track of cameraStream.getTracks()) track.stop();
    cameraStream = null;
    cameraPendingFrame = null;
    cameraPreview.srcObject = null;
    cameraPreview.style.display = 'none';
    cameraButton.textContent = 'Camera';
    cameraButton.classList.remove('is-active');
  }

  function openSocket() {
    return new Promise((resolve, reject) => {
      const url = realtimeUrl();
      socket = new WebSocket(url);
      socket.onopen = () => {
        socket.send(JSON.stringify(buildSessionConfig()));
        if (runtimeDetail) {
          runtimeDetail.textContent = `${captureRate} Hz capture -> ${INPUT_RATE} Hz uplink / `
            + `${playbackContext ? playbackContext.sampleRate : '?'} Hz playback`;
        }
        log(`websocket open  ${url}`);
        setConnection('Connected');
        setModel('Listening');
        resolve();
      };
      socket.onmessage = (message) => {
        if (typeof message.data !== 'string') return;
        try {
          handleEvent(JSON.parse(message.data));
        } catch (error) {
          log(`bad server event: ${error.message}`, true);
        }
      };
      socket.onerror = () => { log('websocket error', true); reject(new Error('websocket error')); };
      socket.onclose = () => { log('websocket closed'); if (running) stop(); };
    });
  }

  async function start() {
    if (running) return;
    running = true;
    callButton.textContent = 'Hang up';
    callButton.classList.add('is-active');
    setConnection('Connecting');
    try {
      await startPlayback();
      await startCapture();
      await openSocket();
      startedAt = Date.now();
      sendTimer = window.setInterval(flushCapture, SEND_INTERVAL_MS);
      clockTimer = window.setInterval(() => {
        const s = Math.floor((Date.now() - startedAt) / 1000);
        if (sessionTimer) {
          sessionTimer.textContent = `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
        }
      }, 500);
      if (cameraButton) cameraButton.disabled = false;
      if (talkButton) talkButton.disabled = false;
    } catch (error) {
      log(`start failed: ${error.message}`, true);
      stop();
    }
  }

  function stop() {
    running = false;
    callButton.textContent = 'Start call';
    callButton.classList.remove('is-active');
    setConnection('Idle');
    setModel('Idle');
    setPlayback('Idle');
    if (sendTimer !== null) window.clearInterval(sendTimer);
    if (clockTimer !== null) window.clearInterval(clockTimer);
    sendTimer = clockTimer = null;
    stopCamera();
    if (socket && socket.readyState === WebSocket.OPEN) {
      try { socket.send(JSON.stringify({ type: 'video.done' })); } catch (_) {}
      try { socket.close(); } catch (_) {}
    }
    socket = null;
    if (captureNode) { try { captureNode.disconnect(); } catch (_) {} captureNode = null; }
    if (captureContext) { captureContext.close().catch(() => {}); captureContext = null; }
    if (playbackNode) { try { playbackNode.disconnect(); } catch (_) {} playbackNode = null; }
    if (playbackContext) { playbackContext.close().catch(() => {}); playbackContext = null; }
    if (mediaStream) { for (const t of mediaStream.getTracks()) t.stop(); mediaStream = null; }
    pendingCapture = [];
    if (cameraButton) cameraButton.disabled = true;
    if (talkButton) talkButton.disabled = true;
  }

  // ------------------------------------------------------------------ wiring
  callButton.addEventListener('click', () => { if (running) stop(); else start(); });

  muteButton.addEventListener('click', () => {
    muted = !muted;
    muteButton.textContent = muted ? 'Unmute' : 'Mute';
    muteButton.classList.toggle('is-active', muted);
    log(muted ? 'microphone muted' : 'microphone live');
  });

  cameraButton.addEventListener('click', () => {
    if (cameraStream) stopCamera(); else startCamera();
  });

  // Push to talk: hold, speak, release. Kept as the reliable fallback for when
  // silence detection misfires -- it cannot false-trigger.
  if (talkButton) {
    const press = (e) => { e.preventDefault(); talkButton.classList.add('is-active'); };
    const release = (e) => {
      e.preventDefault();
      talkButton.classList.remove('is-active');
      sendQuery('push to talk released');
    };
    talkButton.addEventListener('mousedown', press);
    talkButton.addEventListener('mouseup', release);
    talkButton.addEventListener('touchstart', press);
    talkButton.addEventListener('touchend', release);
  }

  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && running && !e.repeat && document.activeElement === document.body) {
      e.preventDefault();
      sendQuery('space bar');
    }
  });

  if (clearLogButton) {
    clearLogButton.addEventListener('click', () => { eventLog.innerHTML = ''; events = 0; eventCount.textContent = '0'; });
  }
  if (playbackMode) {
    playbackMode.addEventListener('change', () => {
      // Retune the live node so the effect is audible on the next reply rather than
      // only after hanging up.
      if (playbackNode) playbackNode.port.postMessage({ type: 'prebuffer', frames: prebufferFrames() });
      log(`playback start: ${playbackMode.value} (${PLAYBACK_PREBUFFER_MS[playbackMode.value]} ms buffered)`);
    });
  }

  if (systemPromptInput && !systemPromptInput.value) systemPromptInput.value = DEFAULT_SYSTEM_PROMPT;
  log(`client assets ${config.assetVersion || 'unversioned'}`);
  setConnection('Idle');
  log('ready -- press Start call');
})();
