// Contact sounds, synthesised with Web Audio and placed binaurally (HRTF panners, listener at the head).
// Impacts: a filtered noise burst for the contact plus a decaying low tone for the body's resonance.
// Harder hits are louder and brighter; larger bodies are lower and ring longer.
// Rubbing: a looped noise through a band-pass filter. The filter centre follows the sliding speed, the
// level follows the sliding power (speed x normal force), and a slow random modulation adds stick-slip
// grain. A few rubbing voices are shared among the loudest sliding contacts. Silent voices stop their
// sources, so idle sounds cost nothing.
// Every call takes an explicit audio time, so the same code renders offline (OfflineAudioContext).

export const MATERIALS = {          // per geom: brightness (filter) and how the surface sounds
  floor: { bright: 0.45, gain: 1.0 },
  table: { bright: 0.75, gain: 1.0 },
  wall: { bright: 0.55, gain: 0.9 },
  box: { bright: 0.65, gain: 1.0 },
  hand: { bright: 0.25, gain: 0.45 },
};

export function createContactAudio(ctx, { maxRub = 4, volume = 0.9, hrtf = true } = {}) {
  const master = ctx.createGain();
  master.gain.value = volume;
  const comp = ctx.createDynamicsCompressor();
  comp.threshold.value = -14; comp.ratio.value = 4; comp.attack.value = 0.003; comp.release.value = 0.15;
  master.connect(comp).connect(ctx.destination);

  const noise = ctx.createBuffer(1, ctx.sampleRate * 2, ctx.sampleRate);
  { const d = noise.getChannelData(0); let b = 0;
    for (let i = 0; i < d.length; i++) { const w = Math.random() * 2 - 1; b = 0.97 * b + 0.03 * w; d[i] = 0.6 * w + 2.5 * b; } }

  const panner = (pos) => {
    const p = ctx.createPanner();
    p.panningModel = hrtf ? 'HRTF' : 'equalpower';
    p.distanceModel = 'inverse'; p.refDistance = 0.6; p.rolloffFactor = 1.2;
    p.positionX.value = pos[0]; p.positionY.value = pos[1]; p.positionZ.value = pos[2];
    p.connect(master);
    return p;
  };

  function setListener(t, pos, fwd, up) {
    const L = ctx.listener;
    if (L.positionX) {
      L.positionX.setValueAtTime(pos[0], t); L.positionY.setValueAtTime(pos[1], t); L.positionZ.setValueAtTime(pos[2], t);
      L.forwardX.setValueAtTime(fwd[0], t); L.forwardY.setValueAtTime(fwd[1], t); L.forwardZ.setValueAtTime(fwd[2], t);
      L.upX.setValueAtTime(up[0], t); L.upY.setValueAtTime(up[1], t); L.upZ.setValueAtTime(up[2], t);
    } else {
      L.setPosition(...pos); L.setOrientation(...fwd, ...up);
    }
  }

  // An impact. speed: closing speed (m/s); mass: effective mass (kg); size: body half extent (m);
  // mat: the two materials. Loudness follows the collision energy on a log scale.
  let lastImpact = -1, recent = 0;
  function impact(t, pos, speed, mass, size, mats) {
    const energy = 0.5 * mass * speed * speed;
    let loud = Math.min(1, Math.log10(1 + energy / 0.01) / 3) * Math.min(mats[0].gain, mats[1].gain);
    recent = recent * Math.exp(-(t - lastImpact) / 0.05) + 1;          // many hits at once: duck each
    lastImpact = t;
    loud /= Math.sqrt(recent);
    if (loud < 0.02) return;
    const bright = 0.5 * (mats[0].bright + mats[1].bright);
    const p = panner(pos);
    // contact click: band of noise, brighter for harder hits
    const src = ctx.createBufferSource(); src.buffer = noise;
    const lp = ctx.createBiquadFilter(); lp.type = 'lowpass';
    lp.frequency.value = 250 + 3800 * bright * Math.min(1, 0.3 + loud); lp.Q.value = 0.7;
    const g = ctx.createGain();
    const dur = 0.05 + 0.25 * Math.min(1, size / 0.25);
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(loud, t + 0.002);
    g.gain.exponentialRampToValueAtTime(0.0005, t + dur);
    src.connect(lp).connect(g).connect(p);
    src.start(t, Math.random() * 1.5, dur + 0.02);
    // body resonance: a low tone that drops a little in pitch
    const f0 = Math.min(420, Math.max(70, 30 / Math.max(size, 0.04)));
    const osc = ctx.createOscillator(); osc.type = 'sine';
    osc.frequency.setValueAtTime(f0, t);
    osc.frequency.exponentialRampToValueAtTime(0.82 * f0, t + 2 * dur);
    const og = ctx.createGain();
    og.gain.setValueAtTime(0, t);
    og.gain.linearRampToValueAtTime(0.7 * loud, t + 0.004);
    og.gain.exponentialRampToValueAtTime(0.0005, t + 2 * dur);
    osc.connect(og).connect(p);
    osc.start(t); osc.stop(t + 2 * dur + 0.02);
    osc.onended = () => p.disconnect();
  }

  // Rubbing voices. rubs: [{key, pos, slide (m/s), fn (N), size, mats}]; call once per frame.
  const voices = Array.from({ length: maxRub }, () => ({ key: null, src: null, idle: 0 }));
  function startVoice(v, t, pos) {
    v.p = panner(pos);
    v.src = ctx.createBufferSource(); v.src.buffer = noise; v.src.loop = true;
    v.src.playbackRate.value = 0.8 + 0.4 * Math.random();
    v.bp = ctx.createBiquadFilter(); v.bp.type = 'bandpass'; v.bp.Q.value = 0.9;
    v.g = ctx.createGain(); v.g.gain.value = 0;
    v.lfo = ctx.createOscillator(); v.lfo.frequency.value = 9 + 8 * Math.random();     // stick-slip flutter
    v.lfoG = ctx.createGain(); v.lfoG.gain.value = 0;
    v.lfo.connect(v.lfoG).connect(v.g.gain);
    v.src.connect(v.bp).connect(v.g).connect(v.p);
    v.src.start(t, Math.random() * 1.5); v.lfo.start(t);
  }
  function stopVoice(v, t) {
    v.src.stop(t); v.lfo.stop(t);
    const p = v.p;
    v.src.onended = () => p.disconnect();
    v.src = null; v.key = null;
  }
  function rub(t, rubs) {
    const level = r => Math.min(0.55, Math.sqrt(r.slide * r.fn) / 9) * Math.min(r.mats[0].gain, r.mats[1].gain);
    const loud = rubs.map(r => ({ ...r, level: level(r) })).filter(r => r.level > 0.015)
                     .sort((a, b) => b.level - a.level).slice(0, maxRub);
    const byKey = new Map(loud.map(r => [r.key, r]));
    for (const v of voices) {                                // keep voices on the contacts they already follow
      const r = v.key !== null ? byKey.get(v.key) : undefined;
      if (r) { byKey.delete(r.key); drive(v, t, r); }
      else if (v.src) {
        v.g.gain.setTargetAtTime(0, t, 0.04); v.lfoG.gain.setTargetAtTime(0, t, 0.04);
        if ((v.idle += 1) > 20) stopVoice(v, t + 0.2);
      }
    }
    for (const r of byKey.values()) {                        // new contacts take free or idle voices
      const v = voices.find(x => !x.src) ?? voices.find(x => x.idle > 0);
      if (!v) break;
      if (!v.src) startVoice(v, t, r.pos);
      v.key = r.key;
      drive(v, t, r);
    }
  }
  function drive(v, t, r) {
    v.idle = 0;
    const bright = 0.5 * (r.mats[0].bright + r.mats[1].bright);
    v.bp.frequency.setTargetAtTime(180 + 2200 * bright * Math.min(1, r.slide / 0.8) + 900 / Math.max(r.size * 10, 1), t, 0.03);
    v.g.gain.setTargetAtTime(r.level, t, 0.03);
    v.lfoG.gain.setTargetAtTime(0.35 * r.level, t, 0.03);
    v.p.positionX.setTargetAtTime(r.pos[0], t, 0.02); v.p.positionY.setTargetAtTime(r.pos[1], t, 0.02);
    v.p.positionZ.setTargetAtTime(r.pos[2], t, 0.02);
  }

  return { setListener, impact, rub, master };
}

// Map the tracker's events (MuJoCo frame) to sound calls (three.js frame: x, z, -y).
export function contactSounds(sim, audio) {
  const mjToThree = p => [p[0], p[2], -p[1]];
  const kind = g => { const n = sim.geomName[g]; return n.startsWith('box') ? 'box' : n.startsWith('hand') ? 'hand' : n; };
  const mass = b => sim.model.body_mass[b];
  const sizeOf = g => { const n = sim.geomName[g], bx = sim.boxes.find(x => x.name === n);
                        return bx ? (bx.half[0] + bx.half[1] + bx.half[2]) / 3 : 0.05; };
  const effMass = (b1, b2) => { const m1 = b1 ? mass(b1) : Infinity, m2 = b2 ? mass(b2) : Infinity;
                                return Number.isFinite(m1 + m2) ? m1 * m2 / (m1 + m2) : Math.min(m1, m2); };
  const bigger = (g1, g2) => Math.max(sizeOf(g1), sizeOf(g2));
  return function play(t, events) {
    for (const e of events.impacts) {
      audio.impact(t, mjToThree(e.pos), e.speed, effMass(e.b1, e.b2), bigger(e.g1, e.g2),
                   [MATERIALS[kind(e.g1)] ?? MATERIALS.box, MATERIALS[kind(e.g2)] ?? MATERIALS.box]);
    }
    audio.rub(t, events.rubs.map(r => ({ key: r.key, pos: mjToThree(r.pos), slide: r.slide, fn: r.fn, size: bigger(r.g1, r.g2),
                                         mats: [MATERIALS[kind(r.g1)] ?? MATERIALS.box, MATERIALS[kind(r.g2)] ?? MATERIALS.box] })));
  };
}
