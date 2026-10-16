// VR Physics Lab: the spring-hand MuJoCo scene in WebXR on the headset (doc §10 of
// analysis/2026-10-15_vr_teleop_locomanipulation_design.md). three.js renders; MuJoCo WASM simulates;
// each controller pulls a proxy hand through a clamped spring (sim_core.js). Surfaces carry painterly
// textures baked once at load (paint.js); contacts make binaural sound (audio.js).
// Frames: three / WebXR is Y up with -Z ahead; MuJoCo is Z up with +Y ahead. The scene's MuJoCo
// bodies live in a group rotated -90 deg about X, so MuJoCo poses render unchanged inside it.
import * as THREE from 'three';
import { createSim, createContactTracker, PARAMS, STATIC, HAND_HALF, qmul } from './sim_core.js';
import { bakePainted, boxUVs } from './paint.js';
import { createContactAudio, contactSounds } from './audio.js';

const VERSION = 'v8';
const MUJOCO_URL = '../engine/mujoco-3.12.0/mujoco.js';   // vendored by deploy.sh (npm @mujoco/mujoco)
const MAX_STEPS = 34;                        // physics steps per frame at most (34 ms of physics)
const PARK = [[-0.25, 0.15, 1.0], [0.25, 0.15, 1.0]];   // hands with no controller (MuJoCo frame)
const IDENTITY = [1, 0, 0, 0];
const PALETTE = {
  box1: '#dcc48e', box3: '#cfa660', box5: '#c08848', box10: '#a5683a', box20: '#85502f', box30: '#643a26',
  table: '#5e4030', floor: '#3a3f45', hand_l: '#3fb6c4', hand_r: '#e8a33d',
};
const FLOW = { table: 0.25, floor: 0.45 };                               // stroke direction field per surface
// metres per texture tile, each at least the surface it covers, so nothing in view repeats
const TILE = { table: 2.0, floor: 10.0, box: 1.0, hand: 0.5 };
const TEX_SIZE = { floor: 1024 };                                        // px; others 512
const FLOOR = 10;                                                         // floor side, m

const $ = id => document.getElementById(id);
function setCheck(id, state, text) {
  const row = $(id);
  row.dataset.state = state;
  row.querySelector('.value').textContent = text;
}
function logError(msg) {
  const el = $('log');
  el.hidden = false;
  el.textContent += msg + '\n';
}
window.addEventListener('error', e => logError(`error: ${e.message}`));
window.addEventListener('unhandledrejection', e => logError(`error: ${e.reason?.message ?? e.reason}`));

// --- frame conversion ---
const Q_XR_TO_MJ = [Math.SQRT1_2, Math.SQRT1_2, 0, 0];         // wxyz, +90 deg about X
const toMjPos = p => [p.x, -p.z, p.y];
const toMjQuat = q => qmul(Q_XR_TO_MJ, [q.w, q.x, q.y, q.z]);
const lerp = (a, b, t) => a.map((x, i) => x + (b[i] - x) * t);
function nlerp(a, b, t) {
  const d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  const bb = d < 0 ? b.map(x => -x) : b;
  const q = lerp(a, bb, t), n = Math.hypot(...q);
  return q.map(x => x / n);
}

// --- start: MuJoCo, the scene, the benchmark, WebXR support ---
let mujoco, sim;
try {
  const { default: loadMujoco } = await import(MUJOCO_URL);
  mujoco = await loadMujoco();
  sim = createSim(mujoco);
  setCheck('chk-mujoco', 'ok', `MuJoCo ${mujoco.mj_versionString?.() ?? '3.12.0'} loaded`);
} catch (e) {
  setCheck('chk-mujoco', 'fail', 'MuJoCo did not load');
  logError(`MuJoCo: ${e?.message ?? e}`);
  throw e;
}

function benchmark() {
  const b = createSim(mujoco);
  const box = b.boxes.find(x => x.name === 'box5');
  const top = b.pose(box.body).pos; top[2] += box.half[2] + HAND_HALF[2] + 0.002;
  b.placeHand(b.hands[1], top, IDENTITY);
  b.placeHand(b.hands[0], PARK[0], IDENTITY);
  b.step(200);
  b.grasp(1);
  const n = 3000, t0 = performance.now();
  for (let k = 0; k < n; k++) { b.hands[1].target.pos[2] = top[2] + 0.2 * Math.sin(k * 0.003); b.step(1); }
  const us = 1000 * (performance.now() - t0) / n;
  b.dispose();
  return us;
}
const benchUs = benchmark();
setCheck('chk-bench', benchUs * 1000 / 72 / 1000 < 4 ? 'ok' : 'warn',
         `${benchUs.toFixed(0)} µs per physics step, ${(benchUs * 1000 / 72 / 1000).toFixed(1)} ms per 72 Hz frame`);

let xrOK = false;
try { xrOK = !!navigator.xr && await navigator.xr.isSessionSupported('immersive-vr'); } catch (e) { logError(`WebXR: ${e.message}`); }
setCheck('chk-xr', xrOK ? 'ok' : 'fail', xrOK ? 'immersive VR available' : 'no immersive VR in this browser');

// --- three.js scene ---
const canvas = $('view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.xr.enabled = true;
renderer.xr.setReferenceSpaceType('local-floor');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1d2126);
const camera = new THREE.PerspectiveCamera(65, 1, 0.02, 40);
camera.position.set(0, 1.65, 0.9);
camera.lookAt(0, 0.85, -0.9);
const hemi = new THREE.HemisphereLight(0xf4ead8, 0x2e3a48, 1.5);
scene.add(hemi);
const sun = new THREE.DirectionalLight(0xfff1dc, 1.6);
sun.position.set(1.2, 3, 0.8);
scene.add(sun);

// painterly textures, baked on the GPU before anything is drawn
const tex = {};
{
  const t0 = performance.now();
  let seed = 11;
  for (const [name, col] of Object.entries(PALETTE)) {
    tex[name] = bakePainted(renderer, col, seed++,
      { ...(name in FLOW && { flow: FLOW[name] }), size: TEX_SIZE[name] ?? 512, ...(name === 'floor' && { grid: FLOOR }),
        amount: name.startsWith('box') || name.startsWith('hand') ? 1.25 : 1.0 });
  }
  renderer.getContext().finish();
  setCheck('chk-paint', 'ok', `${Object.keys(tex).length} textures painted in ${(performance.now() - t0).toFixed(0)} ms`);
}
// two lighting styles, switched with B or Y: soft (Lambert) and banded (toon, four steps)
const bands = new THREE.DataTexture(new Uint8Array([90, 150, 205, 255]), 4, 1, THREE.RedFormat);
bands.minFilter = bands.magFilter = THREE.NearestFilter;
bands.needsUpdate = true;
function painted(name) {
  const soft = new THREE.MeshLambertMaterial({ map: tex[name] });
  const banded = new THREE.MeshToonMaterial({ map: tex[name], gradientMap: bands });
  const m = { soft, banded };
  return m;
}
let lighting = 'soft';
const meshesStyled = [];
function styledMesh(geometry, name) {
  const mats = painted(name);
  const mesh = new THREE.Mesh(geometry, mats.soft);
  meshesStyled.push([mesh, mats]);
  return mesh;
}
function setLighting(style) {
  lighting = style;
  for (const [mesh, mats] of meshesStyled) mesh.material = mats[style];
  hemi.intensity = style === 'banded' ? 1.2 : 1.5;
  sun.intensity = style === 'banded' ? 2.2 : 1.6;
}

const mj = new THREE.Group();
mj.rotation.x = -Math.PI / 2;
scene.add(mj);

{
  const g = new THREE.PlaneGeometry(FLOOR, FLOOR), uv = g.attributes.uv;
  for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) * FLOOR / TILE.floor, uv.getY(i) * FLOOR / TILE.floor);
  const floor = styledMesh(g, 'floor');
  floor.position.z = -0.001;
  mj.add(floor);
}
STATIC.forEach(([name, half, pos], i) => {
  const m = styledMesh(boxUVs(new THREE.BoxGeometry(...half.map(h => 2 * h)), half, TILE[name], 7 + i), name);
  m.position.set(...pos);
  mj.add(m);
});

function textCanvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  return c;
}
const boxMeshes = sim.boxes.map((b, i) => {
  const mesh = styledMesh(boxUVs(new THREE.BoxGeometry(...b.half.map(h => 2 * h)), b.half, TILE.box, 100 + i), b.name);
  const c = textCanvas(256, 128), g = c.getContext('2d');
  g.fillStyle = b.mass >= 10 ? '#f3e6cf' : '#1a1209';           // light text on the dark, heavy boxes
  g.font = 'bold 84px system-ui, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
  g.fillText(`${b.mass} kg`, 128, 66);
  const label = new THREE.Mesh(new THREE.PlaneGeometry(1.6 * b.half[0], 0.8 * b.half[0]),
    new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(c), transparent: true }));
  label.position.z = b.half[2] + 0.001;
  mesh.add(label);
  mj.add(mesh);
  return mesh;
});

const handGeom = new THREE.BoxGeometry(...HAND_HALF.map(h => 2 * h));
const handMeshes = [0, 1].map(i => {
  const m = styledMesh(boxUVs(handGeom.clone(), HAND_HALF, TILE.hand, 50 + i), i ? 'hand_r' : 'hand_l');
  mj.add(m);
  return m;
});
const ghosts = [0, 1].map(() => {
  const g = new THREE.LineSegments(new THREE.EdgesGeometry(handGeom),
    new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.55 }));
  g.visible = false;
  scene.add(g);
  return g;
});
const tethers = [0, 1].map(() => {
  const geom = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(6), 3));
  const line = new THREE.Line(geom, new THREE.LineBasicMaterial({ color: 0xffffff }));
  line.visible = false;
  scene.add(line);
  return line;
});

// status board, standing beyond the table and facing the operator
const boardCanvas = textCanvas(1024, 400);
const boardTex = new THREE.CanvasTexture(boardCanvas);
const board = new THREE.Mesh(new THREE.PlaneGeometry(1.4, 0.547), new THREE.MeshBasicMaterial({ map: boardTex }));
board.position.set(0, 1.75, -1.6);
board.rotation.x = -0.12;
scene.add(board);
function drawBoard(lines) {
  const g = boardCanvas.getContext('2d');
  g.fillStyle = '#14181c'; g.fillRect(0, 0, 1024, 400);
  g.fillStyle = '#e8a33d'; g.font = 'bold 44px system-ui, sans-serif'; g.textBaseline = 'top';
  g.fillText(`VR Physics Lab ${VERSION}`, 36, 28);
  g.fillStyle = '#dfe3e6'; g.font = '32px ui-monospace, Menlo, monospace';
  lines.forEach((l, i) => g.fillText(l, 36, 100 + 46 * i));
  boardTex.needsUpdate = true;
}
drawBoard(['grip: grab the box under your hand', 'A or X: reset   B or Y: lighting', 'stick click: sound on/off']);

// --- controllers ---
const grips = [0, 1].map(i => {
  const g = renderer.xr.getControllerGrip(i);
  g.addEventListener('connected', e => { g.userData.source = e.data; });
  g.addEventListener('disconnected', () => { g.userData.source = null; });
  scene.add(g);
  return g;
});
const handOf = src => (src.handedness === 'left' ? 0 : 1);
const pulse = (src, value, ms) => {
  const a = src?.gamepad?.hapticActuators?.[0];
  if (a?.pulse) a.pulse(Math.min(1, value), ms)?.catch?.(() => {});
};

// --- state ---
const ctl = [0, 1].map(() => ({ prev: null, cur: null, place: true, squeeze: false, button: false,
                                 lastPulse: 0, wasSaturated: false, src: null }));
function park() {
  sim.hands.forEach((h, i) => sim.placeHand(h, PARK[i], IDENTITY));
  ctl.forEach(c => { c.prev = c.cur = null; c.place = true; });
}
let tracker = createContactTracker(sim), simT = 0;
function resetScene() { sim.reset(); park(); tracker = createContactTracker(sim); }

// sound: created on the first click (browsers start audio only after a gesture)
let audioCtx = null, audio = null, play = null, soundOn = true;
function ensureAudio() {
  if (!audioCtx) {
    audioCtx = new AudioContext({ latencyHint: 'interactive' });
    audio = createContactAudio(audioCtx);
    play = contactSounds(sim, audio);
  }
  return audioCtx.resume();
}
function updateListener() {
  if (!audio) return;
  const cam = renderer.xr.isPresenting ? renderer.xr.getCamera() : camera;
  const e = cam.matrixWorld.elements;
  audio.setListener(audioCtx.currentTime, [e[12], e[13], e[14]], [-e[8], -e[9], -e[10]], [e[4], e[5], e[6]]);
}
park();

let acc = 0, lastT = null, physMs = 0, physSteps = 0, frames = 0, fpsT = performance.now(), fps = 0, boardT = 0;

function stepPhysics(dt) {
  acc += Math.min(dt, MAX_STEPS * PARAMS.dt);
  const n = Math.min(Math.floor(acc / PARAMS.dt + 1e-9), MAX_STEPS);
  acc -= n * PARAMS.dt;
  const t0 = performance.now();
  for (let k = 1; k <= n; k++) {
    for (let i = 0; i < 2; i++) {
      const c = ctl[i];
      if (c.cur && c.prev) sim.setTarget(i, lerp(c.prev.pos, c.cur.pos, k / n), nlerp(c.prev.quat, c.cur.quat, k / n));
    }
    sim.step(1);
  }
  ctl.forEach(c => { if (c.cur) c.prev = c.cur; });
  simT += n * PARAMS.dt;
  const events = tracker(simT);
  if (play && soundOn && audioCtx.state === 'running') play(audioCtx.currentTime, events);
  physMs = 0.9 * physMs + 0.1 * (performance.now() - t0);
  physSteps = n;
}

function readControllers(now) {
  for (const g of grips) {
    const src = g.userData.source;
    if (!src || !g.visible) continue;
    const i = handOf(src), c = ctl[i], h = sim.hands[i];
    c.src = src;
    const pose = { pos: toMjPos(g.position), quat: toMjQuat(g.quaternion) };
    if (c.place) { sim.placeHand(h, pose.pos, pose.quat); c.prev = pose; c.place = false; }
    c.cur = pose;
    ghosts[i].visible = true;
    ghosts[i].position.copy(g.position);
    ghosts[i].quaternion.copy(g.quaternion);

    const b = src.gamepad?.buttons ?? [];
    const squeeze = !!b[1]?.pressed, button = !!b[4]?.pressed, button2 = !!b[5]?.pressed, stick = !!b[3]?.pressed;
    if (squeeze && !c.squeeze) { if (sim.grasp(i)) pulse(src, 0.6, 40); }
    if (!squeeze && c.squeeze) sim.release(i);
    if (button && !c.button) { resetScene(); pulse(src, 0.4, 30); }
    if (button2 && !c.button2) { setLighting(lighting === 'soft' ? 'banded' : 'soft'); pulse(src, 0.3, 30); }
    if (stick && !c.stick) { soundOn = !soundOn; if (!soundOn && audio) audio.rub(audioCtx.currentTime, []); pulse(src, 0.3, 30); }
    c.squeeze = squeeze; c.button = button; c.button2 = button2; c.stick = stick;

    const ratio = Math.hypot(...h.force) / PARAMS.fMax;   // haptics: spring force against the clamp
    if (h.saturated && !c.wasSaturated) pulse(src, 1, 120);
    else if (ratio > 0.08 && now - c.lastPulse > 60) { pulse(src, 0.15 + 0.85 * ratio, 70); c.lastPulse = now; }
    c.wasSaturated = h.saturated;
  }
}

const v3 = new THREE.Vector3();
function syncMeshes() {
  sim.boxes.forEach((b, i) => {
    const { pos, quat } = sim.pose(b.body);
    boxMeshes[i].position.set(...pos);
    boxMeshes[i].quaternion.set(quat[1], quat[2], quat[3], quat[0]);
  });
  sim.hands.forEach((h, i) => {
    const { pos, quat } = sim.pose(h.body);
    handMeshes[i].position.set(...pos);
    handMeshes[i].quaternion.set(quat[1], quat[2], quat[3], quat[0]);
    const t = tethers[i];
    t.visible = ghosts[i].visible;
    if (!t.visible) return;
    mj.localToWorld(v3.set(...pos));
    const a = t.geometry.attributes.position;
    a.setXYZ(0, ghosts[i].position.x, ghosts[i].position.y, ghosts[i].position.z);
    a.setXYZ(1, v3.x, v3.y, v3.z);
    a.needsUpdate = true;
    const r = Math.min(1, Math.hypot(...h.force) / PARAMS.fMax);
    t.material.color.setRGB(0.4 + 0.6 * r, 1 - 0.75 * r, 0.45 - 0.3 * r);
  });
}

function updateBoard(now) {
  frames++;
  if (now - fpsT > 500) { fps = 1000 * frames / (now - fpsT); frames = 0; fpsT = now; }
  if (now - boardT < 250) return;
  boardT = now;
  const hand = i => {
    const h = sim.hands[i];
    return `${i ? 'right' : 'left '} ${Math.hypot(...h.force).toFixed(0).padStart(3)} N` +
           `${h.saturated ? ' at limit' : ''}${h.held ? ` holding ${h.held.mass} kg` : ''}`;
  };
  drawBoard([`${fps.toFixed(0)} fps   physics ${physMs.toFixed(1)} ms (${physSteps} steps)`,
             hand(0), hand(1), `light ${lighting}   sound ${soundOn ? 'on' : 'off'}`, 'grip: grab  A/X: reset  B/Y: light']);
}

function resize() {
  if (renderer.xr.isPresenting) return;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

renderer.setAnimationLoop((time) => {
  const now = performance.now();
  const dt = lastT === null ? 0 : (time - lastT) / 1000;
  lastT = time;
  if (renderer.xr.isPresenting) readControllers(now);
  stepPhysics(dt);
  updateListener();
  syncMeshes();
  updateBoard(now);
  renderer.render(scene, camera);
});

// --- entering and leaving VR ---
const enter = $('enter');
enter.disabled = !xrOK;
enter.addEventListener('click', async () => {
  try {
    ensureAudio();
    const session = await navigator.xr.requestSession('immersive-vr', { optionalFeatures: ['local-floor', 'bounded-floor'] });
    session.addEventListener('end', () => {
      ghosts.forEach(g => { g.visible = false; });
      grips.forEach(g => { g.userData.source = null; });
      park();
      enter.textContent = 'Enter VR';
      resize();
    });
    await renderer.xr.setSession(session);
    if (session.updateTargetFrameRate && session.supportedFrameRates?.includes(72)) {
      session.updateTargetFrameRate(72).catch(() => {});
    }
    resetScene();
    lastT = null;
    enter.textContent = 'In VR';
  } catch (e) {
    logError(`entering VR failed: ${e.message}`);
  }
});

// sound check without the headset on: knocks ahead, right, behind and left of the preview camera
$('soundcheck').addEventListener('click', async () => {
  await ensureAudio();
  updateListener();
  const t = audioCtx.currentTime + 0.1, mats = [{ bright: 0.65, gain: 1 }, { bright: 0.75, gain: 1 }];
  const e = camera.matrixWorld.elements, p = [e[12], e[13], e[14]];
  [[0, -1.5], [1.5, 0], [0, 1.5], [-1.5, 0]].forEach(([x, z], i) =>
    audio.impact(t + 0.6 * i, [p[0] + x, p[1] - 0.4, p[2] + z], 2.0, 4, 0.12, mats));
});
