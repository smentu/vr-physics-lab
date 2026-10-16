// Hands Lab: the physics lab with articulated hands driven by controller-free hand tracking (WebXR hand input, 25
// joints per hand). The tracked joint positions become a wrist target for the clamped arm spring and 21 joint angles
// for the finger servos (hands_core.js); objects are held by friction. Physical push buttons on the table reset the
// scene and switch light and sound. Outside VR, a scripted right hand pinches and lifts the cup through the same
// retargeting path (demo).
// Frames: three / WebXR is Y up with -Z ahead; MuJoCo is Z up with +Y ahead. MuJoCo bodies live in a group rotated
// -90 deg about X, so MuJoCo poses render unchanged inside it.
import * as THREE from 'three';
import { createContactTracker, STATIC } from './sim_core.js';
import { createHandSim, HAND_PARAMS, PARK, PALM_DOWN, RELAXED, XR_JOINTS, BUTTON_R, BUTTON_H, TABLE_TOP } from './hands_core.js';
import { bakePainted, boxUVs } from './paint.js';
import { createContactAudio, contactSounds } from './audio.js';

const VERSION = 'v8';
const MUJOCO_URL = '../engine/mujoco-3.12.0/mujoco.js';   // vendored by deploy.sh (npm @mujoco/mujoco)
const DT = HAND_PARAMS.dt;
const MAX_STEPS = Math.round(0.034 / DT);    // physics steps per frame at most (34 ms of physics)
const BUDGET_MS = 5;                         // physics per 72 Hz frame above which the noslip passes are dropped
const PALETTE = {
  box1: '#dcc48e', box3: '#cfa660', box5: '#c08848', box10: '#a5683a', box20: '#85502f', box30: '#643a26',
  cup: '#b9c9d3', ball: '#c8524a', block: '#e0c25e', bar: '#7f9f69',
  table: '#5e4030', floor: '#3a3f45', hand_l: '#3fb6c4', hand_r: '#e8a33d',
  btn_reset: '#d65a50', btn_light: '#e8c33d', btn_sound: '#5fb57f',
};
const FLOW = { table: 0.25, floor: 0.45 };
const TILE = { table: 2.0, floor: 10.0, box: 1.0 };
const TEX_SIZE = { floor: 1024 };
const FLOOR = 10;
// tracked-skeleton bones as XR joint index pairs
const BONES = [[0, 1], [1, 2], [2, 3], [3, 4],
  ...[5, 10, 15, 20].flatMap(b => [[0, b], [b, b + 1], [b + 1, b + 2], [b + 2, b + 3], [b + 3, b + 4]])];

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
const toMjPos = p => [p.x, -p.z, p.y];
const mjToThree = p => [p[0], p[2], -p[1]];
const lerp = (a, b, t) => a.map((x, i) => x + (b[i] - x) * t);
function nlerp(a, b, t) {
  const d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  const bb = d < 0 ? b.map(x => -x) : b;
  const q = lerp(a, bb, t), n = Math.hypot(...q);
  return q.map(x => x / n);
}
const add = (a, b) => a.map((x, i) => x + b[i]);

// --- MuJoCo and the scene ---
let mujoco, sim;
try {
  const { default: loadMujoco } = await import(MUJOCO_URL);
  mujoco = await loadMujoco();
  sim = createHandSim(mujoco);
  setCheck('chk-mujoco', 'ok', `MuJoCo ${mujoco.mj_versionString?.() ?? '3.12.0'} loaded, ${sim.model.nv} degrees of freedom`);
} catch (e) {
  setCheck('chk-mujoco', 'fail', 'MuJoCo did not load');
  logError(`MuJoCo: ${e?.message ?? e}`);
  throw e;
}

// The demo: the right hand pinches the cup at its rim from above (fingertip IK), lifts, sways, puts it back, lets go.
// Its targets are made as tracking would make them: joint positions of the scripted pose, then retargeting.
function makeDemo(s) {
  const cup = s.boxes.find(b => b.name === 'cup');
  const C = add(s.pose(cup.body).pos, [0, 0, 0.025]);
  const W = add(C, [0.02, -0.11, 0.055]), up = add(W, [0, 0, 0.09]);
  const tips = gap => ({ thumb: add(C, [0, -(0.035 + gap), 0]), index: add(C, [-0.01, Math.sqrt(0.035 ** 2 - 0.01 ** 2) + gap, 0]),
                         middle: add(C, [0.012, Math.sqrt(0.035 ** 2 - 0.012 ** 2) + gap, 0]) });
  const curled = RELAXED.map((x, k) => (k >= 13 ? [0, 1.4, 1.6, 0.8][(k - 13) % 4] : x));
  const wide = s.solveTips(1, W, PALM_DOWN, tips(0.035), curled), squeeze = s.solveTips(1, W, PALM_DOWN, tips(-0.015), wide);
  const lifted = add(W, [0, 0, 0.15]);
  // keyframes: time (s), wrist, angles; the wrist sways sideways while lifted
  const keys = [[0, up, wide], [0.4, up, wide], [1.2, W, wide], [1.7, W, squeeze], [2.0, W, squeeze], [3.0, lifted, squeeze],
                [4.4, lifted, squeeze], [5.4, W, squeeze], [5.9, W, wide], [6.7, up, wide], [7.2, up, wide]];
  const period = keys[keys.length - 1][0];
  return {
    period,
    at(t) {
      t %= period;
      let k = 0;
      while (k < keys.length - 2 && t > keys[k + 1][0]) k++;
      const [t0, p0, q0] = keys[k], [t1, p1, q1] = keys[k + 1], u = Math.min(1, Math.max(0, (t - t0) / (t1 - t0)));
      const e = u * u * (3 - 2 * u);
      let pos = lerp(p0, p1, e);
      if (t > 3.0 && t < 4.4) pos = add(pos, [0.035 * Math.sin(2 * Math.PI * 1.2 * (t - 3.0)), 0, 0]);
      const q = lerp(q0, q1, e);
      return s.fkPoints(1, pos, PALM_DOWN, q);        // the joint positions tracking would report
    },
  };
}

// Cost: the demo's grasp and lift in a second simulation, timed over its contact-heavy part.
function benchmark() {
  const b = createHandSim(mujoco);
  const demo = makeDemo(b);
  b.placeHand(0, PARK[0], PALM_DOWN, RELAXED);
  let n = 0, ms = 0;
  for (let t = 0; t < 4.0; t += DT) {
    const r = b.retarget(1, demo.at(t));
    if (t === 0) b.placeHand(1, r.pos, r.quat, r.q); else b.setTarget(1, r.pos, r.quat, r.q);
    const t0 = performance.now();
    b.step(1);
    if (t > 1.5) { ms += performance.now() - t0; n++; }
  }
  b.dispose();
  return 1000 * ms / n;
}
const benchUs = benchmark();
const stepsPerFrame = 1 / 72 / DT;
let frameMs = benchUs * stepsPerFrame / 1000;
if (frameMs > BUDGET_MS) sim.model.opt.noslip_iterations = 0;
setCheck('chk-bench', frameMs <= BUDGET_MS ? 'ok' : 'warn',
  `${benchUs.toFixed(0)} µs per ${DT * 1000} ms physics step, ${frameMs.toFixed(1)} ms per 72 Hz frame` +
  (frameMs > BUDGET_MS ? '; slip correction off to keep the frame rate' : ''));

let xrOK = false;
try { xrOK = !!navigator.xr && await navigator.xr.isSessionSupported('immersive-vr'); } catch (e) { logError(`WebXR: ${e.message}`); }
setCheck('chk-xr', xrOK ? 'ok' : 'fail', xrOK ? 'immersive VR available; hand tracking is requested on entry' : 'no immersive VR in this browser');

// --- three.js scene ---
const canvas = $('view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.xr.enabled = true;
renderer.xr.setReferenceSpaceType('local-floor');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1d2126);
const camera = new THREE.PerspectiveCamera(60, 1, 0.02, 40);
camera.position.set(-0.05, 1.12, -0.02);         // the preview looks at the cup from the operator's right
camera.lookAt(-0.38, 0.84, -0.45);
const hemi = new THREE.HemisphereLight(0xf4ead8, 0x2e3a48, 1.5);
scene.add(hemi);
const sun = new THREE.DirectionalLight(0xfff1dc, 1.6);
sun.position.set(1.2, 3, 0.8);
scene.add(sun);

const tex = {};
{
  const t0 = performance.now();
  let seed = 11;
  for (const [name, col] of Object.entries(PALETTE)) {
    tex[name] = bakePainted(renderer, col, seed++,
      { ...(name in FLOW && { flow: FLOW[name] }), size: TEX_SIZE[name] ?? 512, ...(name === 'floor' && { grid: FLOOR }),
        amount: name === 'table' || name === 'floor' ? 1.0 : 1.25 });
  }
  renderer.getContext().finish();
  setCheck('chk-paint', 'ok', `${Object.keys(tex).length} textures painted in ${(performance.now() - t0).toFixed(0)} ms`);
}
const bands = new THREE.DataTexture(new Uint8Array([90, 150, 205, 255]), 4, 1, THREE.RedFormat);
bands.minFilter = bands.magFilter = THREE.NearestFilter;
bands.needsUpdate = true;
let lighting = 'soft';
const meshesStyled = [];
function styledMesh(geometry, name) {
  const mats = { soft: new THREE.MeshLambertMaterial({ map: tex[name] }),
                 banded: new THREE.MeshToonMaterial({ map: tex[name], gradientMap: bands }) };
  const mesh = new THREE.Mesh(geometry, mats[lighting]);
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
function labelPlane(text, w, h, fg) {
  const c = textCanvas(256, 128), g = c.getContext('2d');
  g.fillStyle = fg; g.font = 'bold 72px system-ui, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
  g.fillText(text, 128, 66);
  return new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(c), transparent: true }));
}
// objects: boxes keep their mass labels; round things are cylinders (MuJoCo axis z) and spheres
const objMeshes = sim.boxes.map((b, i) => {
  let geom;
  if (b.shape === 'sphere') geom = new THREE.SphereGeometry(b.size[0], 24, 16);
  else if (b.shape === 'cylinder') geom = new THREE.CylinderGeometry(b.size[0], b.size[0], 2 * b.size[1], 28).rotateX(Math.PI / 2);
  else geom = boxUVs(new THREE.BoxGeometry(...b.half.map(h => 2 * h)), b.half, TILE.box, 100 + i);
  const mesh = styledMesh(geom, b.name);
  if (b.name.startsWith('box')) {
    const label = labelPlane(`${b.mass} kg`, 1.6 * b.half[0], 0.8 * b.half[0], b.mass >= 10 ? '#f3e6cf' : '#1a1209');
    label.position.z = b.half[2] + 0.001;
    mesh.add(label);
  }
  mj.add(mesh);
  return mesh;
});
// push buttons on the table, labelled on top
const buttonMeshes = sim.buttons.map(btn => {
  const m = styledMesh(new THREE.CylinderGeometry(BUTTON_R, BUTTON_R, BUTTON_H, 28).rotateX(Math.PI / 2), `btn_${btn.name}`);
  m.position.set(btn.xy[0], btn.xy[1], TABLE_TOP + BUTTON_H / 2);
  const label = labelPlane(btn.name, 1.9 * BUTTON_R, 0.95 * BUTTON_R, '#14181c');
  label.position.z = BUTTON_H / 2 + 0.0008;
  m.add(label);
  mj.add(m);
  return m;
});
// hands: one mesh per MuJoCo geom (palm box, finger capsules), posed from geom_xpos / geom_xmat
const handGeoms = [];
for (const h of sim.hands) {
  for (const g of h.geoms) {
    const s = sim.model.geom_size, type = sim.model.geom_type[g];
    const geom = type === mujoco.mjtGeom.mjGEOM_CAPSULE.value
      ? new THREE.CapsuleGeometry(s[3 * g], 2 * s[3 * g + 1], 4, 10).rotateX(Math.PI / 2)
      : new THREE.BoxGeometry(2 * s[3 * g], 2 * s[3 * g + 1], 2 * s[3 * g + 2]);
    const mesh = styledMesh(geom, h.name);
    mj.add(mesh);
    handGeoms.push([g, mesh]);
  }
}
// the tracked hands: joint dots and bones (white), and a tether from the tracked wrist to the physics wrist
const ghosts = [0, 1].map(() => {
  const dots = new THREE.InstancedMesh(new THREE.SphereGeometry(0.0045, 8, 6),
    new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.7 }), XR_JOINTS.length);
  const bones = new THREE.LineSegments(new THREE.BufferGeometry().setAttribute('position',
    new THREE.Float32BufferAttribute(new Float32Array(6 * BONES.length), 3)),
    new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.45 }));
  const tether = new THREE.Line(new THREE.BufferGeometry().setAttribute('position',
    new THREE.Float32BufferAttribute(new Float32Array(6), 3)), new THREE.LineBasicMaterial({ color: 0xffffff }));
  for (const o of [dots, bones, tether]) { o.visible = false; o.frustumCulled = false; scene.add(o); }
  return { dots, bones, tether, pts: null };
});
const m4 = new THREE.Matrix4(), v3 = new THREE.Vector3();
function showGhost(i, pts) {                   // pts: 25 joint positions, MuJoCo frame
  const G = ghosts[i];
  G.pts = pts;
  const p3 = pts.map(mjToThree);
  p3.forEach((p, k) => { m4.makeTranslation(p[0], p[1], p[2]); G.dots.setMatrixAt(k, m4); });
  G.dots.instanceMatrix.needsUpdate = true;
  const a = G.bones.geometry.attributes.position;
  BONES.forEach(([u, v], k) => { a.setXYZ(2 * k, ...p3[u]); a.setXYZ(2 * k + 1, ...p3[v]); });
  a.needsUpdate = true;
  G.dots.visible = G.bones.visible = G.tether.visible = true;
}
function hideGhost(i) { const G = ghosts[i]; G.dots.visible = G.bones.visible = G.tether.visible = false; G.pts = null; }

// status board
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
  g.fillText(`Hands Lab ${VERSION}`, 36, 28);
  g.fillStyle = '#dfe3e6'; g.font = '30px ui-monospace, Menlo, monospace';
  lines.forEach((l, i) => g.fillText(l, 36, 96 + 44 * i));
  boardTex.needsUpdate = true;
}
drawBoard(['put the controllers down and use your hands', 'buttons on the table: reset, light, sound']);

// --- input: hands (and controllers, for their buttons) ---
const slots = [0, 1].map(k => {
  const ctrl = renderer.xr.getController(k), hand = renderer.xr.getHand(k);
  const slot = { src: null, hand, buttons: {} };
  ctrl.addEventListener('connected', e => { slot.src = e.data; });
  ctrl.addEventListener('disconnected', () => { slot.src = null; });
  scene.add(ctrl);
  return slot;
});
// per physics hand: targets this frame and last (interpolated across the physics steps), tracking state
const track = [0, 1].map(() => ({ prev: null, cur: null, place: true, state: 'not seen', lostT: 0 }));
function setFrameTarget(i, r) {
  const c = track[i];
  if (c.place) { sim.placeHand(i, r.pos, r.quat, r.q); c.prev = r; c.place = false; }
  c.cur = r;
}

function park() {
  sim.hands.forEach((h, i) => sim.placeHand(i, PARK[i], PALM_DOWN, RELAXED));
  track.forEach(c => { c.prev = c.cur = null; c.place = true; c.state = 'not seen'; });
  [0, 1].forEach(hideGhost);
}
let tracker = createContactTracker(sim), simT = 0;
function resetScene() { sim.reset(); park(); tracker = createContactTracker(sim); }

let audioCtx = null, audio = null, play = null, soundOn = true;
function ensureAudio() {
  if (!audioCtx) {
    audioCtx = new AudioContext({ latencyHint: 'interactive' });
    audio = createContactAudio(audioCtx);
    play = contactSounds(sim, audio);
  }
  return audioCtx.resume();
}
function setSound(on) { soundOn = on; if (!soundOn && audio) audio.rub(audioCtx.currentTime, []); }
function updateListener() {
  if (!audio) return;
  const cam = renderer.xr.isPresenting ? renderer.xr.getCamera() : camera;
  const e = cam.matrixWorld.elements;
  audio.setListener(audioCtx.currentTime, [e[12], e[13], e[14]], [-e[8], -e[9], -e[10]], [e[4], e[5], e[6]]);
}

let controllersSeen = false;
function readInputs(now) {
  const seen = [false, false];
  controllersSeen = false;
  for (const slot of slots) {
    const src = slot.src;
    if (!src) continue;
    const i = src.handedness === 'left' ? 0 : 1;
    if (src.hand) {
      const J = slot.hand.joints;
      const pts = XR_JOINTS.map(n => J[n]);
      if (pts.every(j => j && j.visible)) {
        const mjPts = pts.map(j => toMjPos(j.position));
        setFrameTarget(i, sim.retarget(i, mjPts));
        showGhost(i, mjPts);
        track[i].state = 'tracked';
        seen[i] = true;
      }
    } else if (src.gamepad) {                   // a controller: its buttons still work (A/X reset, B/Y light, stick sound)
      controllersSeen = true;
      const b = src.gamepad.buttons, prev = slot.buttons;
      const now4 = !!b[4]?.pressed, now5 = !!b[5]?.pressed, now3 = !!b[3]?.pressed;
      if (now4 && !prev[4]) resetScene();
      if (now5 && !prev[5]) setLighting(lighting === 'soft' ? 'banded' : 'soft');
      if (now3 && !prev[3]) setSound(!soundOn);
      slot.buttons = { 3: now3, 4: now4, 5: now5 };
    }
  }
  for (const i of [0, 1]) {                     // a hand out of view keeps its last pose
    const c = track[i];
    if (seen[i]) { c.lostT = now; continue; }
    if (c.state === 'tracked') c.state = 'lost';
    if (c.state === 'lost' && now - c.lostT > 600) hideGhost(i);
  }
}

// physical buttons: a press is a hand touching a button after at least 0.3 s untouched
const pressState = sim.buttons.map(() => ({ down: false, since: 0 }));
const ACTIONS = { reset: () => resetScene(), light: () => setLighting(lighting === 'soft' ? 'banded' : 'soft'),
                  sound: () => setSound(!soundOn) };
function pollButtons(t, contacts) {
  const handGeom = new Set(sim.hands.flatMap(h => [...h.geoms]));
  sim.buttons.forEach((btn, k) => {
    const touched = contacts.some(c => (c.g1 === btn.geom && handGeom.has(c.g2)) || (c.g2 === btn.geom && handGeom.has(c.g1)));
    const st = pressState[k];
    if (touched && !st.down && t - st.since > 0.3) {
      st.down = true; st.since = t;
      buttonMeshes[k].position.z = TABLE_TOP + BUTTON_H / 2 - 0.003;
      ACTIONS[btn.name]();
    } else if (!touched && st.down) {
      st.down = false; st.since = t;
      buttonMeshes[k].position.z = TABLE_TOP + BUTTON_H / 2;
    }
  });
}

// demo outside VR
let demo = makeDemo(sim), demoOn = true, demoT = 0;
const DEMO_PARK_L = [0.35, 0.25, 1.05];              // the idle left hand, out of the preview's view
function demoStep(dt) {
  demoT += dt;
  if (demoT >= demo.period) { demoT = 0; resetScene(); }
  if (track[0].place) { sim.placeHand(0, DEMO_PARK_L, PALM_DOWN, RELAXED); track[0].place = false; }
  const pts = demo.at(demoT);
  setFrameTarget(1, sim.retarget(1, pts));
  showGhost(1, pts);
  track[1].state = 'demo';
}

park();
window.handsLab = { sim, track, demoTime: () => demoT };   // for chrome://inspect and the headless check
let acc = 0, lastT = null, physMs = 0, physSteps = 0, frames = 0, fpsT = performance.now(), fps = 0, boardT = 0;
let touching = [new Set(), new Set()];

function stepPhysics(dt) {
  acc += Math.min(dt, MAX_STEPS * DT);
  const n = Math.min(Math.floor(acc / DT + 1e-9), MAX_STEPS);
  acc -= n * DT;
  const t0 = performance.now();
  for (let k = 1; k <= n; k++) {
    for (let i = 0; i < 2; i++) {
      const c = track[i];
      if (c.cur && c.prev) sim.setTarget(i, lerp(c.prev.pos, c.cur.pos, k / n), nlerp(c.prev.quat, c.cur.quat, k / n),
                                         lerp(c.prev.q, c.cur.q, k / n));
    }
    sim.step(1);
  }
  track.forEach(c => { if (c.cur) c.prev = c.cur; });
  simT += n * DT;
  physMs = 0.9 * physMs + 0.1 * (performance.now() - t0);
  physSteps = n;
  if (!n) return;
  const contacts = sim.contacts();
  pollButtons(simT, contacts);
  touching = sim.hands.map(h => new Set(contacts.flatMap(c =>
    h.geoms.has(c.g1) && !h.geoms.has(c.g2) ? [sim.geomName[c.g2]] : h.geoms.has(c.g2) && !h.geoms.has(c.g1) ? [sim.geomName[c.g1]] : [])));
  const events = tracker(simT);
  if (play && soundOn && audioCtx.state === 'running') play(audioCtx.currentTime, events);
}

const mat3 = new THREE.Matrix4();
function syncMeshes() {
  sim.boxes.forEach((b, i) => {
    const { pos, quat } = sim.pose(b.body);
    objMeshes[i].position.set(...pos);
    objMeshes[i].quaternion.set(quat[1], quat[2], quat[3], quat[0]);
  });
  const gx = sim.data.geom_xpos, gm = sim.data.geom_xmat;
  for (const [g, mesh] of handGeoms) {
    mesh.position.set(gx[3 * g], gx[3 * g + 1], gx[3 * g + 2]);
    const m = 9 * g;
    mat3.set(gm[m], gm[m + 1], gm[m + 2], 0, gm[m + 3], gm[m + 4], gm[m + 5], 0, gm[m + 6], gm[m + 7], gm[m + 8], 0, 0, 0, 0, 1);
    mesh.quaternion.setFromRotationMatrix(mat3);
  }
  sim.hands.forEach((h, i) => {
    const G = ghosts[i];
    if (!G.pts) return;
    const a = G.tether.geometry.attributes.position;
    a.setXYZ(0, ...mjToThree(G.pts[0]));
    mj.localToWorld(v3.set(...sim.pose(h.body).pos));
    a.setXYZ(1, v3.x, v3.y, v3.z);
    a.needsUpdate = true;
    const r = Math.min(1, Math.hypot(...h.force) / HAND_PARAMS.fMax);
    G.tether.material.color.setRGB(0.4 + 0.6 * r, 1 - 0.75 * r, 0.45 - 0.3 * r);
  });
}

function updateBoard(now) {
  frames++;
  if (now - fpsT > 500) { fps = 1000 * frames / (now - fpsT); frames = 0; fpsT = now; }
  if (now - boardT < 250) return;
  boardT = now;
  const hand = i => {
    const h = sim.hands[i], things = [...touching[i]].filter(n => !n.startsWith('btn_'));
    return `${i ? 'right' : 'left '} ${track[i].state.padEnd(8)} arm ${Math.hypot(...h.force).toFixed(0).padStart(3)} N` +
           `${h.saturated ? '!' : ' '} grip ${(100 * sim.grip(i)).toFixed(0).padStart(2)} %` + (things.length ? `  ${things.join(' ')}` : '');
  };
  drawBoard([`${fps.toFixed(0)} fps  physics ${physMs.toFixed(1)} ms (${physSteps} steps)  slip fix ${sim.model.opt.noslip_iterations ? 'on' : 'off'}`,
             hand(0), hand(1),
             controllersSeen ? 'controllers in use: put them down to use your hands' : `light ${lighting}   sound ${soundOn ? 'on' : 'off'}`,
             'buttons on the table: reset, light, sound']);
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
  const dt = lastT === null ? 0 : Math.min((time - lastT) / 1000, 0.1);
  lastT = time;
  if (renderer.xr.isPresenting) readInputs(now);
  else if (demoOn) demoStep(Math.min(dt, MAX_STEPS * DT));   // the script keeps pace with the physics
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
    const session = await navigator.xr.requestSession('immersive-vr',
      { optionalFeatures: ['local-floor', 'bounded-floor', 'hand-tracking'] });
    session.addEventListener('end', () => {
      slots.forEach(s => { s.src = null; });
      resetScene();
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
$('demo').addEventListener('click', () => {
  demoOn = !demoOn;
  $('demo').textContent = demoOn ? 'Pause demo' : 'Play demo';
  if (!demoOn) track[1].state = 'not seen';
  else { demoT = 0; resetScene(); }
});
$('soundcheck').addEventListener('click', async () => {
  await ensureAudio();
  updateListener();
  const t = audioCtx.currentTime + 0.1, mats = [{ bright: 0.65, gain: 1 }, { bright: 0.75, gain: 1 }];
  const e = camera.matrixWorld.elements, p = [e[12], e[13], e[14]];
  [[0, -1.5], [1.5, 0], [0, 1.5], [-1.5, 0]].forEach(([x, z], i) =>
    audio.impact(t + 0.6 * i, [p[0] + x, p[1] - 0.4, p[2] + z], 2.0, 4, 0.12, mats));
});
