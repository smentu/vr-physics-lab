// Translucency room (sss.html): a fully dark room, one table, and the four media of sss.js in each of the
// three shapes, plus the low-poly Stanford bunny (bunny.js; its thickness comes from a signed distance field),
// lit by a single lamp that the operator can pick up. The trigger switches the shading
// technique, so the same objects can be compared live. Same spring hands and MuJoCo scene code as the lab
// (sim_core.js, with this page's own object list); no sound.
// Frames: three / WebXR is Y up with -Z ahead; MuJoCo is Z up with +Y ahead. MuJoCo bodies live in a group
// rotated -90 deg about X.
import * as THREE from 'three';
import { createSim, PARAMS, HAND_HALF, qmul } from './sim_core.js';
import { bakePainted, boxUVs } from './paint.js';
import { MEDIA, SSS_MODES, sssMaterials, sssGeometry } from './sss.js';
import { BUNNY } from './bunny.js';

const VERSION = 'v6';
const MUJOCO_URL = '../engine/mujoco-3.12.0/mujoco.js';
const MAX_STEPS = 34;
const PARK = [[-0.25, 0.15, 1.0], [0.25, 0.15, 1.0]];
const IDENTITY = [1, 0, 0, 0];
const TOP = 0.75;                                            // table top height, m
const TABLE = ['table', [0.85, 0.3, TOP / 2], [0, 0.65, TOP / 2]];
// columns by medium, rows by shape (front to back)
const MEDIUM_X = { wax: -0.42, jade: -0.14, gummy: 0.14, skin: 0.42 };
const SHAPES = { sphere: [0.45, [0.04]], box: [0.65, [0.035, 0.035, 0.035]], cylinder: [0.85, [0.03, 0.06]] };
const DENSITY = { wax: 900, jade: 3300, gummy: 1200, skin: 1000 };     // kg/m3
const SPEC = { wax: 0.35, jade: 0.6, gummy: 0.9, skin: 0.3 };         // glossy coat
// the lamp floats (gravity compensated, damped) and starts 40 cm above the middle of the table
const LAMP = { r: 0.02, mass: 0.15, start: [0, 0.65, TOP + 0.4], color: 0xfff0dc, intensity: 0.3 };
const ORBIT = { cx: 0, cy: 0.65, rx: 0.28, ry: 0.2, z: TOP + 0.05, period: 24 };   // lamp path in the preview
const BUNNY_POS = [0.67, 0.62, TOP + BUNNY.half[2] + 0.002];
const DEFAULT_MODE = 4;                                      // diffusion + single scattering
const query = new URLSearchParams(location.search);

const volume = (shape, s) => (shape === 'sphere' ? 4 / 3 * Math.PI * s[0] ** 3
  : shape === 'cylinder' ? 2 * Math.PI * s[0] ** 2 * s[1] : 8 * s[0] * s[1] * s[2]);
const lift = (shape, s) => (shape === 'box' ? s[2] : shape === 'cylinder' ? s[1] : s[0]);
const OBJECTS = Object.entries(MEDIUM_X).flatMap(([m, x]) => Object.entries(SHAPES).map(([shape, [y, s]]) =>
  [`${m}_${shape}`, +(DENSITY[m] * volume(shape, s)).toFixed(3), s, [x, y, TOP + lift(shape, s) + 0.001], shape]));
OBJECTS.push(['bunny', +(DENSITY.jade * BUNNY.volume).toFixed(3), BUNNY.half, BUNNY_POS, 'mesh']);
OBJECTS.push(['lamp', LAMP.mass, [LAMP.r], LAMP.start, 'sphere', { gravcomp: 1, damping: 0.3 }]);

const $ = id => document.getElementById(id);
function setCheck(id, state, text) {
  const row = $(id);
  row.dataset.state = state;
  row.querySelector('.value').textContent = text;
}
function logError(msg) { const el = $('log'); el.hidden = false; el.textContent += msg + '\n'; }
window.addEventListener('error', e => logError(`error: ${e.message}`));
window.addEventListener('unhandledrejection', e => logError(`error: ${e.reason?.message ?? e.reason}`));

const Q_XR_TO_MJ = [Math.SQRT1_2, Math.SQRT1_2, 0, 0];
const toMjPos = p => [p.x, -p.z, p.y];
const toMjQuat = q => qmul(Q_XR_TO_MJ, [q.w, q.x, q.y, q.z]);
const lerp = (a, b, t) => a.map((x, i) => x + (b[i] - x) * t);
function nlerp(a, b, t) {
  const d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  const q = lerp(a, d < 0 ? b.map(x => -x) : b, t), n = Math.hypot(...q);
  return q.map(x => x / n);
}

// --- physics ---
let mujoco, sim;
try {
  const { default: loadMujoco } = await import(MUJOCO_URL);
  mujoco = await loadMujoco();
  sim = createSim(mujoco, PARAMS, { boxes: OBJECTS, statics: [TABLE], meshes: { bunny: BUNNY } });
  setCheck('chk-mujoco', 'ok', `MuJoCo ${mujoco.mj_versionString?.() ?? '3.12.0'} loaded, ${OBJECTS.length - 1} objects and a lamp`);
} catch (e) {
  setCheck('chk-mujoco', 'fail', 'MuJoCo did not load');
  logError(`MuJoCo: ${e?.message ?? e}`);
  throw e;
}
let xrOK = false;
try { xrOK = !!navigator.xr && await navigator.xr.isSessionSupported('immersive-vr'); } catch (e) { logError(`WebXR: ${e.message}`); }
setCheck('chk-xr', xrOK ? 'ok' : 'fail', xrOK ? 'immersive VR available' : 'no immersive VR in this browser');

// --- renderer and the dark room: no ambient, no sky, one point light in the lamp ---
const canvas = $('view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.xr.enabled = true;
renderer.xr.setReferenceSpaceType('local-floor');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);
const camera = new THREE.PerspectiveCamera(55, 1, 0.02, 40);
camera.position.set(0.1, 1.12, 0.02);
camera.lookAt(0.1, 0.9, -0.64);
const mj = new THREE.Group();
mj.rotation.x = -Math.PI / 2;
scene.add(mj);

const t0 = performance.now();
const paint = (col, seed, opts = {}) => new THREE.MeshLambertMaterial({ map: bakePainted(renderer, col, seed, opts) });
{
  const g = new THREE.PlaneGeometry(10, 10);
  const floor = new THREE.Mesh(g, paint('#3a3f45', 21, { size: 1024, flow: 0.45, grid: 10 }));
  floor.position.z = -0.001;
  mj.add(floor);
  const [, half, pos] = TABLE;
  const table = new THREE.Mesh(boxUVs(new THREE.BoxGeometry(...half.map(h => 2 * h)), half, 2.0, 7), paint('#5e4030', 22, { flow: 0.25 }));
  table.position.set(...pos);
  mj.add(table);
}
const handGeom = new THREE.BoxGeometry(...HAND_HALF.map(h => 2 * h));
const handMeshes = ['#3fb6c4', '#e8a33d'].map((c, i) => {
  const m = new THREE.Mesh(boxUVs(handGeom.clone(), HAND_HALF, 0.5, 50 + i), paint(c, 30 + i, { amount: 1.25 }));
  mj.add(m);
  return m;
});
renderer.getContext().finish();
const paintMs = performance.now() - t0;

// translucent objects: one material per technique, switched together
let mode = Math.min(SSS_MODES.length - 1, Math.max(0, +(query.get('mode') ?? DEFAULT_MODE)));
const sssMeshes = [];
const t1 = performance.now();
let lamp = null, lampLight = null, bunny = null;
let bunnyMedium = query.get('bunny') in MEDIA ? query.get('bunny') : 'jade';
const bodyMeshes = sim.boxes.map(b => {
  let mesh;
  if (b.name === 'lamp') {
    mesh = new THREE.Mesh(new THREE.SphereGeometry(LAMP.r, 32, 16), new THREE.MeshBasicMaterial({ color: LAMP.color, toneMapped: false }));
    lampLight = new THREE.PointLight(LAMP.color, LAMP.intensity, 0, 2);
    mesh.add(lampLight);
    lamp = { body: b, mesh };
  } else if (b.name === 'bunny') {
    const shape = { type: 'mesh', ...BUNNY };
    const mats = Object.fromEntries(Object.keys(MEDIA).map(m => [m, sssMaterials(m, shape, { spec: SPEC[m] })]));
    mesh = new THREE.Mesh(sssGeometry(shape), mats[bunnyMedium][mode]);
    bunny = { mesh, mats };
    sssMeshes.push([mesh, null]);
  } else {
    const medium = b.name.split('_')[0];
    const shape = { type: b.shape, half: b.shape === 'sphere' ? [b.size[0], 0, 0] : b.shape === 'cylinder' ? [b.size[0], 0, b.size[1]] : b.size };
    const mats = sssMaterials(medium, shape, { spec: SPEC[medium] });
    mesh = new THREE.Mesh(sssGeometry(shape), mats[mode]);
    sssMeshes.push([mesh, mats]);
  }
  mj.add(mesh);
  return mesh;
});
const tableMs = performance.now() - t1;

// labels on the table's front strip and a status board, both faintly self-lit (B or Y hides them)
function textCanvas(w, h) { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; }
const dim = new THREE.Color(0.35, 0.37, 0.4);
const overlays = [];
function drawLabel(label, text) {
  const c = label.material.map.image, g = c.getContext('2d');
  g.clearRect(0, 0, c.width, c.height);
  g.fillStyle = '#ffffff'; g.font = '600 40px system-ui, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
  g.fillText(text, c.width / 2, 34);
  label.material.map.needsUpdate = true;
}
function makeLabel(text, x, w = 0.12) {
  const label = new THREE.Mesh(new THREE.PlaneGeometry(w, 0.03),
    new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(textCanvas(256 * w / 0.12, 64)), transparent: true, color: dim, toneMapped: false }));
  label.position.set(x, 0.375, TOP + 0.0015);
  drawLabel(label, text);
  mj.add(label);
  overlays.push(label);
  return label;
}
for (const [m, x] of Object.entries(MEDIUM_X)) makeLabel(m, x);
const bunnyLabel = makeLabel(`bunny: ${bunnyMedium}`, BUNNY_POS[0], 0.2);
const boardCanvas = textCanvas(1024, 300);
const boardTex = new THREE.CanvasTexture(boardCanvas);
const board = new THREE.Mesh(new THREE.PlaneGeometry(1.1, 0.322),
  new THREE.MeshBasicMaterial({ map: boardTex, color: dim, toneMapped: false }));
board.position.set(0, 1.55, -1.45);
board.rotation.x = -0.1;
scene.add(board);
overlays.push(board);
function drawBoard(lines) {
  const g = boardCanvas.getContext('2d');
  g.fillStyle = '#000'; g.fillRect(0, 0, 1024, 300);
  g.fillStyle = '#e8a33d'; g.font = 'bold 40px system-ui, sans-serif'; g.textBaseline = 'top';
  g.fillText(`Translucency room ${VERSION}`, 30, 22);
  g.fillStyle = '#dfe3e6'; g.font = '30px ui-monospace, Menlo, monospace';
  lines.forEach((l, i) => g.fillText(l, 30, 88 + 42 * i));
  boardTex.needsUpdate = true;
}

function setMode(k) {
  mode = (k + SSS_MODES.length) % SSS_MODES.length;
  for (const [mesh, mats] of sssMeshes) mesh.material = (mats ?? bunny.mats[bunnyMedium])[mode];
  $('mode').textContent = `${mode + 1} / ${SSS_MODES.length}: ${SSS_MODES[mode]}`;
}
function setBunny(m) {
  bunnyMedium = m;
  bunny.mesh.material = bunny.mats[m][mode];
  drawLabel(bunnyLabel, `bunny: ${m}`);
}
const startMode = mode;
for (let k = 0; k < SSS_MODES.length; k++) { setMode(k); renderer.compile(scene, camera); }   // no stall on first switch
setMode(startMode);
setCheck('chk-tables', 'ok', `floor and table painted in ${paintMs.toFixed(0)} ms; tables and ${SSS_MODES.length} shaders ready in ${(performance.now() - t1).toFixed(0)} ms`);

// --- controllers ---
const grips = [0, 1].map(i => {
  const g = renderer.xr.getControllerGrip(i);
  g.addEventListener('connected', e => { g.userData.source = e.data; });
  g.addEventListener('disconnected', () => { g.userData.source = null; });
  scene.add(g);
  return g;
});
const pulse = (src, value, ms) => { const a = src?.gamepad?.hapticActuators?.[0]; if (a?.pulse) a.pulse(Math.min(1, value), ms)?.catch?.(() => {}); };
const ghosts = [0, 1].map(() => {
  const g = new THREE.LineSegments(new THREE.EdgesGeometry(handGeom), new THREE.LineBasicMaterial({ color: 0x8c969f, transparent: true, opacity: 0.5 }));
  g.visible = false;
  scene.add(g);
  return g;
});
const ctl = [0, 1].map(() => ({ prev: null, cur: null, place: true }));
function park() {
  sim.hands.forEach((h, i) => sim.placeHand(h, PARK[i], IDENTITY));
  ctl.forEach(c => { c.prev = c.cur = null; c.place = true; });
}
park();

function readControllers(dt) {
  for (const g of grips) {
    const src = g.userData.source;
    if (!src || !g.visible) continue;
    const i = src.handedness === 'left' ? 0 : 1, c = ctl[i], h = sim.hands[i];
    const pose = { pos: toMjPos(g.position), quat: toMjQuat(g.quaternion) };
    if (c.place) { sim.placeHand(h, pose.pos, pose.quat); c.prev = pose; c.place = false; }
    c.cur = pose;
    ghosts[i].visible = true;
    ghosts[i].position.copy(g.position);
    ghosts[i].quaternion.copy(g.quaternion);
    const b = src.gamepad?.buttons ?? [], ax = src.gamepad?.axes ?? [];
    const now = { trigger: !!b[0]?.pressed, squeeze: !!b[1]?.pressed, stick: !!b[3]?.pressed, a: !!b[4]?.pressed, b: !!b[5]?.pressed };
    if (now.stick && !c.stick) { const ms = Object.keys(MEDIA); setBunny(ms[(ms.indexOf(bunnyMedium) + 1) % ms.length]); pulse(src, 0.3, 30); }
    if (now.trigger && !c.trigger) { setMode(mode + (i ? 1 : -1)); pulse(src, 0.3, 30); }
    if (now.squeeze && !c.squeeze) { if (sim.grasp(i)) pulse(src, 0.6, 40); }
    if (!now.squeeze && c.squeeze) sim.release(i);
    if (now.a && !c.a) { sim.reset(); park(); pulse(src, 0.4, 30); }
    if (now.b && !c.b) { overlays.forEach(o => { o.visible = !o.visible; }); pulse(src, 0.3, 30); }
    Object.assign(c, now);
    const stick = ax[3] ?? 0;                     // thumbstick forward (negative) brightens the lamp
    if (Math.abs(stick) > 0.15) lampLight.intensity = Math.min(2, Math.max(0.005, lampLight.intensity * Math.exp(-1.5 * stick * dt)));
  }
}

let acc = 0;
function stepPhysics(dt) {
  acc += Math.min(dt, MAX_STEPS * PARAMS.dt);
  const n = Math.min(Math.floor(acc / PARAMS.dt + 1e-9), MAX_STEPS);
  acc -= n * PARAMS.dt;
  for (let k = 1; k <= n; k++) {
    for (let i = 0; i < 2; i++) {
      const c = ctl[i];
      if (c.cur && c.prev) sim.setTarget(i, lerp(c.prev.pos, c.cur.pos, k / n), nlerp(c.prev.quat, c.cur.quat, k / n));
    }
    sim.step(1);
  }
  ctl.forEach(c => { if (c.cur) c.prev = c.cur; });
}

// in the 2D preview the lamp hangs where the physics has it; ?orbit=1 makes it circle between the objects and
// ?lamp=<deg> fixes it at one point of that circle (physics ignores it in both cases)
const fixedAngle = query.has('lamp') ? +query.get('lamp') * Math.PI / 180 : null;
function syncMeshes(time) {
  sim.boxes.forEach((b, i) => {
    const { pos, quat } = sim.pose(b.body);
    bodyMeshes[i].position.set(...pos);
    bodyMeshes[i].quaternion.set(quat[1], quat[2], quat[3], quat[0]);
  });
  sim.hands.forEach((h, i) => {
    handMeshes[i].visible = renderer.xr.isPresenting;      // parked hands would block the 2D preview
    const { pos, quat } = sim.pose(h.body);
    handMeshes[i].position.set(...pos);
    handMeshes[i].quaternion.set(quat[1], quat[2], quat[3], quat[0]);
  });
  if (!renderer.xr.isPresenting && (fixedAngle !== null || query.has('orbit'))) {
    const a = fixedAngle ?? 2 * Math.PI * time / 1000 / ORBIT.period;
    lamp.mesh.position.set(ORBIT.cx + ORBIT.rx * Math.cos(a), ORBIT.cy + ORBIT.ry * Math.sin(a), ORBIT.z);
  }
}

// GPU time per frame (both eyes), where the browser exposes timer queries
const gl = renderer.getContext(), timer = gl.getExtension('EXT_disjoint_timer_query_webgl2');
const queries = [];
let gpuMs = null;
function gpuBegin() {
  if (!timer || queries.length > 4) return null;
  const q = gl.createQuery();
  gl.beginQuery(timer.TIME_ELAPSED_EXT, q);
  return q;
}
function gpuEnd(q) {
  if (!q) return;
  gl.endQuery(timer.TIME_ELAPSED_EXT);
  queries.push(q);
  while (queries.length && gl.getQueryParameter(queries[0], gl.QUERY_RESULT_AVAILABLE)) {
    const done = queries.shift();
    if (!gl.getParameter(timer.GPU_DISJOINT_EXT)) {
      const ms = gl.getQueryParameter(done, gl.QUERY_RESULT) / 1e6;
      gpuMs = gpuMs === null ? ms : 0.9 * gpuMs + 0.1 * ms;
    }
    gl.deleteQuery(done);
  }
}

let frames = 0, fpsT = performance.now(), fps = 0, boardT = 0;
function updateBoard(now) {
  frames++;
  if (now - fpsT > 500) { fps = 1000 * frames / (now - fpsT); frames = 0; fpsT = now; }
  if (now - boardT < 250) return;
  boardT = now;
  drawBoard([`${mode + 1}/${SSS_MODES.length} ${SSS_MODES[mode]}`,
             `lamp ${(lampLight.intensity * 1000).toFixed(0)} mcd   ${fps.toFixed(0)} fps   GPU ${gpuMs === null ? 'n/a' : gpuMs.toFixed(1) + ' ms'}`,
             'trigger R/L: next/previous   stick: lamp brightness',
             'grip: grab   stick click: bunny medium',
             'A/X: reset   B/Y: hide this board and the labels']);
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

let lastT = null;
renderer.setAnimationLoop((time) => {
  const dt = lastT === null ? 0 : (time - lastT) / 1000;
  lastT = time;
  if (renderer.xr.isPresenting) readControllers(dt);
  stepPhysics(dt);
  syncMeshes(time);
  updateBoard(performance.now());
  const q = gpuBegin();
  renderer.render(scene, camera);
  gpuEnd(q);
});
window.__ready = true;

// --- page controls ---
$('mode').addEventListener('click', () => setMode(mode + 1));
$('prev').addEventListener('click', () => setMode(mode - 1));
$('bunny').addEventListener('click', () => { const ms = Object.keys(MEDIA); setBunny(ms[(ms.indexOf(bunnyMedium) + 1) % ms.length]); $('bunny').textContent = `Bunny: ${bunnyMedium}`; });
window.addEventListener('keydown', e => { if (e.key === 'm' || e.key === 'ArrowRight') setMode(mode + 1); if (e.key === 'n' || e.key === 'ArrowLeft') setMode(mode - 1); });
const enter = $('enter');
enter.disabled = !xrOK;
enter.addEventListener('click', async () => {
  try {
    const session = await navigator.xr.requestSession('immersive-vr', { optionalFeatures: ['local-floor', 'bounded-floor'] });
    session.addEventListener('end', () => {
      ghosts.forEach(g => { g.visible = false; });
      grips.forEach(g => { g.userData.source = null; });
      park();
      enter.textContent = 'Enter VR';
      resize();
    });
    await renderer.xr.setSession(session);
    if (session.updateTargetFrameRate && session.supportedFrameRates?.includes(72)) session.updateTargetFrameRate(72).catch(() => {});
    sim.reset(); park();
    lastT = null;
    enter.textContent = 'In VR';
  } catch (e) {
    logError(`entering VR failed: ${e.message}`);
  }
});
