// Articulated physics hands for the VR Physics Lab, driven by WebXR hand tracking (25 joints per hand) on MuJoCo
// WASM. Shared by the headset page (hands_app.js) and the Node test (test_hands_core.mjs).
// - Each hand is a free palm body pulled to the tracked wrist by the lab's clamped spring (the arm: 200 N, 15 N m).
// - Five fingers hang off the palm on hinge joints: two at each knuckle (sideways, bend), one at each of the other
//   finger joints; the thumb has two at its base, two at its knuckle and one at its end joint (21 per hand).
// - Torque-limited position servos drive the joints toward angles read off the tracked joint positions. The
//   servo limits are the finger strength: a finger pressed into an object stops at the surface and squeezes.
// - Fingers and palm collide with the scene (not with each other). Objects are held by friction alone, no welds.
// Frames: MuJoCo Z up, +Y ahead, metres (as sim_core.js). The palm frame: origin at the wrist, +y toward the middle
// knuckle, +z out of the palm, +x = y x z, which is the thumb side of the right hand and the little-finger side of the
// left. The left hand is the right one mirrored in x.
import { STATIC, BOXES, bodiesXML, qmul, qconj, rot, qrotvec, clampNorm } from './sim_core.js';

export const HAND_PARAMS = {
  dt: 0.002,            // physics step, s (the arm and the servos are damped implicitly; 2 ms is stable)
  kLin: 3000,           // N/m, the arm (as the lab)
  fMax: 200,            // N per hand
  kRot: 30,             // N m/rad
  tMax: 15,             // N m per hand
  loadMass: 0.5,        // kg: the arm's damping is critical for the hand plus this load
  loadInertia: 1e-3,    // kg m^2, likewise for the wrist
  teleport: 0.4,        // m: a tracked wrist this far from its physics hand moves the hand there instead of pulling
  // elliptic friction cones, stiff friction (impratio) and two noslip passes: no creep under a held load
  option: 'cone="elliptic" impratio="10" noslip_iterations="2"',
  flags: '',            // extra MJCF <flag> attributes
};

// Things for the fingers, on the near half of the table; the lab's boxes stay where they were.
// Same format as sim_core.js BOXES: name, mass kg, size m, position, shape.
export const SMALL = [
  ['cup', 0.3, [0.035, 0.05], [-0.40, 0.45, 0.801], 'cylinder'],
  ['ball', 0.15, [0.035], [-0.22, 0.45, 0.786], 'sphere'],
  ['block', 0.12, [0.025, 0.025, 0.025], [-0.06, 0.45, 0.776]],
  ['bar', 0.25, [0.12, 0.015, 0.015], [0.15, 0.45, 0.766]],
];
export const HAND_SCENE = { boxes: [...BOXES, ...SMALL], statics: STATIC };
// push buttons standing on the table's near right corner: name, centre x y (MuJoCo frame)
export const BUTTONS = [['reset', [0.40, 0.42]], ['light', [0.48, 0.42]], ['sound', [0.56, 0.42]]];
export const BUTTON_R = 0.028, BUTTON_H = 0.008, TABLE_TOP = 0.75;

// The WebXR hand joints in XRHand order.
const FINGER_XR = ['index-finger', 'middle-finger', 'ring-finger', 'pinky-finger'];
export const XR_JOINTS = ['wrist', 'thumb-metacarpal', 'thumb-phalanx-proximal', 'thumb-phalanx-distal', 'thumb-tip',
  ...FINGER_XR.flatMap(f => ['metacarpal', 'phalanx-proximal', 'phalanx-intermediate', 'phalanx-distal', 'tip']
    .map(j => `${f}-${j}`))];

// --- the hand model (right hand, palm frame) ---
// An adult hand of about 19 cm. name, knuckle position, segment lengths (to the next joint; the last to the tip), radii.
const FINGERS = [
  ['index', [0.024, 0.093, 0], [0.040, 0.024, 0.022], [0.0095, 0.0085, 0.0078]],
  ['middle', [0.0, 0.095, 0], [0.044, 0.028, 0.023], [0.0098, 0.0088, 0.0080]],
  ['ring', [-0.019, 0.089, 0], [0.041, 0.027, 0.023], [0.0092, 0.0083, 0.0076]],
  ['little', [-0.036, 0.080, 0], [0.032, 0.020, 0.021], [0.0082, 0.0074, 0.0070]],
];
// The thumb's base frame: its +y along the relaxed thumb bone, its +z toward the pad (the bend direction).
const THUMB = { base: [0.020, 0.028, 0.005], dir: [0.6, 0.75, 0.15], pad: [-0.6, 0, 0.8],
                len: [0.044, 0.032, 0.028], rad: [0.012, 0.0105, 0.0095] };
const PALM = { center: [-0.005, 0.047, 0], half: [0.039, 0.045, 0.014], mass: 0.30 };
const SEG_MASS = [0.020, 0.012, 0.008], THUMB_MASS = [0.040, 0.020, 0.012];
// joint kinds: range (rad), strength (N m), damping (N m s/rad)
const KIND = {
  abd: [[-0.4, 0.4], 0.5, 0.02], mcp: [[-0.5, 1.65], 1.6, 0.04], pip: [[-0.1, 1.9], 0.9, 0.015],
  dip: [[-0.3, 1.5], 0.45, 0.006],
  tcmc_abd: [[-1.0, 1.0], 1.6, 0.04], tcmc: [[-1.0, 1.0], 1.6, 0.04],
  tmcp_abd: [[-0.4, 0.4], 0.5, 0.02], tmcp: [[-0.4, 1.1], 1.0, 0.02], tip: [[-0.4, 1.5], 0.5, 0.008],
};
const SATURATE = 0.35;   // rad of servo error at which a joint reaches its strength
const ARMATURE = 2e-5;   // kg m^2 per finger joint, keeps the stiff servos on light segments stable
// The 21 joints of a hand in servo order: thumb, then index, middle, ring, little.
export const HAND_JOINTS = [['thumb_cmc_abd', 'tcmc_abd'], ['thumb_cmc', 'tcmc'], ['thumb_mcp_abd', 'tmcp_abd'],
  ['thumb_mcp', 'tmcp'], ['thumb_ip', 'tip'],
  ...FINGERS.flatMap(([f]) => [[`${f}_abd`, 'abd'], [`${f}_mcp`, 'mcp'], [`${f}_pip`, 'pip'], [`${f}_dip`, 'dip']])];
export const JOINT_RANGE = HAND_JOINTS.map(([, k]) => KIND[k][0]);
// A loosely curled hand, for a hand that has not been tracked yet.
export const RELAXED = HAND_JOINTS.map(([n, k]) => (k === 'mcp' ? 0.25 : k === 'pip' ? 0.35 : k === 'dip' ? 0.2
  : k === 'tmcp' || k === 'tip' ? 0.2 : 0));

// fingertip index in XR_JOINTS and each finger's joints in servo order
export const TIPS = { thumb: 4, index: 9, middle: 14, ring: 19, little: 24 };
export const FINGER_DOFS = { thumb: [0, 1, 2, 3, 4], index: [5, 6, 7, 8], middle: [9, 10, 11, 12], ring: [13, 14, 15, 16],
                             little: [17, 18, 19, 20] };

// --- small vector and quaternion helpers (wxyz) ---
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = a => { const n = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / n, a[1] / n, a[2] / n]; };
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
function solve3(A, b) {                           // A x = b for a 3x3 A (Cramer)
  const d = M => M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
               + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
  const D = d(A);
  return [0, 1, 2].map(c => d(A.map((row, r) => row.map((x, k) => (k === c ? b[r] : x)))) / D);
}
const qaxis = (ax, a) => { const s = Math.sin(a / 2); return [Math.cos(a / 2), ax[0] * s, ax[1] * s, ax[2] * s]; };
export function quatFromAxes(x, y, z) {          // rotation whose columns are the unit vectors x, y, z
  const [m00, m10, m20] = x, [m01, m11, m21] = y, [m02, m12, m22] = z;
  const tr = m00 + m11 + m22;
  let q;
  if (tr > 0) { const s = 2 * Math.sqrt(tr + 1); q = [s / 4, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s]; }
  else if (m00 > m11 && m00 > m22) { const s = 2 * Math.sqrt(1 + m00 - m11 - m22); q = [(m21 - m12) / s, s / 4, (m01 + m10) / s, (m02 + m20) / s]; }
  else if (m11 > m22) { const s = 2 * Math.sqrt(1 + m11 - m00 - m22); q = [(m02 - m20) / s, (m01 + m10) / s, s / 4, (m12 + m21) / s]; }
  else { const s = 2 * Math.sqrt(1 + m22 - m00 - m11); q = [(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, s / 4]; }
  const n = Math.hypot(...q);
  return q.map(c => c / n);
}
const THUMB_Q = (() => {                          // the right thumb's base frame in the palm frame
  const y = norm(THUMB.dir), z = norm(sub(THUMB.pad, y.map(c => c * dot(THUMB.pad, y))));
  return quatFromAxes(cross(y, z), y, z);
})();
// A side's geometry: the left hand mirrors positions in x, the thumb frame (q -> w, x, -y, -z) and the sideways axes.
function sideSpec(side) {
  const m = side === 'l', mx = v => (m ? [-v[0], v[1], v[2]] : v);
  return { mirror: m, sA: m ? -1 : 1, mx, thumbQ: m ? [THUMB_Q[0], THUMB_Q[1], -THUMB_Q[2], -THUMB_Q[3]] : THUMB_Q };
}

// --- MJCF ---
const f6 = v => v.map(x => +x.toFixed(6)).join(' ');
const f12 = v => v.map(x => +x.toPrecision(12)).join(' ');   // quaternions: retargeting assumes the exact thumb frame
function handXML(side, pos, quat) {
  const S = sideSpec(side), g = n => `hand_${side}_${n}`;
  const abdAxis = `0 0 ${S.sA}`;
  const surf = 'contype="2" conaffinity="1" priority="1" solref="0.004 1" solimp="0.95 0.99 0.001"';
  const joint = (name, kind) => `<joint name="${side}_${name}" type="hinge" axis="${kind.endsWith('abd') ? abdAxis : '1 0 0'}" ` +
    `range="${KIND[kind][0].join(' ')}" damping="${KIND[kind][2]}" armature="${ARMATURE}"/>`;
  const capsule = (name, len, r, mass, tip) => `<geom name="${g(name)}" type="capsule" fromto="0 0 0 0 ${+(tip ? len - r : len).toFixed(6)} 0" ` +
    `size="${r}" mass="${mass}" condim="4" friction="1.2 0.01 0.0002" ${surf}/>`;
  const site = (k, p) => `<site name="${side}_j${k}" pos="${f6(p)}" size="0.003"/>`;
  const T = THUMB;
  const thumb = `<body name="${g('thumb1')}" pos="${f6(S.mx(T.base))}" quat="${f12(S.thumbQ)}" gravcomp="1">
      ${joint('thumb_cmc_abd', 'tcmc_abd')}${joint('thumb_cmc', 'tcmc')}${capsule('thumb1', T.len[0], T.rad[0], THUMB_MASS[0])}${site(1, [0, 0, 0])}
      <body name="${g('thumb2')}" pos="0 ${T.len[0]} 0" gravcomp="1">
        ${joint('thumb_mcp_abd', 'tmcp_abd')}${joint('thumb_mcp', 'tmcp')}${capsule('thumb2', T.len[1], T.rad[1], THUMB_MASS[1])}${site(2, [0, 0, 0])}
        <body name="${g('thumb3')}" pos="0 ${T.len[1]} 0" gravcomp="1">
          ${joint('thumb_ip', 'tip')}${capsule('thumb3', T.len[2], T.rad[2], THUMB_MASS[2], true)}${site(3, [0, 0, 0])}${site(4, [0, T.len[2], 0])}
        </body>
      </body>
    </body>`;
  const fingers = FINGERS.map(([f, knuckle, L, R], i) => {
    const k = 5 + 5 * i;
    return `${site(k, S.mx([0.6 * knuckle[0], 0.012, 0]))}
    <body name="${g(f + '1')}" pos="${f6(S.mx(knuckle))}" gravcomp="1">
      ${joint(f + '_abd', 'abd')}${joint(f + '_mcp', 'mcp')}${capsule(f + '1', L[0], R[0], SEG_MASS[0])}${site(k + 1, [0, 0, 0])}
      <body name="${g(f + '2')}" pos="0 ${L[0]} 0" gravcomp="1">
        ${joint(f + '_pip', 'pip')}${capsule(f + '2', L[1], R[1], SEG_MASS[1])}${site(k + 2, [0, 0, 0])}
        <body name="${g(f + '3')}" pos="0 ${L[1]} 0" gravcomp="1">
          ${joint(f + '_dip', 'dip')}${capsule(f + '3', L[2], R[2], SEG_MASS[2], true)}${site(k + 3, [0, 0, 0])}${site(k + 4, [0, L[2], 0])}
        </body>
      </body>
    </body>`;
  }).join('\n    ');
  return `<body name="hand_${side}" pos="${f6(pos)}" quat="${f12(quat)}" gravcomp="1"><freejoint name="${side}_root"/>
    <geom name="hand_${side}" type="box" pos="${f6(S.mx(PALM.center))}" size="${f6(PALM.half)}" mass="${PALM.mass}" condim="3" friction="1.0 0.005 0.0001" ${surf}/>
    ${site(0, [0, 0, 0])}
    ${thumb}
    ${fingers}
  </body>`;
}
const actuatorsXML = side => HAND_JOINTS.map(([n, k]) => {
  const [range, strength] = KIND[k];
  return `<position name="${side}_${n}" joint="${side}_${n}" kp="${+(strength / SATURATE).toFixed(4)}" ctrlrange="${range.join(' ')}" forcerange="${-strength} ${strength}"/>`;
}).join('\n    ');

// Hands parked palm down, fingers ahead, before tracking starts.
export const PARK = [[-0.25, 0.15, 1.0], [0.25, 0.15, 1.0]];
export const PALM_DOWN = [0, 0, 1, 0];            // 180 deg about y: palm +z down, fingers +y ahead

export function handSceneXML(p = HAND_PARAMS, { boxes, statics } = HAND_SCENE) {
  const buttons = BUTTONS.map(([n, [x, y]]) =>
    `<geom name="btn_${n}" type="cylinder" size="${BUTTON_R} ${BUTTON_H / 2}" pos="${x} ${y} ${TABLE_TOP + BUTTON_H / 2}" friction="1.0 0.005 0.0001"/>`).join('\n    ');
  return `<mujoco model="articulated_hands">
  <compiler angle="radian"/>
  <option timestep="${p.dt}" integrator="implicitfast" ${p.option}>${p.flags ? `<flag ${p.flags}/>` : ''}</option>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="6 6 0.1" friction="0.8 0.005 0.0001"/>
    ${statics.map(([n, s, x]) => `<geom name="${n}" type="box" pos="${x.join(' ')}" size="${s.join(' ')}"/>`).join('\n    ')}
    ${buttons}
    ${bodiesXML(boxes)}
    ${handXML('l', PARK[0], PALM_DOWN)}
    ${handXML('r', PARK[1], PALM_DOWN)}
  </worldbody>
  <actuator>
    ${actuatorsXML('l')}
    ${actuatorsXML('r')}
  </actuator>
</mujoco>`;
}

// --- retargeting: tracked joint positions -> palm pose + 21 joint angles ---
// Only positions are used, so no convention for the tracked joints' orientations is needed. A two-hinge joint
// (sideways about +-z, then bend about x) reproduces a bone direction d exactly; a one-hinge joint takes the bend in
// its plane. Near a 90 deg bend the sideways angle is ill-defined and is faded out.
function twoHinge(d, sA, range0, range1) {
  let f = Math.asin(clamp(d[2], -1, 1)), a = Math.atan2(-d[0], d[1]);
  if (Math.abs(a) > Math.PI / 2) { a -= Math.sign(a) * Math.PI; f = (d[2] >= 0 ? Math.PI : -Math.PI) - f; }
  a *= Math.min(1, Math.hypot(d[0], d[1]) / 0.35);
  return [clamp(sA * a, ...range0), clamp(f, ...range1)];
}
const bend = (b, range) => clamp(Math.atan2(b[2], b[1]), ...range);
const X = [1, 0, 0];

// pts: 25 positions (MuJoCo frame) in XR_JOINTS order. Returns the wrist target { pos, quat } and the angles q.
export function retarget(side, pts) {
  const S = sideSpec(side), Z = [0, 0, S.sA];
  const W = pts[0], y = norm(sub(pts[11], W));                       // wrist -> middle knuckle
  let z = cross(sub(pts[6], pts[21]), y);                            // (index - little knuckle) x y: out of the right palm
  z = norm(z.map(c => c * S.sA));
  z = norm(sub(z, y.map(c => c * dot(z, y))));
  const quat = quatFromAxes(cross(y, z), y, z);
  const qc = qconj(quat), L = pts.map(P => rot(qc, sub(P, W)));     // positions in the palm frame
  const q = new Array(21);
  const R = JOINT_RANGE;
  { // thumb: base (2), knuckle (2), end joint (1)
    let Q = S.thumbQ;
    const d0 = rot(qconj(Q), norm(sub(L[2], L[1])));
    [q[0], q[1]] = twoHinge(d0, S.sA, R[0], R[1]);
    Q = qmul(qmul(Q, qaxis(Z, q[0])), qaxis(X, q[1]));
    const d1 = rot(qconj(Q), norm(sub(L[3], L[2])));
    [q[2], q[3]] = twoHinge(d1, S.sA, R[2], R[3]);
    Q = qmul(qmul(Q, qaxis(Z, q[2])), qaxis(X, q[3]));
    q[4] = bend(rot(qconj(Q), sub(L[4], L[3])), R[4]);
  }
  for (let i = 0; i < 4; i++) {
    const k = 5 + 5 * i, j = 5 + 4 * i;
    const d = norm(sub(L[k + 2], L[k + 1]));
    [q[j], q[j + 1]] = twoHinge(d, S.sA, R[j], R[j + 1]);
    let Q = qmul(qaxis(Z, q[j]), qaxis(X, q[j + 1]));
    q[j + 2] = bend(rot(qconj(Q), sub(L[k + 3], L[k + 2])), R[j + 2]);
    Q = qmul(Q, qaxis(X, q[j + 2]));
    q[j + 3] = bend(rot(qconj(Q), sub(L[k + 4], L[k + 3])), R[j + 3]);
  }
  return { pos: W.slice(), quat, q };
}

// --- the simulation ---
export function createHandSim(mujoco, p = HAND_PARAMS, scene = HAND_SCENE) {
  const model = mujoco.MjModel.from_xml_string(handSceneXML(p, scene));
  const data = new mujoco.MjData(model);
  const scratch = new mujoco.MjData(model);       // forward kinematics of poses that are not the simulated one
  const id = (type, name) => mujoco.mj_name2id(model, type, name);
  const T = mujoco.mjtObj;
  const BODY = T.mjOBJ_BODY.value, JOINT = T.mjOBJ_JOINT.value, ACT = T.mjOBJ_ACTUATOR.value, SITE = T.mjOBJ_SITE.value,
        GEOM = T.mjOBJ_GEOM.value;
  const bodyMass = Array.from(model.body_mass), bodyParent = Array.from(model.body_parentid);
  const hands = ['l', 'r'].map(side => {
    const body = id(BODY, `hand_${side}`), j = model.body_jntadr[body];
    const joints = HAND_JOINTS.map(([n]) => id(JOINT, `${side}_${n}`));
    const subtree = [];                             // the palm and every finger body
    for (let b = 0; b < model.nbody; b++) { let a = b; while (a > 0 && a !== body) a = bodyParent[a]; if (a === body) subtree.push(b); }
    const geoms = new Set();
    for (let g = 0; g < model.ngeom; g++) if (subtree.includes(model.geom_bodyid[g])) geoms.add(g);
    return { side, name: `hand_${side}`, body, qadr: model.jnt_qposadr[j], vadr: model.jnt_dofadr[j],
             jq: joints.map(k => model.jnt_qposadr[k]), act: HAND_JOINTS.map(([n]) => id(ACT, `${side}_${n}`)),
             sites: XR_JOINTS.map((_, k) => id(SITE, `${side}_j${k}`)), bodies: subtree, geoms,
             mass: subtree.reduce((s, b) => s + bodyMass[b], 0),
             target: null, force: [0, 0, 0], torque: [0, 0, 0], saturated: false };
  });
  const bound = (shape, s) => (shape === 'sphere' ? [s[0], s[0], s[0]] : shape === 'cylinder' ? [s[0], s[0], s[1]] : s);
  const boxes = scene.boxes.map(([n, m, s, , shape = 'box']) => {
    const b = id(BODY, n);
    return { name: n, mass: m, half: bound(shape, s), size: s, shape, body: b, qadr: model.jnt_qposadr[model.body_jntadr[b]] };
  });
  const buttons = BUTTONS.map(([n, xy]) => ({ name: n, geom: id(GEOM, `btn_${n}`), xy }));

  let V = null;                                     // typed-array views into the WASM heap, refetched after a growth
  const views = () => {
    if (!V || V.xpos.length === 0) V = { qpos: data.qpos, qvel: data.qvel, xpos: data.xpos, xquat: data.xquat,
      xipos: data.xipos, xfrc: data.xfrc_applied, ctrl: data.ctrl, cvel: data.cvel, subtreeCom: data.subtree_com,
      siteXpos: data.site_xpos, actForce: data.actuator_force };
    return V;
  };
  const pose = b => { const v = views();
    return { pos: Array.from(v.xpos.subarray(3 * b, 3 * b + 3)), quat: Array.from(v.xquat.subarray(4 * b, 4 * b + 4)) }; };

  // the wrist's rotational inertia, at the relaxed pose
  mujoco.mj_forward(model, data);
  for (const h of hands) {
    const w = pose(h.body).pos, xi = views().xipos, I = model.body_inertia;
    h.inertia = h.bodies.reduce((s, b) => {
      const r2 = (xi[3 * b] - w[0]) ** 2 + (xi[3 * b + 1] - w[1]) ** 2 + (xi[3 * b + 2] - w[2]) ** 2;
      return s + bodyMass[b] * r2 * 2 / 3 + (I[3 * b] + I[3 * b + 1] + I[3 * b + 2]) / 3;
    }, 0);
    // The arm's damping is the palm free joint's dof damping, which implicitfast integrates implicitly: as an
    // explicit force on the 0.3 kg palm it would be unstable at a 2 ms step. Only the spring is clamped, so a moving
    // hand is a little weaker in the direction it moves and stronger against being dragged.
    const dd = model.dof_damping;
    dd.fill(2 * Math.sqrt(p.kLin * (h.mass + p.loadMass)), h.vadr, h.vadr + 3);
    dd.fill(2 * Math.sqrt(p.kRot * (h.inertia + p.loadInertia)), h.vadr + 3, h.vadr + 6);
  }
  const qpos0 = Float64Array.from(data.qpos);

  function setFingers(h, q) { views().ctrl.set(q, h.act[0]); }
  function placeHand(i, pos, quat, q = RELAXED) {   // teleport: tracking starts, or comes back far away
    const h = hands[i], { qpos, qvel } = views();
    qpos.set(pos, h.qadr); qpos.set(quat, h.qadr + 3);
    h.jq.forEach((a, k) => { qpos[a] = q[k]; });
    qvel.fill(0, h.vadr, h.vadr + 6 + 21);
    h.target = { pos: pos.slice(), quat: quat.slice(), q: q.slice() };
    setFingers(h, q);
    mujoco.mj_forward(model, data);
  }
  function setTarget(i, pos, quat, q) {
    const h = hands[i];
    h.target = { pos, quat, q };
    setFingers(h, q);
  }

  function arm(h) {                                 // the clamped spring, held at the wrist (damping: dof_damping)
    const { xfrc, xipos } = views();
    const { pos, quat } = pose(h.body), t0 = h.target;
    let f = [0, 1, 2].map(k => p.kLin * (t0.pos[k] - pos[k]));
    const e = qrotvec(qmul(t0.quat, qconj(quat)));
    let t = [0, 1, 2].map(k => p.kRot * e[k]);
    h.saturated = Math.hypot(...f) > p.fMax || Math.hypot(...t) > p.tMax;
    f = clampNorm(f, p.fMax); t = clampNorm(t, p.tMax);
    h.force = f; h.torque = t;
    // xfrc_applied acts at the palm body's centre of mass; move the force to the wrist
    const r = sub(pos, [xipos[3 * h.body], xipos[3 * h.body + 1], xipos[3 * h.body + 2]]), rf = cross(r, f);
    xfrc.set([...f, t[0] + rf[0], t[1] + rf[1], t[2] + rf[2]], 6 * h.body);
  }

  function step(n) {
    for (const [i, h] of hands.entries()) {       // a wrist that reappears far away moves the hand there
      if (h.target && Math.hypot(...sub(pose(h.body).pos, h.target.pos)) > p.teleport)
        placeHand(i, h.target.pos, h.target.quat, h.target.q);
    }
    for (let k = 0; k < n; k++) {
      for (const h of hands) if (h.target) arm(h);
      mujoco.mj_step(model, data);
    }
  }

  function reset() {
    mujoco.mj_resetData(model, data);
    views().qpos.set(qpos0);
    for (const h of hands) { h.target = null; setFingers(h, RELAXED); }
    mujoco.mj_forward(model, data);
  }

  // the 25 joint positions of a hand: simulated, or (fk) of a given wrist pose and angles
  function handPoints(i) {
    const s = views().siteXpos;
    return hands[i].sites.map(k => Array.from(s.subarray(3 * k, 3 * k + 3)));
  }
  function fkPoints(i, pos, quat, q) {
    const h = hands[i], qp = scratch.qpos;
    qp.set(pos, h.qadr); qp.set(quat, h.qadr + 3);
    h.jq.forEach((a, k) => { qp[a] = q[k]; });
    mujoco.mj_kinematics(model, scratch);
    const s = scratch.site_xpos;
    return h.sites.map(k => Array.from(s.subarray(3 * k, 3 * k + 3)));
  }
  // Angles that put chosen fingertips at given world points, the wrist pose held: damped least squares on the forward
  // kinematics, one finger at a time. targets: { thumb: [x, y, z], index: ..., ... }. For scripted grasps (tests, demo).
  function solveTips(i, pos, quat, targets, q0 = RELAXED, iters = 100) {
    const q = q0.slice(), R = JOINT_RANGE;
    for (const [f, goal] of Object.entries(targets)) {
      const js = FINGER_DOFS[f], tip = TIPS[f];
      for (let it = 0; it < iters; it++) {
        const p0 = fkPoints(i, pos, quat, q)[tip], e = sub(goal, p0);
        if (Math.hypot(...e) < 1e-5) break;
        const J = js.map(j => { const qq = q.slice(); qq[j] += 1e-5;
          return sub(fkPoints(i, pos, quat, qq)[tip], p0).map(c => c / 1e-5); });   // columns
        const A = [0, 1, 2].map(r => [0, 1, 2].map(c => J.reduce((s, col) => s + col[r] * col[c], 0) + (r === c ? 1e-4 : 0)));
        const y = solve3(A, e);
        js.forEach((j, k) => { q[j] = clamp(q[j] + dot(J[k], y), ...R[j]); });
      }
    }
    return q;
  }
  const jointAngles = i => { const qp = views().qpos; return hands[i].jq.map(a => qp[a]); };
  // servo torque as a fraction of each joint's strength, averaged over the hand: how hard the hand squeezes
  const strengths = HAND_JOINTS.map(([, k]) => KIND[k][1]);
  const grip = i => { const af = views().actForce, h = hands[i];
    return h.act.reduce((s, a, k) => s + Math.abs(af[a]) / strengths[k], 0) / h.act.length; };

  // --- contacts and velocities (for sound and the board) ---
  const geomBody = Array.from(model.geom_bodyid);
  const geomName = Array.from({ length: model.ngeom }, (_, g) => { const a = model.geom(g), n = a.name; a.delete(); return n; });
  const rootOf = Array.from(model.body_rootid);
  const cforce = new mujoco.DoubleBuffer(6);
  function bodyVelocity(b) {                       // world velocity of the body origin and angular velocity
    if (b === 0) return { lin: [0, 0, 0], ang: [0, 0, 0] };
    const { cvel, subtreeCom, xpos } = views(), r = rootOf[b];
    const ang = [cvel[6 * b], cvel[6 * b + 1], cvel[6 * b + 2]], l = [cvel[6 * b + 3], cvel[6 * b + 4], cvel[6 * b + 5]];
    const w = cross(ang, [xpos[3 * b] - subtreeCom[3 * r], xpos[3 * b + 1] - subtreeCom[3 * r + 1], xpos[3 * b + 2] - subtreeCom[3 * r + 2]]);
    return { lin: [l[0] + w[0], l[1] + w[1], l[2] + w[2]], ang };
  }
  function contacts() {
    const out = [], n = data.ncon;
    if (!n) return out;
    const cv = data.contact;
    for (let i = 0; i < n; i++) {
      const c = cv.get(i);
      if (!c) continue;
      const g1 = c.geom1, g2 = c.geom2, exclude = c.exclude;
      const pos = Array.from(c.pos), frame = Array.from(c.frame);
      c.delete();
      if (exclude) continue;
      mujoco.mj_contactForce(model, data, i, cforce);
      out.push({ g1, g2, b1: geomBody[g1], b2: geomBody[g2], pos, normal: frame.slice(0, 3), fn: Math.abs(cforce.GetView()[0]) });
    }
    cv.delete();
    return out;
  }

  const dispose = () => { cforce.delete(); scratch.delete(); data.delete(); model.delete(); };
  return { mujoco, model, data, params: p, hands, boxes, buttons, pose, views, placeHand, setTarget, step, reset,
           handPoints, fkPoints, solveTips, jointAngles, grip, retarget: (i, pts) => retarget(hands[i].side, pts),
           contacts, bodyVelocity, geomBody, geomName, dispose };
}
