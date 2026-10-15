// Spring-hand VR scene on MuJoCo WASM (npm @mujoco/mujoco 3.12.0, the offline MuJoCo version):
// shared by the headset page (app.js) and the Node test (test_sim_core.mjs).
// Design: analysis/2026-10-15_vr_teleop_locomanipulation_design.md §10.
// MuJoCo frame: Z up, +Y forward (away from the operator), metres, kg.
// Each proxy hand is a free body pulled to its controller pose by a clamped linear + torsional
// spring-damper (the arm); a grasp is a weld between the hand and a box, switched at run time.

export const PARAMS = {
  dt: 0.001,            // physics step, s
  kLin: 3000,           // N/m
  fMax: 200,            // N per hand (arm strength)
  kRot: 30,             // N m/rad
  tMax: 15,             // N m per hand (wrist strength)
  handMass: 0.5,        // kg
  graspReach: 0.03,     // m beyond the hand's surface within which a squeeze grasps
  autoRelease: 0.45,    // m controller-to-hand gap at which a grasp lets go
};

export const STATIC = [  // name, half size m, position (MuJoCo frame); the floor is the plane z = 0
  ['wall', [2, 0.05, 1.2], [0, 1.6, 1.2]],
  ['table', [0.6, 0.3, 0.375], [0, 0.65, 0.375]],
];

export const BOXES = [   // name, mass kg, half size m, initial position (MuJoCo frame)
  ['box1', 1, [0.08, 0.08, 0.08], [-0.35, 0.65, 0.83]],
  ['box3', 3, [0.10, 0.10, 0.10], [-0.05, 0.65, 0.85]],
  ['box5', 5, [0.12, 0.10, 0.10], [0.30, 0.65, 0.85]],
  ['box10', 10, [0.15, 0.12, 0.12], [-0.95, 0.30, 0.12]],
  ['box20', 20, [0.20, 0.15, 0.15], [1.00, 0.30, 0.15]],
  ['box30', 30, [0.25, 0.18, 0.18], [0.00, 1.25, 0.18]],
];
export const HAND_HALF = [0.04, 0.045, 0.05];   // a fist-sized box in grip-space axes
const HAND_REACH = 0.045;                   // hand centre to its surface, for grasp distance

export function sceneXML(p = PARAMS) {
  const boxes = BOXES.map(([n, m, s, x]) =>
    `<body name="${n}" pos="${x.join(' ')}"><freejoint/>` +
    `<geom name="${n}" type="box" size="${s.join(' ')}" mass="${m}" friction="0.8 0.005 0.0001"/></body>`).join('\n');
  const hands = ['hand_l', 'hand_r'].map((n, i) =>
    `<body name="${n}" pos="${i ? 0.2 : -0.2} 0.2 1.0"><freejoint/>` +
    `<geom name="${n}" type="box" size="${HAND_HALF.join(' ')}" mass="${p.handMass}" contype="2" conaffinity="1" friction="1.0 0.005 0.0001" priority="1" solref="0.004 1" solimp="0.95 0.99 0.001"/></body>`).join('\n');
  const welds = ['hand_l', 'hand_r'].flatMap(h => BOXES.map(([n]) =>
    `<weld name="${h}-${n}" body1="${h}" body2="${n}" active="false" solref="0.005 1"/>`)).join('\n');
  return `<mujoco model="spring_hands">
  <option timestep="${p.dt}" integrator="implicitfast"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="6 6 0.1" friction="0.8 0.005 0.0001"/>
    ${STATIC.map(([n, s, x]) => `<geom name="${n}" type="box" pos="${x.join(' ')}" size="${s.join(' ')}"/>`).join('\n    ')}
    ${boxes}
    ${hands}
  </worldbody>
  <equality>
    ${welds}
  </equality>
</mujoco>`;
}

// --- quaternions, wxyz ---
export const qmul = (a, b) => [
  a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
  a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
  a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
  a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]];
export const qconj = q => [q[0], -q[1], -q[2], -q[3]];
function qrotvec(q) {            // rotation vector of unit quaternion q (shortest arc)
  let [w, x, y, z] = q;
  if (w < 0) { w = -w; x = -x; y = -y; z = -z; }
  const s = Math.hypot(x, y, z);
  if (s < 1e-9) return [2 * x, 2 * y, 2 * z];
  const a = 2 * Math.atan2(s, w) / s;
  return [a * x, a * y, a * z];
}
function rot(q, v) {             // rotate v by q
  const r = qmul(qmul(q, [0, v[0], v[1], v[2]]), qconj(q));
  return [r[1], r[2], r[3]];
}
const clampNorm = (v, m) => { const n = Math.hypot(...v); return n > m ? v.map(c => c * m / n) : v; };

export function createSim(mujoco, p = PARAMS) {
  const model = mujoco.MjModel.from_xml_string(sceneXML(p));
  const data = new mujoco.MjData(model);
  const id = (type, name) => mujoco.mj_name2id(model, type, name);
  const BODY = mujoco.mjtObj.mjOBJ_BODY.value, EQ = mujoco.mjtObj.mjOBJ_EQUALITY.value;
  const hands = ['hand_l', 'hand_r'].map(n => {
    const b = id(BODY, n);
    const j = model.body_jntadr[b];
    return { name: n, body: b, qadr: model.jnt_qposadr[j], vadr: model.jnt_dofadr[j],
             target: null, held: null, force: [0, 0, 0], torque: [0, 0, 0], saturated: false };
  });
  const boxes = BOXES.map(([n, m, s]) => {
    const b = id(BODY, n);
    return { name: n, mass: m, half: s, body: b, qadr: model.jnt_qposadr[model.body_jntadr[b]] };
  });
  const eqId = {};
  for (const h of hands) for (const bx of boxes) eqId[`${h.name}-${bx.name}`] = id(EQ, `${h.name}-${bx.name}`);
  mujoco.mj_forward(model, data);
  const qpos0 = Float64Array.from(data.qpos);
  // data.eq_active (a bool array) has no WASM binding in 3.12.0; equalities are switched through
  // mj_setState with the EQ_ACTIVE component instead, from this mirror
  const EQ_STATE = mujoco.mjtState.mjSTATE_EQ_ACTIVE.value;
  const eqActive = new Array(model.neq).fill(0);
  const setEq = (e, on) => { eqActive[e] = on ? 1 : 0; mujoco.mj_setState(model, data, eqActive, EQ_STATE); };

  // typed-array views into the WASM heap, cached; a heap growth detaches them (length 0), then refetch
  let V = null;
  const views = () => {
    if (!V || V.xpos.length === 0) V = { qpos: data.qpos, qvel: data.qvel, xpos: data.xpos, xquat: data.xquat,
                                         xfrc: data.xfrc_applied };
    return V;
  };
  const pose = b => { const v = views();
    return { pos: Array.from(v.xpos.subarray(3 * b, 3 * b + 3)), quat: Array.from(v.xquat.subarray(4 * b, 4 * b + 4)) }; };
  const holders = bx => hands.filter(h => h.held === bx).length;

  function placeHand(h, pos, quat) {           // teleport (tracking start / reset)
    const { qpos: q, qvel: v } = views();
    q.set(pos, h.qadr); q.set(quat, h.qadr + 3);
    v.fill(0, h.vadr, h.vadr + 6);
    h.target = { pos: pos.slice(), quat: quat.slice() };
  }

  function springs() {
    const { xfrc: xf, qvel: v } = views();
    const g = -model.opt.gravity[2];
    for (const h of hands) {
      if (!h.target) continue;
      const { pos, quat } = pose(h.body);
      const lin = [v[h.vadr], v[h.vadr + 1], v[h.vadr + 2]];
      const angLocal = [v[h.vadr + 3], v[h.vadr + 4], v[h.vadr + 5]];
      const ang = rot(quat, angLocal);
      let mEff = p.handMass, iEff = 2e-3;
      if (h.held) {
        const share = 1 / holders(h.held);
        const r = pose(h.held.body).pos.map((c, i) => c - pos[i]);
        mEff += h.held.mass * share;
        iEff += h.held.mass * share * (r[0] ** 2 + r[1] ** 2 + r[2] ** 2 + h.held.half.reduce((a, s) => a + s * s, 0) / 3);
      }
      const cLin = 2 * Math.sqrt(p.kLin * mEff), cRot = 2 * Math.sqrt(p.kRot * iEff);
      let f = [0, 1, 2].map(i => p.kLin * (h.target.pos[i] - pos[i]) - cLin * lin[i]);
      f[2] += p.handMass * g;                       // the arm carries the hand itself
      const e = qrotvec(qmul(h.target.quat, qconj(quat)));
      let t = [0, 1, 2].map(i => p.kRot * e[i] - cRot * ang[i]);
      h.saturated = Math.hypot(...f) > p.fMax || Math.hypot(...t) > p.tMax;
      f = clampNorm(f, p.fMax); t = clampNorm(t, p.tMax);
      h.force = f; h.torque = t;
      xf.set([...f, ...t], 6 * h.body);
    }
  }

  function boxDistance(h, bx) {                  // hand centre to box surface, minus the hand's half-height
    const hp = pose(h.body).pos, b = pose(bx.body);
    const local = rot(qconj(b.quat), hp.map((c, i) => c - b.pos[i]));
    const out = local.map((c, i) => Math.max(Math.abs(c) - bx.half[i], 0));
    return Math.hypot(...out) - HAND_REACH;
  }

  function grasp(hi) {
    const h = hands[hi];
    if (h.held) return null;
    let best = null, bd = p.graspReach;
    for (const bx of boxes) { const d = boxDistance(h, bx); if (d < bd) { bd = d; best = bx; } }
    if (!best) return null;
    // weld relpose: pose of body2 (box) in body1's (hand) frame
    const a = pose(h.body), b = pose(best.body);
    const relPos = rot(qconj(a.quat), b.pos.map((c, i) => c - a.pos[i]));
    const relQuat = qmul(qconj(a.quat), b.quat);
    const e = eqId[`${h.name}-${best.name}`];
    const ed = model.eq_data;
    ed.set([0, 0, 0, ...relPos, ...relQuat, 1], e * 11);
    setEq(e, true);
    h.held = best;
    return best.name;
  }

  function release(hi) {
    const h = hands[hi];
    if (!h.held) return;
    setEq(eqId[`${h.name}-${h.held.name}`], false);
    h.held = null;
  }

  function step(n) {
    for (let k = 0; k < n; k++) {
      springs();
      mujoco.mj_step(model, data);
    }
    for (const [i, h] of hands.entries()) {      // a grasp that has been dragged far away lets go
      if (h.held && h.target) {
        const gap = Math.hypot(...pose(h.body).pos.map((c, j) => c - h.target.pos[j]));
        if (gap > p.autoRelease) release(i);
      }
    }
  }

  function reset() {
    for (let i = 0; i < hands.length; i++) release(i);
    mujoco.mj_resetData(model, data);
    eqActive.fill(0);
    views().qpos.set(qpos0);
    for (const h of hands) h.target = null;
    mujoco.mj_forward(model, data);
  }

  const dispose = () => { data.delete(); model.delete(); };

  return { mujoco, model, data, params: p, hands, boxes, pose, eqActive: () => eqActive.slice(), dispose, placeHand, grasp, release, step, reset,
           setTarget: (hi, pos, quat) => { hands[hi].target = { pos, quat }; } };
}
