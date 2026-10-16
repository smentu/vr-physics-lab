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
  ['table', [0.6, 0.3, 0.375], [0, 0.65, 0.375]],
];

// name, mass kg, size m, initial position (MuJoCo frame)[, shape]. Shape 'box' (default; size = half sizes),
// 'sphere' (radius), 'cylinder' (radius, half height; axis z) or 'mesh' (size = bounding half sizes; the mesh of
// the same name in scene.meshes, { vertices, faces }; it collides as its convex hull, with the hull's inertia).
// An optional sixth element { gravcomp, damping } makes a body float (gravity compensation, 0..1) and damps its
// free joint (N s/m, N m s/rad). A scene is { boxes, statics[, meshes] }; the lab's is the default, and other
// pages (sss_app.js) pass their own.
export const BOXES = [
  ['box1', 1, [0.08, 0.08, 0.08], [-0.35, 0.65, 0.83]],
  ['box3', 3, [0.10, 0.10, 0.10], [-0.05, 0.65, 0.85]],
  ['box5', 5, [0.12, 0.10, 0.10], [0.30, 0.65, 0.85]],
  ['box10', 10, [0.15, 0.12, 0.12], [-0.95, 0.30, 0.12]],
  ['box20', 20, [0.20, 0.15, 0.15], [1.00, 0.30, 0.15]],
  ['box30', 30, [0.25, 0.18, 0.18], [-1.15, 0.95, 0.18]],
];
export const HAND_HALF = [0.04, 0.045, 0.05];   // a fist-sized box in grip-space axes
const HAND_REACH = 0.045;                   // hand centre to its surface, for grasp distance

export const LAB_SCENE = { boxes: BOXES, statics: STATIC };

// The free objects of a scene as MJCF bodies. Round bodies get rolling friction (condim 6), or a sphere set on the
// table rolls away.
export function bodiesXML(BOXES) {
  return BOXES.map(([n, m, s, x, shape = 'box', o = {}]) =>
    `<body name="${n}" pos="${x.join(' ')}"${o.gravcomp ? ` gravcomp="${o.gravcomp}"` : ''}>` +
    (o.damping ? `<joint type="free" damping="${o.damping}"/>` : '<freejoint/>') +
    `<geom name="${n}" type="${shape}" ${shape === 'mesh' ? `mesh="${n}"` : `size="${s.join(' ')}"`} mass="${m}" ` +
    (shape === 'box' ? 'friction="0.8 0.005 0.0001"' : 'condim="6" friction="0.8 0.005 0.002"') + '/></body>').join('\n');
}

export function sceneXML(p = PARAMS, { boxes: BOXES, statics: STATIC, meshes = {} } = LAB_SCENE) {
  const boxes = bodiesXML(BOXES);
  const hands = ['hand_l', 'hand_r'].map((n, i) =>
    `<body name="${n}" pos="${i ? 0.2 : -0.2} 0.2 1.0"><freejoint/>` +
    `<geom name="${n}" type="box" size="${HAND_HALF.join(' ')}" mass="${p.handMass}" contype="2" conaffinity="1" friction="1.0 0.005 0.0001" priority="1" solref="0.004 1" solimp="0.95 0.99 0.001"/></body>`).join('\n');
  const welds = ['hand_l', 'hand_r'].flatMap(h => BOXES.map(([n]) =>
    `<weld name="${h}-${n}" body1="${h}" body2="${n}" active="false" solref="0.005 1"/>`)).join('\n');
  return `<mujoco model="spring_hands">
  <option timestep="${p.dt}" integrator="implicitfast"/>
  <asset>${Object.entries(meshes).map(([n, g]) =>
    `<mesh name="${n}" inertia="convex" vertex="${Array.from(g.vertices).join(' ')}" face="${Array.from(g.faces).join(' ')}"/>`).join('')}</asset>
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
export function qrotvec(q) {            // rotation vector of unit quaternion q (shortest arc)
  let [w, x, y, z] = q;
  if (w < 0) { w = -w; x = -x; y = -y; z = -z; }
  const s = Math.hypot(x, y, z);
  if (s < 1e-9) return [2 * x, 2 * y, 2 * z];
  const a = 2 * Math.atan2(s, w) / s;
  return [a * x, a * y, a * z];
}
export function rot(q, v) {             // rotate v by q
  const r = qmul(qmul(q, [0, v[0], v[1], v[2]]), qconj(q));
  return [r[1], r[2], r[3]];
}
export const clampNorm = (v, m) => { const n = Math.hypot(...v); return n > m ? v.map(c => c * m / n) : v; };

export function createSim(mujoco, p = PARAMS, scene = LAB_SCENE) {
  const { boxes: BOXES } = scene;
  const model = mujoco.MjModel.from_xml_string(sceneXML(p, scene));
  const data = new mujoco.MjData(model);
  const id = (type, name) => mujoco.mj_name2id(model, type, name);
  const BODY = mujoco.mjtObj.mjOBJ_BODY.value, EQ = mujoco.mjtObj.mjOBJ_EQUALITY.value;
  const hands = ['hand_l', 'hand_r'].map(n => {
    const b = id(BODY, n);
    const j = model.body_jntadr[b];
    return { name: n, body: b, qadr: model.jnt_qposadr[j], vadr: model.jnt_dofadr[j],
             target: null, held: null, force: [0, 0, 0], torque: [0, 0, 0], saturated: false };
  });
  // half: the bounding box's half sizes (grasp distance, held inertia); size: the geom size
  const bound = (shape, s) => (shape === 'sphere' ? [s[0], s[0], s[0]] : shape === 'cylinder' ? [s[0], s[0], s[1]] : s);
  const boxes = BOXES.map(([n, m, s, , shape = 'box']) => {
    const b = id(BODY, n);
    return { name: n, mass: m, half: bound(shape, s), size: s, shape, body: b, qadr: model.jnt_qposadr[model.body_jntadr[b]] };
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

  // --- contacts and velocities (for sound) ---
  const geomBody = Array.from(model.geom_bodyid);
  const geomName = Array.from({ length: model.ngeom }, (_, g) => { const a = model.geom(g), n = a.name; a.delete(); return n; });
  const bodyDof = Array.from({ length: model.nbody }, (_, b) =>
    model.body_jntadr[b] >= 0 ? model.jnt_dofadr[model.body_jntadr[b]] : -1);
  const cforce = new mujoco.DoubleBuffer(6);

  function bodyVelocity(b) {                    // world linear and angular velocity of a free body
    const adr = bodyDof[b];
    if (adr < 0) return { lin: [0, 0, 0], ang: [0, 0, 0] };
    const { qvel: v, xquat } = views();
    const q = Array.from(xquat.subarray(4 * b, 4 * b + 4));
    return { lin: [v[adr], v[adr + 1], v[adr + 2]], ang: rot(q, [v[adr + 3], v[adr + 4], v[adr + 5]]) };
  }

  function contacts() {                         // every active contact point: geoms, bodies, point, normal (geom1 -> geom2), normal force
    const out = [];
    const n = data.ncon;
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
      out.push({ g1, g2, b1: geomBody[g1], b2: geomBody[g2], pos, normal: frame.slice(0, 3),
                 fn: Math.abs(cforce.GetView()[0]) });
    }
    cv.delete();
    return out;
  }

  const dispose = () => { cforce.delete(); data.delete(); model.delete(); };

  return { mujoco, model, data, params: p, hands, boxes, pose, eqActive: () => eqActive.slice(), dispose,
           contacts, bodyVelocity, geomBody, geomName, placeHand, grasp, release, step, reset,
           setTarget: (hi, pos, quat) => { hands[hi].target = { pos, quat }; } };
}

// Contact events for sound, once per rendered frame. Points are merged per geom pair; units m/s, N, J;
// positions in the MuJoCo frame.
// - An impact is the start of a contact: the pair touches after being apart for at least `minGap` s,
//   or, within a sustained contact, its normal force jumps (above `spikeRatio` x its recent average
//   + `spikeMin` N) or a corner lands with more than `sustainedEnergyMin` J (a pushed box tipping
//   back onto its face). It is also required to close at more than `impactMin` m/s along the normal
//   (at the previous frame's velocities) and to carry more than `energyMin` J. Without the gap, the
//   spike rule and the energy floors, a hand wobbling and rolling while it slides keeps landing on
//   the surface it rubs, and every landing would thud.
// - A rub is a contact with tangential sliding.
export function createContactTracker(sim, { impactMin = 0.15, energyMin = 0.015, minGap = 0.06, spikeRatio = 3,
                                            spikeMin = 30, sustainedEnergyMin = 0.15, rubMin = 0.03,
                                            refractory = 0.08 } = {}) {
  let prevVel = new Map();
  const lastImpact = new Map();
  const state = new Map();                      // pair -> { touching, since (s), fnAvg (N) }
  const bodyMass = Array.from(sim.model.body_mass);
  const effMass = (b1, b2) => {                 // the world (and anything without mass) is immovable
    const m1 = bodyMass[b1] || Infinity, m2 = bodyMass[b2] || Infinity;
    return Number.isFinite(m1) && Number.isFinite(m2) ? m1 * m2 / (m1 + m2) : Math.min(m1, m2);
  };
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const pointVel = (vel, origin, p) => {
    const r = p.map((c, i) => c - origin[i]), w = cross(vel.ang, r);
    return vel.lin.map((c, i) => c + w[i]);
  };
  return function update(t) {
    const cs = sim.contacts();
    const bodies = new Set();
    cs.forEach(c => { bodies.add(c.b1); bodies.add(c.b2); });
    for (const h of sim.hands) bodies.add(h.body);
    for (const b of sim.boxes) bodies.add(b.body);
    const vel = new Map(), origin = new Map();
    for (const b of bodies) { vel.set(b, sim.bodyVelocity(b)); origin.set(b, sim.pose(b).pos); }
    const pairs = new Map();
    for (const c of cs) {
      const key = Math.min(c.g1, c.g2) * 4096 + Math.max(c.g1, c.g2);
      const o1 = origin.get(c.b1), o2 = origin.get(c.b2);
      const p1 = prevVel.get(c.b1) ?? vel.get(c.b1), p2 = prevVel.get(c.b2) ?? vel.get(c.b2);
      const dPrev = pointVel(p1, o1, c.pos).map((x, i) => x - pointVel(p2, o2, c.pos)[i]);
      const approach = dot(dPrev, c.normal);                 // > 0: closing along the normal
      const d = pointVel(vel.get(c.b1), o1, c.pos).map((x, i) => x - pointVel(vel.get(c.b2), o2, c.pos)[i]);
      const dn = dot(d, c.normal);
      const slide = Math.hypot(...d.map((x, i) => x - dn * c.normal[i]));
      let e = pairs.get(key);
      if (!e) pairs.set(key, e = { key, g1: c.g1, g2: c.g2, b1: c.b1, b2: c.b2, approach: 0, impactPos: c.pos,
                                    fn: 0, slideF: 0, posF: [0, 0, 0], n: 0 });
      if (approach > e.approach) { e.approach = approach; e.impactPos = c.pos; }
      e.fn += c.fn; e.slideF += slide * c.fn; e.n++;
      e.posF = e.posF.map((x, i) => x + c.pos[i] * c.fn);
    }
    const impacts = [], rubs = [];
    for (const e of pairs.values()) {
      const pos = e.fn > 1e-6 ? e.posF.map(x => x / e.fn) : e.impactPos;
      let st = state.get(e.key);
      const onset = !st || (!st.touching && t - st.since >= minGap);
      const energy = 0.5 * effMass(e.b1, e.b2) * e.approach * e.approach;
      const sustained = !!st && st.touching && (e.fn > spikeRatio * st.fnAvg + spikeMin || energy > sustainedEnergyMin);
      if ((onset || sustained) && e.approach > impactMin && energy > energyMin
          && t - (lastImpact.get(e.key) ?? -1) > refractory) {
        lastImpact.set(e.key, t);
        impacts.push({ key: e.key, g1: e.g1, g2: e.g2, b1: e.b1, b2: e.b2, pos: e.impactPos, speed: e.approach, energy });
      }
      if (!st) state.set(e.key, st = { touching: true, since: t, fnAvg: e.fn });
      else {
        if (!st.touching) { st.touching = true; st.since = t; }
        st.fnAvg = 0.8 * st.fnAvg + 0.2 * e.fn;
      }
      const slide = e.fn > 1e-6 ? e.slideF / e.fn : 0;
      if (slide > rubMin) rubs.push({ key: e.key, g1: e.g1, g2: e.g2, b1: e.b1, b2: e.b2, pos, slide, fn: e.fn });
    }
    for (const [key, st] of state) {             // pairs that came apart this frame; forget long-gone ones
      if (pairs.has(key)) continue;
      if (st.touching) { st.touching = false; st.since = t; }
      else if (t - st.since > 2) state.delete(key);
    }
    prevVel = vel;
    return { impacts, rubs };
  };
}
