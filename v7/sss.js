// Cheap subsurface scattering for simple shapes (sphere, box, cylinder), for a mobile VR GPU.
// One forward pass, no extra render targets: every technique is per-pixel ALU plus at most three
// fetches per light from a small lookup table (and, on a mesh, a distance-field texture). The shape is known analytically in object space, so the
// distance through the object toward a light, and the surface normal where that light enters, are
// exact ray-shape intersections instead of a baked thickness map.
//
// Techniques (SSS_MODES), cheapest first; each is a define, so a mode is its own shader program:
//   0 lambert      opaque diffuse with the medium's multiple-scattering albedo (the baseline)
//   1 wrap         wrap lighting per channel (Green 2004, GPU Gems 1 ch. 16)
//   2 fast         1 + "fast translucency" (Barre-Brisebois & Bouchard, GDC 2011), with the
//                  analytic thickness along -N in place of the baked local-thickness map
//   3 diffusion    pre-integrated diffusion on curved surfaces (Penner 2011) with the Burley
//                  normalized-diffusion profile (Christensen & Burley 2015), plus transmission
//                  through the analytic thickness toward each light (the Burley transmittance that
//                  Unity HDRP uses), lit by the irradiance at the analytic entry point
//   4 diffusion+ss 3 + single scattering along the view ray, integrated with SS_STEPS
//                  importance-spaced samples, each with an analytic shadow ray to the surface
//   5 lit surface  the Burley profile integrated over the whole lit surface at straight-line (through
//                  the body) distance, the idea of translucent shadow maps (Dachsbacher & Stamminger
//                  2003) with the lit surface known analytically: a box sums its lit faces as planes
//                  (closed form); a sphere and a cylinder's side read precomputed tables of the
//                  integral over a sphere / an infinite cylinder. Replaces both terms of mode 3.
//   6 lit surf.+ss 5 + the single scattering of mode 4
// Meshes (shape type 'mesh', uShape 3): the thickness queries sphere-trace a signed distance field in a 3D
// texture (SDF_STEPS steps), normals come from its gradient, curvature from screen-space derivatives; modes 5
// and 6 fall back to the diffusion path of modes 3 and 4 there. The bunny's field is built offline (bunny.js).
// Checked against volumetric path tracing: analysis/2026-10-15_cheap_subsurface_scattering.md.
// Lights: three.js directional, point and hemisphere lights (ShaderMaterial with lights: true), in the
// same units as MeshLambertMaterial, so these objects sit in a Lambert scene at the same brightness.
// Media are homogeneous with a matched boundary (index 1): no refraction, no internal reflection; an
// optional glossy coat (uSpec) is added on top in every mode.
import * as THREE from 'three';

export const SSS_MODES = ['lambert', 'wrap', 'fast translucency', 'diffusion', 'diffusion + single scattering',
  'lit surface', 'lit surface + single scattering'];

// Media in 1/mm: absorption, scattering, Henyey-Greenstein g. 'skin' is Jensen et al. 2001 skin1
// (reduced scattering, so g = 0). The others are chosen to look like their names at 5-10 cm.
export const MEDIA = {
  wax: { sigmaA: [0.0015, 0.0035, 0.012], sigmaS: [0.75, 0.80, 0.85], g: 0.0 },
  jade: { sigmaA: [0.012, 0.0025, 0.010], sigmaS: [0.15, 0.18, 0.15], g: 0.0 },
  gummy: { sigmaA: [0.003, 0.060, 0.050], sigmaS: [0.05, 0.05, 0.05], g: 0.0 },
  skin: { sigmaA: [0.032, 0.17, 0.48], sigmaS: [0.74, 0.88, 1.01], g: 0.0 },
};

// Derived per-channel quantities, SI units (1/m, m). Diffusion theory with a matched boundary:
// sigma_tr = sqrt(3 sa st'), total diffuse reflectance Rd(alpha') (Jensen 2001, A = 1), Burley shape
// d = l_d / s with l_d = 1 / sigma_tr. The default s = 3.5 + 100 (A - 0.33)^4 is the Christensen & Burley
// (2015) fit as applied here; `burleyS` overrides it (the bench sweeps it against path-traced references).
export function mediumParams(m, { burleyS } = {}) {
  const ch = [0, 1, 2].map(c => {
    const sa = m.sigmaA[c] * 1000, ss = m.sigmaS[c] * 1000, ssr = ss * (1 - m.g), str = sa + ssr;
    const ap = ssr / str, k = Math.sqrt(3 * (1 - ap));
    const A = 0.5 * ap * (1 + Math.exp(-4 / 3 * k)) * Math.exp(-k);
    const sigmaTr = Math.sqrt(3 * sa * str);
    const s = burleyS ?? 3.5 + 100 * (A - 0.33) ** 4;
    return { A, sigmaTr, d: 1 / sigmaTr / s, sigmaS: ss, sigmaT: sa + ss };
  });
  const v = key => ch.map(c => c[key]);
  return { A: v('A'), sigmaTr: v('sigmaTr'), d: v('d'), sigmaS: v('sigmaS'), sigmaT: v('sigmaT'), g: m.g };
}

// Pre-integrated diffusion table D(cos theta, y), y = u / (1 + u), u = d * curvature: the Burley profile
// integrated over a sphere of radius r = d / u, illuminated by cos+ (Penner 2011), normalized so the
// flat limit is cos+ theta. Polar integration around the shading point (rho R(rho) has no singularity):
// D = int (e^-q + e^-q/3) int_0^pi cos+(theta + q u cos phi) dphi dq / (pi int (e^-q + e^-q/3) dq), q = rho / d,
// truncated at the antipode (q u = pi) and at q = 30. One channel; each colour channel looks it up
// with its own u.
export const LUT_SIZE = [64, 32];
export function preintegratedTable([W, H] = LUT_SIZE, nq = 48, nphi = 12) {   // ~50 ms; max error 0.0014
  const out = new Float32Array(W * H);
  for (let j = 0; j < H; j++) {
    const y = j / (H - 1), u = y < 1 ? y / (1 - y) : 1e6;
    const qmax = Math.min(30, u > 0 ? Math.PI / u : 30), dq = qmax / nq;
    for (let i = 0; i < W; i++) {
      const theta = Math.acos(-1 + 2 * i / (W - 1));
      let num = 0, den = 0;
      for (let a = 0; a < nq; a++) {
        const q = (a + 0.5) * dq, w = Math.exp(-q) + Math.exp(-q / 3);
        let ring = 0;
        for (let b = 0; b < nphi; b++) ring += Math.max(0, Math.cos(theta + q * u * Math.cos((b + 0.5) * Math.PI / nphi)));
        num += w * ring / nphi; den += w;
      }
      out[j * W + i] = num / den;
    }
  }
  return out;
}

// Modified Bessel K0 (Abramowitz & Stegun 9.8.5, 9.8.6; |error| < 1e-7)
export function besselK0(x) {
  if (x <= 2) {
    const t = (x / 3.75) ** 2, u = (x / 2) ** 2;
    const i0 = 1 + t * (3.5156229 + t * (3.0899424 + t * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))));
    return -Math.log(x / 2) * i0 + (-0.57721566 + u * (0.42278420 + u * (0.23069756 + u * (0.03488590 + u * (0.00262698 + u * (0.00010750 + u * 0.0000074))))));
  }
  const t = 2 / x;
  return Math.exp(-x) / Math.sqrt(x) * (1.25331414 + t * (-0.07832358 + t * (0.02189568 + t * (-0.01062446 + t * (0.00587872 + t * (-0.00251540 + t * 0.00053208))))));
}

// Lit-surface tables (mode 5), indexed like the pre-integrated one: x = cos of the angle between the shading
// normal and the light (sphere) or the light's component across the axis (cylinder), y = u / (1 + u),
// u = d / radius. Value: (1 / E) times the irradiance-weighted Burley profile integrated over the lit
// surface, profile normalized to 1 on a plane (A applied in the shader); flat limit cos+.
// Sphere, radius 1, chord r = 2 sin(alpha / 2) to the source point; sin(alpha) dalpha / r = dr, so
// G = int_0^2 dr (e^-r/u + e^-r/3u) / (8 pi u) int_0^2pi dpsi cos+(theta_Q), r spaced by the slow exponential.
export function sphereTable([W, H] = LUT_SIZE, nr = 64, npsi = 24) {
  const out = new Float32Array(W * H);
  for (let j = 0; j < H; j++) {
    const y = j / (H - 1), u = y < 1 ? Math.max(y / (1 - y), 1e-4) : 1e4;
    const k = 1 - Math.exp(-2 / (3 * u));
    for (let i = 0; i < W; i++) {
      const ct = -1 + 2 * i / (W - 1), st = Math.sqrt(Math.max(0, 1 - ct * ct));
      let g = 0;
      for (let a = 0; a < nr; a++) {
        const w = (a + 0.5) / nr, r = -3 * u * Math.log(1 - w * k), dr = 3 * u * k / (1 - w * k) / nr;
        const ca = 1 - r * r / 2, sa = Math.sqrt(Math.max(0, 1 - ca * ca));
        let ring = 0;
        for (let b = 0; b < npsi; b++) ring += Math.max(0, ct * ca - st * sa * Math.cos((b + 0.5) * Math.PI / npsi));
        g += dr * (Math.exp(-r / u) + Math.exp(-r / (3 * u))) / (8 * Math.PI * u) * 2 * Math.PI * ring / npsi;
      }
      out[j * W + i] = g;
    }
  }
  return out;
}
// Infinite cylinder, radius 1: the profile integrated along the axis is 2 (K0(c/u) + K0(c/3u)) / (8 pi u) at
// chord c = 2 sin(phi / 2); G = int_-pi^pi dphi cos+(phi - beta) that, with phi = pi t^3 against the log
// singularity at phi = 0.
export function cylinderTable([W, H] = LUT_SIZE, nphi = 96) {
  const out = new Float32Array(W * H);
  for (let j = 0; j < H; j++) {
    const y = j / (H - 1), u = y < 1 ? Math.max(y / (1 - y), 1e-4) : 1e4;
    for (let i = 0; i < W; i++) {
      const beta = Math.acos(-1 + 2 * i / (W - 1));
      let g = 0;
      for (let a = 0; a < nphi; a++) {
        const t = (a + 0.5) / nphi, phi = Math.PI * t ** 3, dphi = 3 * Math.PI * t * t / nphi;
        const c = 2 * Math.sin(phi / 2);
        const line = 2 * (besselK0(c / u) + besselK0(c / (3 * u))) / (8 * Math.PI * u);
        g += dphi * line * (Math.max(0, Math.cos(phi - beta)) + Math.max(0, Math.cos(-phi - beta)));
      }
      out[j * W + i] = g;
    }
  }
  return out;
}

function tableTexture(t) {
  const [W, H] = LUT_SIZE, half = new Uint16Array(t.length);
  for (let i = 0; i < t.length; i++) half[i] = THREE.DataUtils.toHalfFloat(t[i]);
  const tex = new THREE.DataTexture(half, W, H, THREE.RedFormat, THREE.HalfFloatType);
  tex.minFilter = tex.magFilter = THREE.LinearFilter;
  tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.needsUpdate = true;
  return tex;
}
const textures = {};
export const preintegratedTexture = () => (textures.pre ??= tableTexture(preintegratedTable()));
export const sphereTexture = () => (textures.sphere ??= tableTexture(sphereTable()));
export const cylinderTexture = () => (textures.cyl ??= tableTexture(cylinderTable()));

const VERT = /* glsl */`
attribute float curvature;          // meshes only (per vertex, sssGeometry); 0 for the analytic shapes
varying float vCurv;
varying vec3 vViewPos;
varying vec3 vNormal;
varying vec3 vObjPos;
void main() {
  vCurv = curvature;
  vObjPos = position;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vViewPos = mv.xyz;
  vNormal = normalMatrix * normal;
  gl_Position = projectionMatrix * mv;
}`;

const FRAG = /* glsl */`
#include <common>
#include <lights_pars_begin>
uniform mat4 modelViewMatrix;       // set per draw by three.js; rigid (no scale) for these objects
uniform vec3 uAlbedo;               // total diffuse albedo A per channel (linear)
uniform vec3 uAlbedoMS;             // A without its single-scattering share (mode 4 adds that share explicitly)
uniform vec3 uSigmaTr;              // diffusion extinction, 1/m
uniform vec3 uBurleyS;              // 1 / d, Burley shape, 1/m
uniform vec3 uBurleyD;              // d, m
uniform vec3 uSigmaS;               // scattering, 1/m
uniform vec3 uSigmaT;               // extinction, 1/m
uniform float uG;                   // Henyey-Greenstein g
uniform vec3 uWrap;                 // wrap per channel (mode 1, 2)
uniform int uShape;                 // 0 sphere (radius uHalf.x), 1 box (half sizes), 2 cylinder (radius x, half height z, axis z)
uniform vec3 uHalf;
uniform float uSpec;                // glossy coat strength, 0 = none
uniform sampler2D uPreLUT;
uniform sampler2D uLitLUT;          // mode 5: sphere or cylinder lit-surface table
uniform highp sampler3D uSDF;       // mesh: signed distance field over the box uSdfMin + [0, uSdfSize]
uniform vec3 uSdfMin;
uniform vec3 uSdfSize;
uniform float uSdfDmax;
varying vec3 vViewPos;
varying vec3 vNormal;
varying vec3 vObjPos;
varying float vCurv;

// --- analytic shape, object space ---
// mesh (uShape 3): signed distance field in a 3D texture, 8-bit, negative inside
float sdfAt(vec3 p) {
  vec3 uvw = (p - uSdfMin) / uSdfSize;
  if (any(lessThan(uvw, vec3(0.0))) || any(greaterThan(uvw, vec3(1.0)))) return uSdfDmax;
  return (texture(uSDF, uvw).r - 0.5) * 2.0 * uSdfDmax;
}
float exitDist(vec3 o, vec3 d) {    // distance from o (inside or on the surface) along unit d to the boundary
  if (uShape == 3) {                // sphere tracing; stops at the first exit
    // The field's zero surface and the mesh differ by up to a few mm, so a shading point can sit just outside
    // the field's interior: step in small steps until inside (at most SDF_ENTER), and only then look for the
    // exit. Without this the ray "exits" at once and the point gets no transmitted light (dark blotches).
    float t = SDF_MIN_STEP;
    bool inside = false;
    for (int k = 0; k < SDF_STEPS; k++) {
      float sd = sdfAt(o + d * t);
      if (sd < 0.0) { inside = true; t += max(-sd, SDF_MIN_STEP); }
      else if (inside || t >= SDF_ENTER) break;
      else t += SDF_MIN_STEP;
    }
    return inside ? t : 0.0;
  }
  if (uShape == 0) {
    float b = dot(o, d), c = dot(o, o) - uHalf.x * uHalf.x;
    return max(-b + sqrt(max(b * b - c, 0.0)), 0.0);
  } else if (uShape == 1) {
    vec3 inv = 1.0 / (abs(d) + 1e-6);
    vec3 t = (uHalf - o * sign(d)) * inv;
    return max(min(t.x, min(t.y, t.z)), 0.0);
  }
  float a = dot(d.xy, d.xy), b = dot(o.xy, d.xy), c = dot(o.xy, o.xy) - uHalf.x * uHalf.x;
  float tr = a > 1e-8 ? (-b + sqrt(max(b * b - a * c, 0.0))) / a : 1e9;
  float tz = (uHalf.z - o.z * sign(d.z)) / (abs(d.z) + 1e-6);
  return max(min(tr, tz), 0.0);
}
vec3 normalAt(vec3 q) {             // outward normal at a boundary point
  if (uShape == 3) {
    vec2 e = vec2(SDF_MIN_STEP, 0.0);
    return normalize(vec3(sdfAt(q + e.xyy) - sdfAt(q - e.xyy), sdfAt(q + e.yxy) - sdfAt(q - e.yxy), sdfAt(q + e.yyx) - sdfAt(q - e.yyx)));
  }
  if (uShape == 0) return normalize(q);
  if (uShape == 1) {
    vec3 r = abs(q) / uHalf;
    return r.x > r.y && r.x > r.z ? vec3(sign(q.x), 0, 0) : r.y > r.z ? vec3(0, sign(q.y), 0) : vec3(0, 0, sign(q.z));
  }
  return abs(q.z) / uHalf.z > length(q.xy) / uHalf.x ? vec3(0, 0, sign(q.z)) : vec3(normalize(q.xy), 0.0);
}
float curvature(vec3 p) {           // 1 / radius in the direction of strongest bending (0 on flat faces)
  if (uShape == 0) return 1.0 / uHalf.x;
  if (uShape == 2) return abs(p.z) < uHalf.z - 1e-3 ? 1.0 / uHalf.x : 0.0;
  return 0.0;
}

// view space to object space; transpose(R) v == v * R in GLSL
vec3 toObjDir(vec3 v) { return v * mat3(modelViewMatrix); }
vec3 toObjPos(vec3 p) { return (p - modelViewMatrix[3].xyz) * mat3(modelViewMatrix); }

float lut(float nl, float u) {
  vec2 sz = vec2(${LUT_SIZE[0]}.0, ${LUT_SIZE[1]}.0);
  vec2 uv = (vec2(nl * 0.5 + 0.5, u / (1.0 + u)) * (sz - 1.0) + 0.5) / sz;
  return texture2D(uPreLUT, uv).r;
}
float litLut(float x, float u) {
  vec2 sz = vec2(${LUT_SIZE[0]}.0, ${LUT_SIZE[1]}.0);
  vec2 uv = (vec2(x * 0.5 + 0.5, u / (1.0 + u)) * (sz - 1.0) + 0.5) / sz;
  return texture2D(uLitLUT, uv).r;
}
vec3 planeT(float z) {              // Burley profile integrated over a plane at distance z (normalized, no A)
  vec3 e = exp(-max(z, 0.0) * uBurleyS / 3.0);
  return 0.25 * (e * e * e + 3.0 * e);
}
// mode 5: (1 / E) x the profile integrated over the lit surface, per channel; po, lo object space
vec3 litSurface(vec3 po, vec3 lo) {
  if (uShape == 0) {
    vec3 u = uBurleyD / uHalf.x;
    float ct = dot(po, lo) / uHalf.x;
    return vec3(litLut(ct, u.r), litLut(ct, u.g), litLut(ct, u.b));
  }
  vec3 acc = vec3(0.0);
  if (uShape == 1) {                // at most three lit faces; the one on the light's side of each axis
    for (int i = 0; i < 3; i++) acc += abs(lo[i]) * planeT(uHalf[i] - sign(lo[i]) * po[i]);
    return acc;
  }
  float lp = length(lo.xy), rho = length(po.xy);
  if (abs(po.z) < uHalf.z - 1e-4 && lp > 1e-4) {      // on the side: the infinite-cylinder table
    vec3 u = uBurleyD / uHalf.x;
    float cb = dot(po.xy / rho, lo.xy / lp);
    acc += lp * vec3(litLut(cb, u.r), litLut(cb, u.g), litLut(cb, u.b));
  } else if (lp > 1e-4) {           // on a cap: the side's value at the nearest rim point, attenuated over the gap
    vec3 u = uBurleyD / uHalf.x;
    float cb = rho > 1e-5 ? dot(po.xy / rho, lo.xy / lp) : 0.0;
    acc += lp * vec3(litLut(cb, u.r), litLut(cb, u.g), litLut(cb, u.b)) * planeT(uHalf.x - rho);
  }
  acc += abs(lo.z) * planeT(uHalf.z - sign(lo.z) * po.z);   // the lit cap
  return acc;
}
float hg(float c) {                 // Henyey-Greenstein phase, per steradian
  float g2 = uG * uG;
  return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * uG * c, 1e-4), 1.5));
}

// radiance from one light, without single scattering. L, N, V view space; po, no, lo object space.
// modes 3 and 4 (and 5, 6 on a mesh): pre-integrated diffusion plus transmission through the thickness
vec3 diffusion(vec3 alb, float nl, vec3 E, vec3 po, vec3 lo, float kappa) {
  vec3 u = uBurleyD * kappa;
  vec3 c = alb * RECIPROCAL_PI * E * vec3(lut(nl, u.r), lut(nl, u.g), lut(nl, u.b));
  if (nl < 0.0) {                   // light that enters on the far side and diffuses through
    float t = exitDist(po, lo);
    vec3 q = po + lo * t;
    float eq = max(dot(normalAt(q), lo), 0.0);
    vec3 tr = uAlbedo * 0.25 * (exp(-uBurleyS * t) + 3.0 * exp(-uBurleyS * t / 3.0));
    c += tr * RECIPROCAL_PI * E * eq;
  }
  return c;
}
vec3 sssLight(vec3 L, vec3 E, vec3 N, vec3 V, vec3 po, vec3 no, vec3 lo, float kappa) {
  float nl = dot(N, L);
  vec3 c = vec3(0.0);
#if SSS_MODE == 0
  c = uAlbedo * RECIPROCAL_PI * E * max(nl, 0.0);
#elif SSS_MODE == 1 || SSS_MODE == 2
  c = uAlbedo * RECIPROCAL_PI * E * max((vec3(nl) + uWrap) / (1.0 + uWrap), 0.0);
#if SSS_MODE == 2
  // fast translucency: view-dependent lobe through the object, attenuated by the thickness behind the point
  float t = exitDist(po - no * 1e-4, -no);
  float back = pow(clamp(dot(V, -(L + N * 0.2)), 0.0, 1.0), 4.0) * 2.0;
  c += uAlbedo * RECIPROCAL_PI * E * back * exp(-uSigmaTr * t);
#endif
#else
#if SSS_MODE == 4 || SSS_MODE == 6
  vec3 alb = uAlbedoMS;
#else
  vec3 alb = uAlbedo;
#endif
#if SSS_MODE >= 5
  if (uShape != 3) c = alb * RECIPROCAL_PI * E * litSurface(po, lo);   // no lit-surface integral for a mesh
  else c = diffusion(alb, nl, E, po, lo, kappa);
#else
  c = diffusion(alb, nl, E, po, lo, kappa);
#endif
#endif
  if (uSpec > 0.0 && nl > 0.0) {    // glossy coat, normalized Blinn-Phong with Schlick Fresnel
    vec3 H = normalize(L + V);
    float f = 0.04 + 0.96 * pow(1.0 - clamp(dot(V, H), 0.0, 1.0), 5.0);
    c += uSpec * f * (72.0 / (8.0 * PI)) * pow(max(dot(N, H), 0.0), 64.0) * nl * E;
  }
  return c;
}

#if SSS_MODE == 4 || SSS_MODE == 6
// single scattering along the view ray inside the object: sum over SS_STEPS depths s_k, spaced by the
// mean extinction (stratified inverse CDF, no noise), each lit through an analytic shadow ray
vec3 singleScatter(vec3 po, vec3 vo, vec3 vView) {
  vec3 wi = -vo;                                     // into the object
  float T = exitDist(po - 1e-5 * vo, wi);
  float sbar = (uSigmaT.r + uSigmaT.g + uSigmaT.b) / 3.0;
  float norm = 1.0 - exp(-sbar * T);
  vec3 acc = vec3(0.0);
  for (int k = 0; k < SS_STEPS; k++) {
    float uk = (float(k) + 0.5) / float(SS_STEPS);
    float s = -log(1.0 - uk * norm) / sbar;
    float pdf = sbar * exp(-sbar * s) / norm;
    vec3 x = po + wi * s;
    vec3 xv = vViewPos + (-vView) * s;               // the same point in view space (rigid transform)
    vec3 lsum = vec3(0.0);
#if NUM_DIR_LIGHTS > 0
    for (int i = 0; i < NUM_DIR_LIGHTS; i++) {
      vec3 lo = toObjDir(directionalLights[i].direction);
      float tl = exitDist(x, lo);
      lsum += directionalLights[i].color * hg(dot(lo, wi)) * exp(-uSigmaT * tl);
    }
#endif
#if NUM_POINT_LIGHTS > 0
    for (int i = 0; i < NUM_POINT_LIGHTS; i++) {
      vec3 lv = pointLights[i].position - xv;
      float dist = length(lv);
      vec3 lo = toObjDir(lv / dist);
      float tl = exitDist(x, lo);
      vec3 e = pointLights[i].color * getDistanceAttenuation(dist, pointLights[i].distance, pointLights[i].decay);
      lsum += e * hg(dot(lo, wi)) * exp(-uSigmaT * tl);
    }
#endif
    acc += uSigmaS * exp(-uSigmaT * s) * lsum / pdf;
  }
  return acc / float(SS_STEPS);
}
#endif

void main() {
  vec3 N = normalize(vNormal);
  vec3 V = isOrthographic ? vec3(0.0, 0.0, 1.0) : normalize(-vViewPos);
  vec3 po = vObjPos, no = toObjDir(N);
  // curvature: analytic for the primitives, interpolated per-vertex values on a mesh (screen-space derivatives
  // of the normal are constant per triangle and would facet a low-poly mesh)
  float kappa = uShape == 3 ? vCurv : curvature(po);
  vec3 rad = vec3(0.0);
#if NUM_DIR_LIGHTS > 0
  for (int i = 0; i < NUM_DIR_LIGHTS; i++) {
    vec3 L = directionalLights[i].direction;
    rad += sssLight(L, directionalLights[i].color, N, V, po, no, toObjDir(L), kappa);
  }
#endif
#if NUM_POINT_LIGHTS > 0
  for (int i = 0; i < NUM_POINT_LIGHTS; i++) {
    IncidentLight il;
    getPointLightInfo(pointLights[i], vViewPos, il);
    rad += sssLight(il.direction, il.color, N, V, po, no, toObjDir(il.direction), kappa);
  }
#endif
  vec3 irr = getAmbientLightIrradiance(ambientLightColor);
#if NUM_HEMI_LIGHTS > 0
  for (int i = 0; i < NUM_HEMI_LIGHTS; i++) irr += getHemisphereLightIrradiance(hemisphereLights[i], N);
#endif
  rad += uAlbedo * RECIPROCAL_PI * irr;               // ambient light: diffusion returns it as albedo A
#if SSS_MODE == 4 || SSS_MODE == 6
  rad += singleScatter(po, toObjDir(V), V);
#endif
  gl_FragColor = vec4(rad, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`;

// shape: { type: 'sphere'|'box'|'cylinder', half: [x, y, z] } in metres (sphere: radius in x;
// cylinder: radius in x, half height in z, axis z). One material per object and mode.
const SHAPE = { sphere: 0, box: 1, cylinder: 2, mesh: 3 };
// mesh: shape = { type: 'mesh', half, vertices, faces, sdf: { dims, min, max, dmax, data (base64, 8-bit) } }
const sdfTextures = new WeakMap();
function sdfTexture(sdf) {
  if (!sdfTextures.has(sdf)) {
    const bytes = Uint8Array.from(atob(sdf.data), ch => ch.charCodeAt(0));
    const tex = new THREE.Data3DTexture(bytes, ...sdf.dims);
    tex.format = THREE.RedFormat;
    tex.type = THREE.UnsignedByteType;
    tex.minFilter = tex.magFilter = THREE.LinearFilter;
    tex.wrapS = tex.wrapT = tex.wrapR = THREE.ClampToEdgeWrapping;
    tex.unpackAlignment = 1;
    tex.needsUpdate = true;
    sdfTextures.set(sdf, tex);
  }
  return sdfTextures.get(sdf);
}
// ssShare: the single-scattering albedo removed from A in mode 4, R1 = (alpha / 2)(1 - mu ln(1 + 1/mu)) of a
// semi-infinite medium lit at mu = cos(incidence) (Chandrasekhar), taken at mu = 1: 0.153 alpha; 0 keeps A.
export function sssMaterial(medium, shape, mode, { spec = 0, ssSteps = 8, burleyS, wrapScale = 1, ssShare = 0.153, sdfSteps = 12 } = {}) {
  const p = mediumParams(typeof medium === 'string' ? MEDIA[medium] : medium, { burleyS });
  const size = shape.type === 'box' || shape.type === 'mesh' ? Math.min(...shape.half) : shape.half[0];
  const v3 = a => new THREE.Vector3(...a);
  const m = new THREE.ShaderMaterial({
    vertexShader: VERT, fragmentShader: FRAG, lights: true,
    defines: { SSS_MODE: mode, SS_STEPS: ssSteps, SDF_STEPS: sdfSteps, SDF_MIN_STEP: '0.0015', SDF_ENTER: '0.006' },
    uniforms: THREE.UniformsUtils.merge([THREE.UniformsLib.lights, {
      uAlbedo: { value: v3(p.A) },
      uAlbedoMS: { value: v3(p.A.map((a, c) => Math.max(0, a - ssShare * p.sigmaS[c] / p.sigmaT[c]))) }, uSigmaTr: { value: v3(p.sigmaTr) }, uBurleyS: { value: v3(p.d.map(d => 1 / d)) },
      uBurleyD: { value: v3(p.d) }, uSigmaS: { value: v3(p.sigmaS) }, uSigmaT: { value: v3(p.sigmaT) }, uG: { value: p.g },
      // wrap: the diffusion length relative to the object's size, a common artist's heuristic
      uWrap: { value: v3(p.sigmaTr.map(s => Math.min(1, Math.max(0.05, wrapScale / s / size)))) },
      uShape: { value: SHAPE[shape.type] }, uHalf: { value: v3(shape.half) }, uSpec: { value: spec },
      uPreLUT: { value: null }, uLitLUT: { value: null }, uSDF: { value: null },
      uSdfMin: { value: v3(shape.sdf?.min ?? [0, 0, 0]) },
      uSdfSize: { value: v3(shape.sdf ? shape.sdf.max.map((x, i) => x - shape.sdf.min[i]) : [1, 1, 1]) },
      uSdfDmax: { value: shape.sdf?.dmax ?? 1 },
    }]),
  });
  m.uniforms.uPreLUT.value = preintegratedTexture();   // after the merge, which would clone the texture
  if (shape.sdf) m.uniforms.uSDF.value = sdfTexture(shape.sdf);
  if (mode >= 5 && (shape.type === 'sphere' || shape.type === 'cylinder')) m.uniforms.uLitLUT.value = shape.type === 'sphere' ? sphereTexture() : cylinderTexture();
  return m;
}
export const sssMaterials = (medium, shape, opts) => SSS_MODES.map((_, mode) => sssMaterial(medium, shape, mode, opts));

// Per-vertex curvature of an indexed mesh, 1/m: the mean over its edges of the normal curvature along the edge,
// |(n_j - n_i) . (p_j - p_i)| / |p_j - p_i|^2, then averaged once with the neighbours. Concave and convex count
// alike (the pre-integration softens both).
export function vertexCurvature(g) {
  const p = g.attributes.position.array, n = g.attributes.normal.array, idx = g.index.array, nv = p.length / 3;
  const sum = new Float64Array(nv), cnt = new Float64Array(nv), nbr = Array.from({ length: nv }, () => new Set());
  for (let f = 0; f < idx.length; f += 3) for (let e = 0; e < 3; e++) {
    const i = idx[f + e], j = idx[f + (e + 1) % 3];
    if (nbr[i].has(j)) continue;
    nbr[i].add(j); nbr[j].add(i);
    let dn = 0, dd = 0;
    for (let k = 0; k < 3; k++) { const d = p[3 * j + k] - p[3 * i + k]; dn += (n[3 * j + k] - n[3 * i + k]) * d; dd += d * d; }
    const kappa = Math.abs(dn) / Math.max(dd, 1e-12);
    sum[i] += kappa; cnt[i]++; sum[j] += kappa; cnt[j]++;
  }
  const raw = Array.from(sum, (x, i) => x / Math.max(cnt[i], 1));
  return Float32Array.from(raw, (x, i) => (x + [...nbr[i]].reduce((a, j) => a + raw[j], 0)) / (1 + nbr[i].size));
}

// three.js geometry matching the analytic shape exactly (cylinder axis along z)
export function sssGeometry(shape) {
  const [a, , c] = shape.half;
  if (shape.type === 'sphere') return new THREE.SphereGeometry(a, 96, 64);
  if (shape.type === 'box') return new THREE.BoxGeometry(...shape.half.map(h => 2 * h));
  if (shape.type === 'mesh') {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(shape.vertices, 3));
    g.setIndex(shape.faces);
    g.computeVertexNormals();
    g.setAttribute('curvature', new THREE.Float32BufferAttribute(vertexCurvature(g), 1));
    return g;
  }
  return new THREE.CylinderGeometry(a, a, 2 * c, 96, 1).rotateX(Math.PI / 2);
}
