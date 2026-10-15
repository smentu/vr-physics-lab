// Painterly surface textures: a procedural brush-stroke shader rendered once per object at load into a
// mipmapped, tiling texture. At run time a surface costs one texture fetch; the shader never runs per
// frame. That costs less than evaluating it per pixel, and it does not alias, because the texture has
// mipmaps.
// The look is digital oil: a mottled underpainting under three layers of bristle strokes. The strokes
// follow a smooth flow field, and each is jittered in hue, saturation and value around the base colour.
// Lighter strokes are tinted warm and darker ones cool, and a faint canvas grain goes on top. Strokes are
// scattered on a jittered grid that wraps, so each texture tiles seamlessly and is fixed by its seed.
import * as THREE from 'three';

export function rng(seed) {                     // mulberry32
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const STYLE = {
  size: 512,           // texture side, px
  hue: 4,              // hue jitter, degrees
  sat: 0.045,          // saturation jitter
  val: 0.055,          // lightness jitter
  warmCool: 1.0,       // light strokes tinted warm (peach), dark strokes cool (blue-violet); 1 = up to ~9 % per channel
  grain: 0.025,        // canvas grain
  flow: 0.6,           // 0: strokes share one direction; 1: swirling
  under: 0.8,          // underpainting strength, in units of the jitter
  amount: 1.0,         // scales every variation at once: 0 = flat colour
};

const VERT = /* glsl */`
varying vec2 vUv;
void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

const FRAG = /* glsl */`
precision highp float;
uniform vec3 uBase;                 // sRGB base colour
uniform float uSeed, uFlow, uTheta0, uPhase, uHueJ, uSatJ, uValJ, uWarmCool, uGrain, uUnder, uAmount;
uniform vec2 uFreq;
varying vec2 vUv;
#define TAU 6.28318530718

vec4 hash4(vec3 p) {
  vec4 p4 = fract(vec4(p.xyzx) * vec4(0.1031, 0.1030, 0.0973, 0.1099));
  p4 += dot(p4, p4.wzxy + 33.33);
  return fract((p4.xxyz + p4.yzzw) * p4.zywx);
}
float vnoise(vec2 p, float n, float s) {                    // value noise, periodic on an n x n grid
  vec2 i = floor(p * n), f = fract(p * n);
  f = f * f * (3.0 - 2.0 * f);
  float a = hash4(vec3(mod(i, n), s)).x, b = hash4(vec3(mod(i + vec2(1, 0), n), s)).x;
  float c = hash4(vec3(mod(i + vec2(0, 1), n), s)).x, d = hash4(vec3(mod(i + vec2(1, 1), n), s)).x;
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
vec3 rgb2hsl(vec3 c) {
  float mx = max(c.r, max(c.g, c.b)), mn = min(c.r, min(c.g, c.b)), l = 0.5 * (mx + mn);
  if (mx - mn < 1e-5) return vec3(0.0, 0.0, l);
  float d = mx - mn, s = l > 0.5 ? d / (2.0 - mx - mn) : d / (mx + mn);
  float h = mx == c.r ? (c.g - c.b) / d + (c.g < c.b ? 6.0 : 0.0) : mx == c.g ? (c.b - c.r) / d + 2.0 : (c.r - c.g) / d + 4.0;
  return vec3(h / 6.0, s, l);
}
vec3 hsl2rgb(vec3 c) {
  vec3 rgb = clamp(abs(mod(c.x * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
  return c.z + c.y * (rgb - 0.5) * (1.0 - abs(2.0 * c.z - 1.0));
}
// jitter a colour: r in [0,1)^3 -> hue, saturation, value offsets (HSL), then light strokes are tinted
// warm and dark strokes cool by a multiplicative RGB tilt, so hues never wander through green or magenta
vec3 jitter(vec3 hsl, vec3 r, float scale, float vshift) {
  vec3 g = (r * 2.0 - 1.0) * 1.7 * scale * uAmount;
  float dl = uValJ * g.z + vshift * uAmount;
  vec3 c = hsl2rgb(vec3(fract(hsl.x + uHueJ / 360.0 * g.x), clamp(hsl.y + uSatJ * g.y, 0.0, 1.0), clamp(hsl.z + dl, 0.0, 1.0)));
  float k = uWarmCool * clamp(dl / 0.1, -1.0, 1.0);
  vec3 tilt = k > 0.0 ? mix(vec3(1.0), vec3(1.07, 1.0, 0.88), k) : mix(vec3(1.0), vec3(0.9, 0.95, 1.09), -k);
  return clamp(c * tilt, 0.0, 1.0);
}
float angleAt(vec2 p) {
  return uTheta0 + uFlow * (sin(TAU * dot(uFreq, p) + uPhase) + 0.5 * sin(TAU * (2.0 * p.x - p.y) + 2.0 * uPhase));
}
// the topmost stroke of one layer covering p: rgb and coverage
vec4 strokes(vec2 p, vec3 hsl0, float G, float lenC, float widC, float layer, float scale, float vshift, float alpha) {
  vec2 cell = floor(p * G);
  float best = -1.0;
  vec4 outc = vec4(0.0);
  for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) for (int k = 0; k < 2; k++) {
    vec2 c = cell + vec2(dx, dy), cw = mod(c, G);
    vec4 h = hash4(vec3(cw + 0.37 * float(k), uSeed * 7.13 + layer * 13.7 + float(k) * 3.1));
    vec4 h2 = hash4(vec3(cw.yx + 1.7 + float(k), uSeed * 3.31 + layer * 5.9));
    vec2 center = (c + h.xy) / G;
    float ang = angleAt(fract(center)) + (h.z - 0.5) * 0.8;
    float L = lenC / G * (0.6 + 0.8 * h.w), W = widC / G * (0.6 + 0.8 * h2.x);
    vec2 d = p - center, dir = vec2(cos(ang), sin(ang));
    float s = dot(d, dir) / (0.5 * L);
    float v = dot(d, vec2(-dir.y, dir.x)) - (h2.y - 0.5) * 0.3 * L * (1.0 - s * s);   // a slight bend
    float t = v / W + 0.5;
    if (abs(s) > 1.2 || t < 0.0 || t > 1.0) continue;
    float nb = 7.0, bi = floor(t * nb);
    vec4 hb = hash4(vec3(bi + 0.5, h.x * 97.0, h.y * 57.0 + layer));
    float end = 1.0 - 0.35 * hb.x;                                   // bristles run dry at different lengths
    float along = smoothstep(end, end - 0.15, abs(s - (hb.y - 0.5) * 0.2));
    float across = smoothstep(0.5, 0.38, abs(t - 0.5));
    float groove = 0.8 + 0.2 * smoothstep(0.05, 0.3, abs(fract(t * nb) - 0.5));   // gaps between bristles
    float cov = along * across * groove;
    if (cov < 0.02 || h2.z <= best) continue;
    best = h2.z;
    vec3 rgb = jitter(hsl0, vec3(h.z, h.w, h2.w), scale, (h2.z > 0.5 ? 1.0 : -1.0) * vshift);
    rgb *= 1.0 + 0.06 * uAmount * (hb.z - 0.5);                       // each bristle a little lighter or darker
    outc = vec4(rgb, cov * alpha * (0.55 + 0.45 * hb.w));
  }
  return outc;
}
vec3 srgbToLinear(vec3 c) { return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c)); }
void main() {
  vec2 p = vUv;
  vec3 hsl0 = rgb2hsl(uBase);
  // underpainting: two octaves of periodic noise per channel
  vec3 n = vec3(vnoise(p, 4.0, uSeed + 1.0) + 0.5 * vnoise(p, 9.0, uSeed + 2.0),
                vnoise(p, 4.0, uSeed + 3.0) + 0.5 * vnoise(p, 9.0, uSeed + 4.0),
                vnoise(p, 4.0, uSeed + 5.0) + 0.5 * vnoise(p, 9.0, uSeed + 6.0)) / 1.5;
  vec3 col = jitter(hsl0, n, uUnder, 0.0);
  vec4 s;
  s = strokes(p, hsl0, 5.0, 1.1, 0.5, 0.0, 1.2, 0.0, 0.5);    col = mix(col, s.rgb, s.a);   // broad underlayer
  s = strokes(p, hsl0, 9.0, 1.1, 0.38, 1.0, 1.0, 0.0, 0.55);  col = mix(col, s.rgb, s.a);   // main strokes
  s = strokes(p, hsl0, 16.0, 1.0, 0.3, 2.0, 1.2, 0.04, 0.35); col = mix(col, s.rgb, s.a);   // accents
  col *= 1.0 + uGrain * uAmount * (hash4(vec3(floor(p * 512.0), uSeed)).x * 2.0 - 1.0);
  gl_FragColor = vec4(srgbToLinear(clamp(col, 0.0, 1.0)), 1.0);
}`;

let quad = null;

// Render one painterly texture with `renderer` (before or outside an XR session) and return it.
export function bakePainted(renderer, base, seed = 1, opts = {}) {
  const st = { ...STYLE, ...opts };
  const R = rng(seed);
  const rt = new THREE.WebGLRenderTarget(st.size, st.size, {
    wrapS: THREE.RepeatWrapping, wrapT: THREE.RepeatWrapping, generateMipmaps: true,
    minFilter: THREE.LinearMipmapLinearFilter, magFilter: THREE.LinearFilter, colorSpace: THREE.SRGBColorSpace,
    depthBuffer: false,
  });
  if (!quad) {
    quad = { scene: new THREE.Scene(), camera: new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1),
             mat: new THREE.ShaderMaterial({ vertexShader: VERT, fragmentShader: FRAG, depthTest: false, depthWrite: false,
               uniforms: Object.fromEntries(['uBase', 'uSeed', 'uFlow', 'uTheta0', 'uPhase', 'uHueJ', 'uSatJ', 'uValJ',
                 'uWarmCool', 'uGrain', 'uUnder', 'uAmount', 'uFreq'].map(k => [k, { value: null }])) }) };
    quad.scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), quad.mat));
  }
  const u = quad.mat.uniforms;
  const c = new THREE.Color(base);                       // three converts hex to linear; the shader wants sRGB
  u.uBase.value = new THREE.Vector3(...c.convertLinearToSRGB().toArray());
  u.uSeed.value = (seed % 997) + R();
  u.uFlow.value = st.flow; u.uTheta0.value = R() * Math.PI; u.uPhase.value = R() * 6.283;
  u.uFreq.value = new THREE.Vector2(1 + Math.floor(R() * 2), 1 + Math.floor(R() * 2));
  u.uHueJ.value = st.hue; u.uSatJ.value = st.sat; u.uValJ.value = st.val; u.uWarmCool.value = st.warmCool;
  u.uGrain.value = st.grain; u.uUnder.value = st.under; u.uAmount.value = st.amount;
  const prev = renderer.getRenderTarget(), xr = renderer.xr.enabled;
  renderer.xr.enabled = false;
  renderer.setRenderTarget(rt);
  renderer.render(quad.scene, quad.camera);
  renderer.setRenderTarget(prev);
  renderer.xr.enabled = xr;
  rt.texture.anisotropy = Math.min(4, renderer.capabilities.getMaxAnisotropy());
  rt.texture.userData.renderTarget = rt;
  return rt.texture;
}

// UVs of a three.js BoxGeometry scaled so one texture tile spans `tile` metres on every face, with a
// random offset per face. Faces of different sizes then share one brush scale, and no two faces look
// alike. The texture is fixed to the object, so it shows the object's motion.
export function boxUVs(geometry, half, tile, seed = 1) {
  const R = rng(seed);
  const uv = geometry.attributes.uv;
  // BoxGeometry face order +x, -x, +y, -y, +z, -z; face extents along (u, v)
  const dims = [[half[2], half[1]], [half[2], half[1]], [half[0], half[2]], [half[0], half[2]], [half[0], half[1]], [half[0], half[1]]];
  for (let f = 0; f < 6; f++) {
    const su = 2 * dims[f][0] / tile, sv = 2 * dims[f][1] / tile, ou = R(), ov = R();
    for (let k = 0; k < 4; k++) {
      const i = 4 * f + k;
      uv.setXY(i, uv.getX(i) * su + ou, uv.getY(i) * sv + ov);
    }
  }
  uv.needsUpdate = true;
  return geometry;
}
