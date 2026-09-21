/* ==== LADDER ==== */
// One rung per level the bench can read a block at, coarse first (docs/quantization-filter.md).
// Sizes are shares of the room one block has on screen, not pixels: a phone gives a block a tenth of
// the area a monitor does, and fixed pixels turned the field into one solid blot there.
const LADDER = [
  { name: "ZERO", bits: 0,  sat: 0,    light: 0,    size: 0 },
  { name: "D2",   bits: 2,  sat: 0.26, light: 0.44, size: 0.26 },
  { name: "D4",   bits: 4,  sat: 0.40, light: 0.48, size: 0.38 },
  { name: "D6",   bits: 6,  sat: 0.56, light: 0.52, size: 0.52 },
  { name: "D8",   bits: 8,  sat: 0.74, light: 0.55, size: 0.56 },   // the top rung: the model holds no bf16
];
/* Size grows with the rung but flattens at the top: a sign is read at a glance, and past D6 a larger
   one only crowds its neighbours - the field turns into a blot instead of showing the rung. The top
   of the scale is held at what sat between D6 and D8 before; colour carries the rest of the ladder. */
// A block's colour belongs to its place on the map, not to the lens that happens to cover it: blocks
// that lie together read together, so a region keeps its hue whether a lens is over it or not, and
// removing a lens repaints nothing.
const HUE_SPAN = 280;       // degrees of hue the map is spread over
const HUE_TURN = 190;       // where the span starts, so one corner of the map reads cyan
const HUE_TILT = 0.62;      // how much of the span the horizontal axis carries, the rest is vertical
const LIFT_SAT = 0.18;      // how much colour a lens adds on top of the rung's own

// hsl to rgb, kept small: hue from the place, lightness from the rung, saturation from the lift.
function hsl(h, s, l) {
  const a = s * Math.min(l, 1 - l);
  const f = (n) => {
    const k = (n + h / 30) % 12;
    return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
  };
  return [Math.round(255 * f(0)), Math.round(255 * f(8)), Math.round(255 * f(4))];
}

// The hue of a place: it runs along the diagonal of the map, so neighbours share a colour, far
// regions differ, and nowhere is there a seam - an angle around the center would have one, along the
// ray where it wraps. It does not move when a lens is added, removed or dragged.
function hueAt(x, y) {
  return HUE_TURN + HUE_SPAN * (HUE_TILT * x + (1 - HUE_TILT) * y);
}

// A block is painted by where it lies; the rung it is read at sets how light it is and the lift over
// it how vivid. Behind the glass that leaves a dim but coloured field, never a grey one.
function paintOf(x, y, level, lift) {
  // The rung sets how pure the colour is; the lift adds a little on top. Lightness barely moves -
  // raising it with the rung washed the top of the ladder out into one white tone. Which way the
  // ladder runs is the theme's to say: lighter than the ground at night, darker than it by day.
  const saturation = Math.min(1, LADDER[level].sat + LIFT_SAT * lift);
  return `rgb(${hsl(hueAt(x, y), saturation, RUNG_LIGHT_BY_THEME[themeNow()][level]).join(", ")})`;
}
// How the glass looks at each rung: what is behind it is blurred by that much and veiled in that colour.
// The veil is not the page's own background - a glass has to read as a surface, not as an unlit field.
/* Glass on a budget, the way Aero did it: instead of one expensive blur, a handful of the same
   image offset and stacked at low opacity. The coarser the rung, the wider the offsets - the map
   smears more and reads less - and the veil over it is kept light so a dim glass never goes black. */
/* Two sets, one per theme: the veil is a fog over the map, so it has to be made of the page's own
   light. Dark veils laid over a light page turned the whole field grey. The spreads are shared -
   how much a rung smears is a fact about the rung, not about the palette. */
const GLASS_VEILS = {
  dark: [
    "rgba(9, 14, 26, 0.94)",    // ZERO: emptiness, nothing behind it
    "rgba(13, 20, 38, 0.26)",   // D2: two bits - the map is a haze
    "rgba(12, 18, 34, 0.16)",   // D4: blocks are shapes again, edges soft
    "rgba(11, 17, 32, 0.10)",   // D6: nearly sharp already
    "rgba(11, 17, 32, 0.05)",   // D8: eight bits, a breath of glass left
  ],
  light: [
    "rgba(246, 249, 255, 0.94)",
    "rgba(240, 245, 255, 0.34)",
    "rgba(240, 245, 255, 0.22)",
    "rgba(240, 245, 255, 0.13)",
    "rgba(240, 245, 255, 0.06)",
  ],
};
const GLASS_SPREAD = [0, 5.0, 1.7, 0.7, 0.3];
// the sheet the map is drawn on: near white with a touch of blue by day, deep navy by night
const FIELD_GROUND = { dark: "#0b1120", light: "#f7faff" };
// a rung is lighter than the ground in the dark and darker than it in the light, or it disappears
const RUNG_LIGHT_BY_THEME = {
  dark: [0, 0.44, 0.48, 0.52, 0.55],
  light: [0, 0.62, 0.55, 0.47, 0.40],
};

/* Which palette is on: the reader's own choice when there is one, the device's otherwise. */
function themeNow() {
  const chosen = document.documentElement.getAttribute("data-theme");
  if (chosen === "light" || chosen === "dark") return chosen;
  return matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

function glassLook(index) {
  const theme = themeNow();
  return { spread: GLASS_SPREAD[index], veil: GLASS_VEILS[theme][index] };
}
// where the copies land, in units of the spread, with the weight each carries
const AERO = [[0, 0, 0.40], [1, 0.6, 0.18], [-1, -0.6, 0.18], [0.6, -1, 0.12], [-0.6, 1, 0.12]];
const BF16_BITS = 16;
const SHOWN = 4;   // every SHOWN-th block is drawn: the field reads as points, not as a smear
/* What a block can be drawn as: hearts, faces, stars, small things and a little geometry. None
   of them carries a colour of its own, so each takes the colour of its place on the map, and a
   block keeps the same sign for the life of the page. */
const GLYPHS = ["♥︎", "♡︎", "❤︎", "❥︎", "❦︎", "❧︎", "✿︎", "❀︎", "✾︎", "❁︎", "♠︎", "♣︎", "♦︎", "☺︎", "☻︎", "☼︎", "☀︎", "☁︎", "☂︎", "☃︎", "❄︎", "✹︎", "✺︎", "✻︎", "★︎", "☆︎", "✦︎", "✧︎", "✩︎", "✪︎", "✫︎", "✬︎", "✭︎", "✮︎", "✯︎", "✰︎", "✶︎", "✷︎", "✸︎", "⁂︎", "✳︎", "✴︎", "♪︎", "♫︎", "♬︎", "♩︎", "☕︎", "✈︎", "⚓︎", "⌛︎", "⚡︎", "☘︎", "✂︎", "✎︎", "✉︎", "✆︎", "☯︎", "☮︎", "✔︎", "✚︎", "✜︎", "✠︎", "⚙︎", "⌘︎", "⍟︎", "⧗︎", "❖︎", "❋︎", "❈︎", "▲︎", "▼︎", "◀︎", "▶︎", "◆︎", "◇︎", "○︎", "●︎", "□︎", "■︎", "◈︎", "◉︎", "△︎", "▽︎"];
const GLYPH_SCALE = 2.6;   // a character is drawn larger than the square it replaces
const RADIUS_PER_AREA = 0.18;   // rule 1 on the map: at focus_area 0.5 a new lens covers 0.09 of it
// Lenses of different sizes read better than one size repeated; a placed lens takes the next of these.
const LENS_RADII = [1, 0.62, 1.35, 0.8, 1.1, 0.5];
const PICK_SLACK = 1.25;    // clicking this much past a lens edge still grabs it
/* The zones the page opens with, as shares of the canvas. The hero is square from 760px (site.css) and
   taller than wide below it, so the name sits lower in a phone's field and the zones follow it down;
   on a phone the three run together into one patch behind the name. */
const SQUARE_HERO = "(min-width: 760px)";
const FIRST_ZONES = {
  square: { at: [[0.33, 0.45], [0.67, 0.45], [0.50, 0.61]], scale: 2.0 },
  tall: { at: [[0.33, 0.55], [0.67, 0.55], [0.50, 0.71]], scale: 2.0 },
};
const DPR_CAP = 1.5;        // how fine the canvas is drawn, against how much a frame costs

/* ==== STATE ==== */
const el = (id) => document.getElementById(id);
const canvas = el("map");
const ctx = canvas.getContext("2d");
/* The regulator's settings outlive the page: the docs show the same field, with no sliders on it.
   A control is read from the DOM when the page has it and from storage when it does not. */
const SETTINGS_KEY = "foqlens.controls";
const DEFAULTS = { floor: "1", area: "0.5", strength: "1", combine: "sum" };
const stored = (() => {
  try { return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}") }; }
  catch { return { ...DEFAULTS }; }
})();
// a base precision stored before the ladder lost bf16 (2026-09-17) would point past its top rung
stored.floor = String(Math.min(Number(stored.floor), LADDER.length - 1));
const controls = {};
for (const name of Object.keys(DEFAULTS)) {
  const node = el(name);
  if (node) { node.value = stored[name]; controls[name] = node; }
  else controls[name] = { value: stored[name] };   // no slider on this page: the stored value stands
}
function rememberControls() {
  const out = {};
  for (const name of Object.keys(DEFAULTS)) out[name] = String(controls[name].value);
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(out)); } catch { /* private mode */ }
}
const lenses = [];
let points = null;      // Float32Array [x0, y0, x1, y1, …] in the unit square
let depth = null;       // Float32Array, 0 at the first layer, 1 at the last
let levels = null;      // Uint8Array, the rung each block is read at in the frame being drawn
let view = { w: 0, h: 0, scale: 1, dx: 0, dy: 0, room: 1 };
let drag = null;
const field = document.createElement("canvas");   // the blocks as they are, before the glass
const fieldCtx = field.getContext("2d");
const lift = document.createElement("canvas");    // how much the lenses lift each point, white to black
const liftCtx = lift.getContext("2d");
const sharp = document.createElement("canvas");   // one sharpening layer, masked by the lift
const sharpCtx = sharp.getContext("2d");
// Layers from the glass's blur down to none, each faded by the lift: they add up where the lift is
// strong, so sharpness grows toward the center of a lens instead of jumping at its edge.

/* ==== GEOMETRY ==== */
// The map is drawn "cover": square data, rectangular canvas, no distortion.
function fitView() {
  const rect = canvas.getBoundingClientRect();
  /* A frame costs area: the glass alone lays the field down five times over, and every step of the
     ratio squares that. At 1.5 the signs stay clean against text at the same size, and the frame
     costs half of what it did at 2. */
  const dpr = Math.min(window.devicePixelRatio || 1, DPR_CAP);
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  for (const [buffer, context] of [[field, fieldCtx], [lift, liftCtx], [sharp, sharpCtx]]) {
    buffer.width = canvas.width;
    buffer.height = canvas.height;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  // the map fills the canvas with a margin; x and y keep one scale, so no zone is squashed
  const scale = Math.max(rect.width, rect.height) * 0.92;   // tighter: more blocks per lens, denser field
  // the room a block has: the side of its share of the canvas, so the field reads the same everywhere
  const room = points ? Math.sqrt((rect.width * rect.height) / (points.length / 2)) : 1;
  view = { w: rect.width, h: rect.height, scale, dx: (rect.width - scale) / 2, dy: (rect.height - scale) / 2, room };
}
// one scale for x and y, so a lens is a circle on screen as it is on the map
const toScreen = (x, y) => [view.dx + x * view.scale, view.dy + y * view.scale];
const toMap = (sx, sy) => [(sx - view.dx) / view.scale, (sy - view.dy) / view.scale];
function pointerMap(e) {
  const rect = canvas.getBoundingClientRect();
  return toMap(e.clientX - rect.left, e.clientY - rect.top);
}

/* ==== THE RULES OF docs/quantization-filter.md ==== */
function layout() {
  const floorIndex = Number(controls.floor.value);          // 0 = ZERO … 4 = D8, the whole ladder
  const area = Number(controls.area.value);
  const strength = Number(controls.strength.value);
  const top = LADDER.length - 1;
  const ceiling = floorIndex + Math.floor(strength * (top - floorIndex));   // rule 2
  const radius = RADIUS_PER_AREA * area;                                    // rule 1
  // rule 3: even stops from the ceiling down to the first rung above the floor, the last at the edge
  const rings = [];
  const inner = ceiling - floorIndex;
  for (let i = 0; i < inner; i++) rings.push({ level: ceiling - i, stop: (i + 1) / inner });
  const last = rings.length ? rings[rings.length - 1].stop : 1;
  return { floorIndex, ceiling, radius, rings, last, sum: controls.combine.value === "sum" };
}

// The level of a block and the lift it carries: the lift decides how vivid the block is drawn, the
// nearest lens whose center it magnifies from.
const held = { level: 0, owner: -1, lift: 0 };
function levelOf(x, y, L) {
  held.owner = -1;
  held.lift = 0;
  if (!lenses.length || L.ceiling === L.floorIndex) return (held.level = L.floorIndex);
  let lift = 0, best = 0;                                                   // rules 4 and 5
  for (let i = 0; i < lenses.length; i++) {
    const r = L.radius * lenses[i].scale;
    const one = 1 - Math.hypot(x - lenses[i].x, y - lenses[i].y) / r / L.last;
    if (one <= 0) continue;
    lift = L.sum ? lift + one : Math.max(lift, one);
    if (one > best) { best = one; held.owner = i; }
  }
  if (lift <= 0) return (held.level = L.floorIndex);
  held.lift = Math.min(1, lift);
  const rhoStar = L.last * (1 - held.lift);                                 // rule 6
  for (const ring of L.rings) if (ring.stop >= rhoStar) return (held.level = ring.level);
  return (held.level = L.floorIndex);
}

/* ==== DRAWING ==== */
// Every block at the rung its lenses give it: size and colour per rung, depth in the network as tint.
/* The links between blocks, not a ruled grid: the map comes from co-activation, so blocks that
   light up together lie close, and a short line between neighbours is that fact drawn. Each block
   keeps its few nearest neighbours, found once through a hash of cells so the search is linear
   rather than every block against every other. */
const LINK_NEIGHBOURS = 2;    // lines kept per block
const LINK_REACH = 2.2;       // how far a neighbour may be, in cells
/* The canvas is drawn at the device pixel ratio, so a line of 2 came out as one hairline on a
   retina screen and the web of co-activation read as noise rather than as structure; 3.5 went the
   other way and the lines outweighed the blocks they join. */
const LINK_WIDTH = 2.5;

let links = null;             // [a0, b0, a1, b1, …] indices into points

function buildLinks() {
  const count = depth.length;
  const cell = Math.sqrt(1 / count) * LINK_REACH;
  const key = (cx, cy) => cx * 100003 + cy;
  const cells = new Map();
  for (let i = 0; i < count; i++) {
    const k = key(Math.floor(points[i * 2] / cell), Math.floor(points[i * 2 + 1] / cell));
    (cells.get(k) || cells.set(k, []).get(k)).push(i);
  }
  const out = [];
  for (let i = 0; i < count; i++) {
    const x = points[i * 2], y = points[i * 2 + 1];
    const cx = Math.floor(x / cell), cy = Math.floor(y / cell);
    const near = [];
    for (let ox = -1; ox <= 1; ox++) {
      for (let oy = -1; oy <= 1; oy++) {
        for (const j of cells.get(key(cx + ox, cy + oy)) || []) {
          if (j <= i) continue;                       // each pair once
          const dx = points[j * 2] - x, dy = points[j * 2 + 1] - y;
          near.push([dx * dx + dy * dy, j]);
        }
      }
    }
    near.sort((a, b) => a[0] - b[0]);
    for (const [, j] of near.slice(0, LINK_NEIGHBOURS)) out.push(i, j);
  }
  links = new Int32Array(out);
}

function drawLinks(L) {
  if (!links) return;
  fieldCtx.save();
  fieldCtx.lineWidth = LINK_WIDTH;
  fieldCtx.beginPath();
  for (let k = 0; k < links.length; k += 2) {
    const a = links[k], b = links[k + 1];
    const [ax, ay] = toScreen(points[a * 2], points[a * 2 + 1]);
    const [bx, by] = toScreen(points[b * 2], points[b * 2 + 1]);
    if (Math.max(ax, bx) < 0 || Math.min(ax, bx) > view.w) continue;
    fieldCtx.moveTo(ax, ay);
    fieldCtx.lineTo(bx, by);
  }
  fieldCtx.strokeStyle = "rgba(150, 170, 205, 0.3)";
  fieldCtx.stroke();
  fieldCtx.restore();
}

function drawField(L) {
  fieldCtx.clearRect(0, 0, view.w, view.h);
  /* The sheet the map lies on belongs to the map, not to the page: it is painted here so that both
     the blurred copies and the sharp one cut into the glass carry it. Left transparent, the page
     showed through the blur and the light theme turned the whole field grey. */
  fieldCtx.fillStyle = FIELD_GROUND[themeNow()];
  fieldCtx.fillRect(0, 0, view.w, view.h);
  fieldCtx.textAlign = "center";
  fieldCtx.textBaseline = "middle";
  drawLinks(L);
  /* The level of every block first, then the blocks drawn rung by rung. Setting the font is the
     expensive call on a canvas, and a rung is exactly what sets the size - so it is set once per
     rung instead of once per block, six times instead of three and a half thousand. */
  let bits = 0;
  const level = levels;
  for (let i = 0; i < depth.length; i++) {
    level[i] = levelOf(points[i * 2], points[i * 2 + 1], L);
    bits += LADDER[level[i]].bits;
  }
  for (let rung = 1; rung < LADDER.length; rung++) {
    const size = LADDER[rung].size;
    if (!size) continue;
    // A block is a character, not a square: the map reads as a text of the model rather than a
    // scatter of dots. The glyph is fixed per block, so a block keeps its own shape while the
    // lenses move over it.
    /* The rung alone sets the size. A block read at D8 looks the same whether the base put it
       there or a zone lifted it: one level, one scale - otherwise the same reading has two sizes
       and the picture stops meaning anything. A zone magnifies by raising the level, not by
       adding size on top of it. */
    fieldCtx.font = `${size * view.room * GLYPH_SCALE}px "IBM Plex Mono", ui-monospace, monospace`;
    const lifted = rung > L.floorIndex;
    for (let i = 0; i < depth.length; i++) {
      if (level[i] !== rung) continue;
      const x = points[i * 2], y = points[i * 2 + 1];
      // A zone enlarges what it reads and does not bend it: blocks stay where they are on the map,
      // so at the centre the picture is sharp and true, only larger.
      const [sx, sy] = toScreen(x, y);
      if (sx < -6 || sy < -6 || sx > view.w + 6 || sy > view.h + 6) continue;
      fieldCtx.fillStyle = paintOf(x, y, rung, lifted ? held.lift : 0);
      // behind the glass the map still has to be visible: a coarse base is dim, not empty
      fieldCtx.globalAlpha = lifted ? 1 : 0.72 + 0.22 * depth[i];
      fieldCtx.fillText(GLYPHS[i % GLYPHS.length], sx, sy);
    }
  }
  fieldCtx.globalAlpha = 1;
  return bits / depth.length;
}

function lensPath(target, L) {
  target.beginPath();
  for (const lens of lenses) {
    const [sx, sy] = toScreen(lens.x, lens.y);
    const r = L.radius * lens.scale * L.last * view.scale;
    target.moveTo(sx + r, sy);
    target.arc(sx, sy, r, 0, Math.PI * 2);
  }
}

// How much of the glass is lifted at every point - the same sum of lifts that sets the levels (rules 4-5),
// painted as white where the lenses are strongest. Two lenses that overlap brighten their isthmus by
// themselves, so the bridge between two zones shows up without a rule of its own.
function drawLift(L) {
  /* The mask is read by destination-in, which looks at transparency and not at brightness: away
     from every lens it has to be *clear*, not black. Filled black it was opaque everywhere, the
     sharp layer survived across the whole field, and the glass behind it was never seen. */
  liftCtx.globalCompositeOperation = "source-over";
  liftCtx.clearRect(0, 0, view.w, view.h);
  liftCtx.globalCompositeOperation = L.sum ? "lighter" : "lighten";
  for (const lens of lenses) {
    const [sx, sy] = toScreen(lens.x, lens.y);
    const r = L.radius * lens.scale * L.last * view.scale;
    const cone = liftCtx.createRadialGradient(sx, sy, 0, sx, sy, r);
    cone.addColorStop(0, "rgba(255, 255, 255, 1)");
    cone.addColorStop(1, "rgba(255, 255, 255, 0)");
    liftCtx.fillStyle = cone;
    liftCtx.fillRect(sx - r, sy - r, r * 2, r * 2);
  }
  liftCtx.globalCompositeOperation = "source-over";
}

function draw() {
  if (!points) return;
  const L = layout();
  const meanBits = drawField(L);
  const look = glassLook(L.floorIndex);
  drawLift(L);

  // behind the glass: the field blurred and veiled in the glass's own colour, on the sheet the
  // theme gives the field - transparent, the light palette showed the page through the blur and
  // the whole map read as grey
  ctx.clearRect(0, 0, view.w, view.h);
  if (look.spread) {
    for (const [dx, dy, alpha] of AERO) {
      ctx.globalAlpha = alpha;
      ctx.drawImage(field, dx * look.spread, dy * look.spread, view.w, view.h);
    }
    ctx.globalAlpha = 1;
  } else {
    ctx.drawImage(field, 0, 0, view.w, view.h);
  }
  ctx.fillStyle = look.veil;
  ctx.fillRect(0, 0, view.w, view.h);
  const sheen = ctx.createLinearGradient(0, 0, view.w, view.h);
  sheen.addColorStop(0, "rgba(198, 220, 255, 0.03)");
  sheen.addColorStop(0.5, "rgba(198, 220, 255, 0.004)");
  sheen.addColorStop(1, "rgba(120, 150, 200, 0.025)");
  ctx.fillStyle = sheen;
  ctx.fillRect(0, 0, view.w, view.h);

  // With the floor at the top of the ladder a zone has nothing to lift to, so it must change
  // nothing at all: cutting the glass and laying the same field back in would still cost the
  // sheen and read as a shadow.
  if (!lenses.length || L.ceiling === L.floorIndex) return report(meanBits);

  /* Through the lenses the map is not lit, it is replaced: the glass is cut away where the lift is,
     and the sharp field is laid into the hole. Drawing the sharp copy *over* the glass left both
     visible at once, and their alphas added up into the glare. */
  ctx.globalCompositeOperation = "destination-out";
  ctx.drawImage(lift, 0, 0, view.w, view.h);
  ctx.globalCompositeOperation = "source-over";

  sharpCtx.globalCompositeOperation = "source-over";
  sharpCtx.clearRect(0, 0, view.w, view.h);
  sharpCtx.drawImage(field, 0, 0, view.w, view.h);
  sharpCtx.globalCompositeOperation = "destination-in";   // only what the lenses reach
  sharpCtx.drawImage(lift, 0, 0, view.w, view.h);
  sharpCtx.globalCompositeOperation = "source-over";
  ctx.drawImage(sharp, 0, 0, view.w, view.h);
  ctx.globalAlpha = 1;

  // A lens gathers light, but only a little: in "lighter" this is what washed the map out.
  ctx.globalCompositeOperation = "lighter";
  for (let i = 0; i < lenses.length; i++) {
    const [sx, sy] = toScreen(lenses[i].x, lenses[i].y);
    const r = L.radius * lenses[i].scale * L.last * view.scale;
    const [gr, gg, gb] = hsl(hueAt(lenses[i].x, lenses[i].y), 0.75, 0.62);
    const glow = ctx.createRadialGradient(sx, sy, 0, sx, sy, r);
    glow.addColorStop(0, `rgba(${gr}, ${gg}, ${gb}, 0.05)`);
    glow.addColorStop(0.6, `rgba(${gr}, ${gg}, ${gb}, 0.015)`);
    glow.addColorStop(1, `rgba(${gr}, ${gg}, ${gb}, 0)`);
    ctx.fillStyle = glow;
    ctx.fillRect(sx - r, sy - r, r * 2, r * 2);
  }
  ctx.globalCompositeOperation = "source-over";
  report(meanBits);
}

// A page without the regulator shows the field and nothing else, so every readout is optional.
function report(meanBits) {
  if (!el("bitsOut")) return;
  el("bitsOut").textContent = meanBits.toFixed(2);
  el("shareOut").textContent = Math.round((meanBits / BF16_BITS) * 100) + "%";
  el("lensOut").textContent = String(lenses.length);
  // every rung is named in the one unit they share, so the scale reads as a scale
  // the rung by name first, as the whole bench names it, and the bits it comes to after: a slider and
  // the select it becomes on a narrow screen then read the same
  const rung = LADDER[Number(controls.floor.value)];
  el("floorOut").textContent = `${rung.name} · ${rung.bits}`;
  el("areaOut").textContent = Number(controls.area.value).toFixed(2);
  el("strengthOut").textContent = Number(controls.strength.value).toFixed(2);
}

let queued = false;
function render() {
  if (queued) return;
  queued = true;
  requestAnimationFrame(() => { queued = false; draw(); });
}

/* ==== INTERACTION ==== */
function newLens(x, y) {
  return { x, y, scale: LENS_RADII[lenses.length % LENS_RADII.length] };
}

function lensAt(x, y) {
  const L = layout();
  for (let i = lenses.length - 1; i >= 0; i--) {
    const r = L.radius * lenses[i].scale * L.last * PICK_SLACK;
    if (Math.hypot(x - lenses[i].x, y - lenses[i].y) <= r) return i;  // map units: the lens is a circle there
  }
  return -1;
}

const byFinger = () => matchMedia("(pointer: coarse)").matches;

canvas.addEventListener("pointerdown", (e) => {
  e.preventDefault();                       // no text selection while a lens is dragged
  const [x, y] = pointerMap(e);
  const hit = lensAt(x, y);
  if (byFinger()) {
    // a finger taps: the lens appears or goes at once, and the page is left free to scroll
    if (hit >= 0) lenses.splice(hit, 1);
    else lenses.push(newLens(x, y));
    render();
    return;
  }
  if (hit >= 0) {
    drag = { index: hit, ox: lenses[hit].x - x, oy: lenses[hit].y - y, moved: false };
    if (canvas.hasPointerCapture || e.pointerId !== undefined) try { canvas.setPointerCapture(e.pointerId); } catch { /* synthetic pointer */ }
  } else {
    lenses.push(newLens(x, y));
    render();
  }
});

canvas.addEventListener("pointermove", (e) => {
  if (!drag) return;
  const [x, y] = pointerMap(e);
  const lens = lenses[drag.index];
  if (Math.hypot(x + drag.ox - lens.x, y + drag.oy - lens.y) > 0.002) drag.moved = true;
  lens.x = x + drag.ox;
  lens.y = y + drag.oy;
  render();
});

canvas.addEventListener("pointerup", () => {
  if (drag && !drag.moved) lenses.splice(drag.index, 1);   // a click on a lens removes it
  drag = null;
  render();
});

canvas.addEventListener("wheel", (e) => {
  const [x, y] = pointerMap(e);
  const hit = lensAt(x, y);
  if (hit < 0) return;
  e.preventDefault();
  lenses[hit].scale = Math.min(3, Math.max(0.3, lenses[hit].scale * (e.deltaY < 0 ? 1.12 : 0.89)));
  render();
}, { passive: false });

addEventListener("keydown", (e) => {
  if (e.key === "r" || e.key === "R") { lenses.length = 0; render(); }
});

// a touch screen has no wheel and no keyboard: the hint says what a finger can do, and the sliders
// become numbers - a value is easier to hit and to read than a thumb on a 4 mm track
if (matchMedia("(pointer: coarse)").matches) {
  document.body.classList.add("by-finger");
  // the finger hint is in the markup beside the pointer one, in both languages: .by-finger shows
  // one and hides the other, so no wording lives in this file
  // the floor is a choice of rungs, not a quantity: a finger picks it by name
  const pick = document.createElement("select");
  pick.id = controls.floor.id;
  for (let i = 0; i < LADDER.length; i++) {
    pick.append(new Option(`${LADDER[i].name} · ${LADDER[i].bits} bits`, String(i), false,
                           String(i) === controls.floor.value));
  }
  controls.floor.replaceWith(pick);
  controls.floor = pick;
  pick.addEventListener("input", render);
}

for (const control of Object.values(controls)) {
  if (control.addEventListener) control.addEventListener("input", () => { rememberControls(); render(); });
}
addEventListener("resize", () => { fitView(); render(); });
/* The field is painted in the colours of the theme, so it repaints when the theme changes - by the
   reader's hand on the panel's button, or by the device while the page is open. */
new MutationObserver(render).observe(document.documentElement, { attributeFilter: ["data-theme"] });
matchMedia("(prefers-color-scheme: light)").addEventListener("change", render);

/* ==== DATA ==== */
// The weight map: 14 708 blocks as uint16 pairs, x and y over the unit square.
fetch("site/weight-map.bin")
  .then((r) => r.arrayBuffer())
  .then((buf) => {
    const raw = new Uint16Array(buf);
    const blocks = raw.length / 2;
    /* Half way between the coordinate and its rank. The bare coordinate packs most of the blocks
       into one wall down the left of the embedding, where they overlap into a smear; the bare rank
       spaces them evenly and by doing so erases the clumps, which are the only thing the map has.
       Half of each keeps the clumps and opens the wall enough to see them. */
    /* The wall runs down the first axis, so that one is opened further than the second: the field is
       wider than it is tall, and the first axis is the one with the crowd on it. */
    const SPREAD = [0.85, 0.5];
    const axis = (which) => {
      const value = new Float32Array(blocks);
      for (let i = 0; i < blocks; i++) value[i] = raw[i * 2 + which];
      let lo = Infinity, hi = -Infinity;
      for (const v of value) { if (v < lo) lo = v; if (v > hi) hi = v; }
      const span = hi - lo || 1;
      const order = Array.from(value.keys()).sort((a, b) => value[a] - value[b]);
      const out = new Float32Array(blocks);
      for (let rank = 0; rank < blocks; rank++) {
        const block = order[rank];
        const straight = (value[block] - lo) / span;
        out[block] = straight + SPREAD[which] * (rank / (blocks - 1) - straight);
      }
      return out;
    };
    const [ex, ey] = [axis(0), axis(1)];
    const kept = [];
    for (let i = 0; i < blocks; i += SHOWN) kept.push(i);
    points = new Float32Array(kept.length * 2);
    depth = new Float32Array(kept.length);
    levels = new Uint8Array(kept.length);
    kept.forEach((block, i) => {
      points[i * 2] = ex[block];
      points[i * 2 + 1] = ey[block];
      depth[i] = block / (blocks - 1);
    });
    fitView();   // again, now that the number of blocks is known: it sets the size of a tile
    buildLinks();
    // one lens to start, so the mechanism is visible before anything is touched
    // Directly under the name, where the eye already is. Given as shares of the canvas and turned
    // into map points: the map is fitted to the larger side, so a fixed map coordinate lands
    // somewhere else on every window size - these do not.
    const under = (fx, fy, scale) => {
      const [x, y] = toMap(view.w * fx, view.h * fy);
      return { x, y, scale };
    };
    // a triangle behind the name: two zones above, one below between them, equal radii, so the
    // overlaps are visible as overlaps rather than as one shapeless patch
    const first = matchMedia(SQUARE_HERO).matches ? FIRST_ZONES.square : FIRST_ZONES.tall;
    lenses.push(...first.at.map(([fx, fy]) => under(fx, fy, first.scale)));
    render();
  })
  .catch((e) => {
    // the map is what the field draws; without it the page says so and the reason stays in the console
    document.body.classList.add("no-map");
    console.error("weight map", e);
  });

// The lenses, for a smoke check from the console: place a few and see what the glass does with them.
window.foqlens = { lenses, render };
