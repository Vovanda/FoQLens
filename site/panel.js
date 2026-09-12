/* ==== THE PANEL ==============================================================================
   A push panel: it slides out and moves the whole frame, bar included, because moving only the
   middle would let the panel lie over the bar - shifted and covered at once. The page yields the
   panel's width minus the free gutter, never less than zero; on a wide screen it yields nothing.

   Sizes are read from the CSS rather than repeated here: two copies of a number drift apart. */
const cssPx = (name, fallback) => {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name);
  const n = parseFloat(v);
  return isNaN(n) ? fallback : n;
};
const MENU_W = cssPx("--panelw", 272), CONTENT_MAX = cssPx("--maxw", 1000);
const SIDE_KEY = "foqlens.side";
const OPEN_KEY = "foqlens.nav";
const THEME_KEY = "foqlens.theme";

/* The theme cycles through the three states in this order; "auto" is the default and means the
   device decides. The face of the button says which one is on, not which one comes next. */
const THEMES = ["auto", "light", "dark"];
const THEME_FACE = {
  auto: ["◐", "Theme: follows the device"],
  light: ["☀", "Theme: light"],
  dark: ["☾", "Theme: dark"],
};

function applyTheme(name) {
  const theme = THEMES.includes(name) ? name : "auto";
  document.documentElement.setAttribute("data-theme", theme);
  const [face, title] = THEME_FACE[theme];
  const button = document.getElementById("side-theme");
  button.textContent = face;
  button.title = title;
  button.setAttribute("aria-label", title);
  try { localStorage.setItem(THEME_KEY, theme); } catch { /* private mode */ }
}

function cycleTheme() {
  const now = document.documentElement.getAttribute("data-theme") || "auto";
  applyTheme(THEMES[(THEMES.indexOf(now) + 1) % THEMES.length]);
}
const st = { side: localStorage.getItem(SIDE_KEY) === "left" ? "left" : "right" };

function layoutVars() {
  const root = document.documentElement, W = root.clientWidth;
  const gutter = (W - Math.min(CONTENT_MAX, W)) / 2;
  const wide = gutter >= MENU_W;
  document.body.classList.toggle("wide", wide);
  // the gutter is published on every screen, not only a wide one: the field breaks out of the
  // column by exactly this much to run to the edges of the window
  root.style.setProperty("--gutter", Math.round(gutter) + "px");
  if (wide) {
    root.style.setProperty("--gapend", Math.round(gutter - MENU_W) + "px");
    root.style.setProperty("--bw", Math.round(W - gutter) + "px");
    root.style.setProperty("--cl", Math.round(st.side === "left" ? 0 : gutter) + "px");
    root.style.setProperty("--gutter", Math.round(gutter) + "px");
  }
  applyShift();
}

function applyShift() {
  const root = document.documentElement, body = document.body, W = root.clientWidth;
  const wide = body.classList.contains("wide");
  const gutter = (W - Math.min(CONTENT_MAX, W)) / 2;
  const dir = st.side === "left" ? 1 : -1;
  // panel-hidden parks the panel out of sight while it is carried to the other side
  const open = !body.classList.contains("panel-hidden") && (wide || body.classList.contains("nav-open"));
  root.style.setProperty("--panel-shift", (open ? 0 : dir * Math.round(wide ? MENU_W : gutter)) + "px");
  root.style.setProperty("--shift", (open && !wide ? dir * Math.round(MENU_W - gutter) : 0) + "px");
}

function setNav(open) {
  localStorage.setItem(OPEN_KEY, open ? "1" : "0");
  document.body.classList.toggle("nav-open", open);
  document.getElementById("burger").setAttribute("aria-expanded", String(open));
  applyShift();
}
function closeNav() { setNav(false); }

/* Swapping sides is the same movement as the button: the panel goes out of sight, is carried across
   with the transition off, and comes back. There is no separate animation for it. */
function applySide() {
  document.body.classList.toggle("side-left", st.side === "left");
  layoutVars();
  localStorage.setItem(SIDE_KEY, st.side);
}
let _switching = 0;
function swapSide() {
  if (_switching) return;
  const body = document.body, sb = document.getElementById("sidebar");
  _switching = 1;
  body.classList.add("panel-hidden"); applyShift();
  const show = () => { body.classList.remove("panel-hidden"); applyShift(); _switching = 0; };
  setTimeout(() => {
    body.classList.add("no-anim");
    st.side = st.side === "left" ? "right" : "left";
    applySide();
    void sb.offsetWidth;            // force the reflow so the move lands before transitions return
    body.classList.remove("no-anim");
    requestAnimationFrame(show);
    setTimeout(show, 60);
  }, 260);
}

/* On a phone the panel covers most of the screen, so the first tap on the sheet only closes it -
   the tap is caught and does not reach the field, which would otherwise place a lens. */
const PHONE = matchMedia("(max-width: 640px)");
document.getElementById("below").addEventListener("click", (e) => {
  if (!PHONE.matches || !document.body.classList.contains("nav-open")) return;
  e.preventDefault(); e.stopPropagation();
  closeNav();
}, true);

document.getElementById("burger").onclick = () => setNav(!document.body.classList.contains("nav-open"));
document.getElementById("side-theme").onclick = cycleTheme;
document.getElementById("side-swap").onclick = swapSide;
document.getElementById("side-close").onclick = closeNav;
addEventListener("keydown", (e) => { if (e.key === "Escape") closeNav(); });
new ResizeObserver(layoutVars).observe(document.documentElement);
// the attribute is already on <html> from the head; this puts the matching face on the button
applyTheme(document.documentElement.getAttribute("data-theme"));
applySide();
// the panel opens where it was left, on either page
if (localStorage.getItem(OPEN_KEY) === "1") setNav(true);
