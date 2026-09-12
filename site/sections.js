/* ==== WHERE THE READER IS ====================================================================
   The panel of the front page lists the sections of the page it stands on, and marks the one being
   read - the same mark the documentation page puts on the document it shows, so one panel behaves
   one way across the site.

   Invariant: exactly one link carries aria-current="page" whenever the page has sections; it is the
   last section whose top has passed under the bar, or the first section while the page is above it. */
(() => {
  const links = [...document.querySelectorAll("#menu a[href^='#']")];
  const spots = links
    .map((link) => ({ link, section: document.getElementById(decodeURIComponent(link.hash.slice(1))) }))
    .filter((spot) => spot.section);
  if (!spots.length) return;

  const barHeight = () => {
    const bar = document.querySelector(".bar");
    return bar ? bar.getBoundingClientRect().height : 0;
  };

  let current = null;
  function mark() {
    // a section counts as read once its top is at the bar; the last such one wins
    const line = barHeight() + 1;
    let found = spots[0];
    for (const spot of spots) {
      if (spot.section.getBoundingClientRect().top <= line) found = spot;
    }
    if (found.link === current) return;
    if (current) current.removeAttribute("aria-current");
    found.link.setAttribute("aria-current", "page");
    current = found.link;
  }

  let queued = false;
  function onScroll() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; mark(); });
  }

  addEventListener("scroll", onScroll, { passive: true });
  addEventListener("resize", onScroll);
  mark();
})();
