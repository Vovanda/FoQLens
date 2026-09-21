"""The pages of the site share one frame - this is what keeps them from drifting apart.

Every page carries the same bar and the same footer, and pulls the same stylesheet and the same
scripts. When each page held its own copy they diverged silently: different headers, a menu that
remembered a different side on each page, a button that sat on the wrong side of one of them. These
tests fail the moment that starts again.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ("index.html", "docs.html")
# what differs between pages on purpose: which link is current, and what the panel is called - the
# front page lists its own sections, the documentation page lists the documents
ALLOWED = (
    re.compile(r' aria-current="page"'),
    re.compile(r'title="(The regulator|Documents|Sections of this page)"'),
)
# Static files are linked with a version so a browser cannot serve a stale copy of them; the tests
# match the path and ignore whatever version is on it.
def asset(path: str) -> re.Pattern[str]:
    return re.compile(rf'"{re.escape(path)}(\?v=\d+)?"')


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def block(page: str, opening: str, tag: str) -> str:
    """One element of the page frame, from the given opening tag to its close."""
    start = page.index(opening)
    end = page.index(f"</{tag}>", start) + len(f"</{tag}>")
    return page[start:end]


def normalized(html: str) -> str:
    for pattern in ALLOWED:
        html = pattern.sub("", html)
    return re.sub(r"\s+", " ", html).strip()


@pytest.mark.parametrize("name", PAGES)
def test_every_page_pulls_the_shared_frame(name):
    page = read(name)
    assert asset("site/site.css").search(page), f"{name} does not link the shared stylesheet"
    assert asset("site/panel.js").search(page), f"{name} does not use the shared panel"
    assert "<style>" not in page, f"{name} still carries styles of its own; they belong in site/"


@pytest.mark.parametrize("name", PAGES)
def test_static_files_are_versioned_together(name):
    """One version per page: a half-bumped page serves a new stylesheet against an old script, and
    that is exactly how a browser showed a layout no one had written."""
    page = read(name)
    versions = set(re.findall(r'"site/[\w.-]+\?v=(\d+)"', page))
    unversioned = re.findall(r'"(site/[\w.-]+)"', page)
    assert not unversioned, f"{name} links {unversioned} without a version"
    assert len(versions) == 1, f"{name} mixes versions of its static files: {versions}"


@pytest.mark.parametrize("name", PAGES)
def test_nothing_moves_while_the_page_lays_itself_out(name):
    """The panel's place comes from storage after the HTML is parsed, so the page opens with
    transitions off and turns them on once everything stands where it belongs."""
    page = read(name)
    assert 'class="preload"' in page, f"{name} does not start with transitions off"
    assert 'classList.remove("preload")' in page, (
        f"{name} never turns transitions back on - and it has to do it itself, "
        "a cached script would leave the page frozen"
    )


@pytest.mark.parametrize("name", PAGES)
def test_the_theme_is_set_before_the_first_paint(name):
    """Applied later, the page shows one palette and then swaps to the other."""
    head = read(name)[: read(name).index("</head>")]
    assert 'setAttribute("data-theme"' in head, f"{name} picks its theme after the first paint"
    assert '"foqlens.theme"' in head, f"{name} does not read the remembered theme"


def test_the_bar_is_the_same_on_every_page():
    bars = {name: normalized(block(read(name), '<div class="bar">', "div")) for name in PAGES}
    first, *rest = bars.values()
    for other in rest:
        assert other == first, "the pages carry different bars"


def test_the_footer_is_the_same_on_every_page():
    footers = [normalized(block(read(name), "<footer>", "footer")) for name in PAGES]
    assert len(set(footers)) == 1, "the pages carry different footers"
    assert "sawking.tech" in footers[0]


@pytest.mark.parametrize("name", PAGES)
def test_the_page_says_where_it_is(name):
    page = read(name)
    assert page.count('aria-current="page"') == 1, f"{name} marks no current page, or more than one"


def test_the_documentation_lists_its_documents_without_scripts():
    """A crawler, a link preview and a reader without JavaScript all get the same empty page
    otherwise - which is how an agent came back saying it could not read the site."""
    page = read("docs.html")
    body = page[page.index('<main'):page.index("</main>")]
    listed = len(re.findall(r'\["[\w-]+", "(?:docs|prereg|experiments)/', page))
    assert body.count("<li>") == listed, (
        f"the table of contents in the HTML has {body.count('<li>')} entries against {listed} in the "
        "shelf - a crawler and a reader without JavaScript see the stale one"
    )
    assert "blob/main/" in body, "the documents are not linked to their sources"


def test_the_panel_state_is_shared_between_pages():
    """One person, one preference: the side of the panel, whether it is open, which theme is on and
    which language is read are the reader's, not the page's. Anything else remembered here would be
    per-page state hiding in a shared file."""
    panel = (ROOT / "site" / "panel.js").read_text(encoding="utf-8")
    keys = set(re.findall(r'localStorage\.(?:get|set)Item\((\w+)', panel))
    assert keys <= {"SIDE_KEY", "OPEN_KEY", "THEME_KEY", "LANG_KEY"}, f"the panel remembers something else: {keys}"
    assert 'const SIDE_KEY = "foqlens.side"' in panel
    assert 'const OPEN_KEY = "foqlens.nav"' in panel
    assert 'const THEME_KEY = "foqlens.theme"' in panel
    assert 'const LANG_KEY = "foqlens.lang"' in panel


def test_no_wording_is_written_by_the_scripts():
    """Both languages stand in the markup, so that a reader without scripts - a crawler, a model
    fetching the page - gets the text. A script that writes a sentence would put that sentence
    outside the markup and outside the language switch with it."""
    for name in ("field.js", "panel.js", "sections.js"):
        script = (ROOT / "site" / name).read_text(encoding="utf-8")
        written = re.findall(r'(?:textContent|innerHTML)\s*=\s*("(?:[^"\\]|\\.)*"|`[^`]*`)', script)
        for value in written:
            words = re.findall(r"[A-Za-z]{3,}", re.sub(r"\$\{[^}]*\}", "", value))
            assert not words, f"{name} writes wording into the page: {value}"
