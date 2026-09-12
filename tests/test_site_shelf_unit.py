"""The documentation page shows every document of the repository, and serves them as files.

Two ways the site has already gone quiet on a reader:

- a document was added to `docs/` and never reached the shelf in `docs.html`, so it existed in the
  repository and nowhere else;
- GitHub Pages ran the site through Jekyll, which treats a Markdown file with a YAML header as a
  page of its own and stops serving the file itself. Five documents answered 404 on the published
  site - exactly the five with a header, the ones that carry Hugo's front matter.

The first is caught by comparing the folder with the shelf. The second is kept away by `.nojekyll`,
whose presence is checked here: without it the page loads and every document with a header is gone.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs.html"
# the documents of the bench that the page is expected to carry, by folder
SHELVED = (("docs", "*.md"), ("prereg", "*.md"))


def shelf() -> set[str]:
    """Every path the page lists, as written in its shelf."""
    return set(re.findall(r'"((?:docs|prereg|experiments)/[^"]+\.md)"', PAGE.read_text(encoding="utf-8")))


def test_the_page_serves_files_rather_than_jekyll_pages():
    assert (ROOT / ".nojekyll").exists(), (
        "without .nojekyll GitHub Pages runs Jekyll, which converts every Markdown file that has a "
        "YAML header into a page and stops serving the file - the documents then answer 404"
    )


def test_every_document_of_the_repository_is_on_the_shelf():
    listed = shelf()
    missing = sorted(
        f"{folder}/{path.name}"
        for folder, pattern in SHELVED
        for path in (ROOT / folder).glob(pattern)
        if f"{folder}/{path.name}" not in listed
    )
    assert not missing, f"documents in the repository that the page does not show: {missing}"


def test_the_shelf_points_at_documents_that_exist():
    gone = sorted(path for path in shelf() if not (ROOT / path).exists())
    assert not gone, f"the page lists documents that are not in the repository: {gone}"
