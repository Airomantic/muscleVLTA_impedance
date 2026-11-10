import json
import re
import xml.etree.ElementTree as ET
import zipfile

from PIL import Image
import pytest

from scripts import build_manuscript as manuscript

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
NAMESPACES = (
    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
    'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
    'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"'
)
SOURCE = """# A Title

@authors A. Author

@abstract

The abstract cites [@b] before [@a].

## 1. Section

Body with *x*_{ref}, **bold** and {{a number to fill in}} [@a; @b].

$$ r_{i} ← \\frac{a}{b} + q_{i}^{T}k $$

!figure wide chart
caption: Figure 1. A chart.

!table column 2 1
caption: Table 1. Counts.
| Method | Far
| Ours | {{TBD}}

!algorithm Algorithm 1. Steps
1: first
2: second

!references
"""


def _properties(run: ET.Element) -> list[str]:
    return [child.tag.removeprefix(W) for child in run.find(f"{W}rPr")]


def test_inline_markup_numbers_citations_and_orders_run_properties():
    document = manuscript.Document({"a": "A.", "b": "B."})
    xml = manuscript.inline("*x*_{ref} {{**fill**}} [@b] then [@a; @b]", document, '<w:sz w:val="16"/>')
    runs = ET.fromstring(f"<w:p {NAMESPACES}>{xml}</w:p>")
    texts = [run.find(f"{W}t").text for run in runs]

    assert texts == ["x", "ref", " ", "fill", " ", "[1]", " then ", "[1, 2]"]
    assert document.cited == ["b", "a"]
    assert document.placeholders == 1
    # WordprocessingML prescribes the order of run properties: b, i, sz, highlight, vertAlign.
    assert _properties(runs[0]) == ["i", "sz"]
    assert _properties(runs[1]) == ["sz", "vertAlign"]
    assert _properties(runs[3]) == ["b", "sz", "highlight"]
    with pytest.raises(KeyError, match="@missing"):
        manuscript.inline("[@missing]", document)


def test_equation_syntax_becomes_office_math():
    xml = manuscript.math("r_{i} ← \\frac{a}{b} + q_{i}^{T}k")
    tags = re.findall(r"<m:(sSub|sSup|sSubSup|f|num|den)>", xml)
    assert tags == ["sSub", "f", "num", "den", "sSubSup"]
    # A script attaches to the symbol before it, not to the text before that symbol.
    assert "<m:e><m:r>" + manuscript.MATH_FONT + '<m:t xml:space="preserve">q</m:t></m:r></m:e>' in xml
    assert '<m:t xml:space="preserve"> + </m:t>' in xml


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    sources = tmp_path / "src" / "manuscript"
    sources.mkdir(parents=True)
    (sources / "10_paper.md").write_text(SOURCE, encoding="utf-8")
    (sources / "references.json").write_text(json.dumps({"a": "First reference.", "b": "Second reference."}))
    Image.new("RGB", (200, 100), "white").save(tmp_path / "chart.png")
    (tmp_path / "chart.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100"/>')
    template = tmp_path / "template.docx"
    with zipfile.ZipFile(template, "w") as package:
        package.writestr(
            "[Content_Types].xml",
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="png" ContentType="image/png"/></Types>',
        )
        package.writestr("word/document.xml", f"<w:document {NAMESPACES}><w:body><w:p/></w:body></w:document>")
        package.writestr(
            "word/_rels/document.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId8" Type="http://x/footer" Target="footer1.xml"/>'
            '<Relationship Id="rId9" Type="http://x/relationships/image" Target="media/old.png"/></Relationships>',
        )
        package.writestr("word/media/old.png", b"old figure of the template")
        package.writestr("word/styles.xml", "<styles/>")
    monkeypatch.setattr(manuscript, "ROOT", tmp_path)
    monkeypatch.setattr(manuscript, "SOURCES", sources)
    monkeypatch.setattr(manuscript, "TEMPLATE", template)
    return tmp_path


def test_manuscript_keeps_the_template_and_embeds_figures_as_svg_with_png_fallback(workspace):
    document = manuscript.build(workspace / "paper.docx")

    assert (document.cited, document.equations, document.placeholders) == (["b", "a"], 1, 2)
    with zipfile.ZipFile(workspace / "paper.docx") as package:
        names = set(package.namelist())
        body = ET.fromstring(package.read("word/document.xml")).find(f"{W}body")
        relationships = package.read("word/_rels/document.xml.rels").decode()
        content_types = package.read("[Content_Types].xml").decode()
    # The template's styles stay, its figures go, the new figure comes with both formats.
    assert {"word/styles.xml", "word/media/figure1.png", "word/media/figure1.svg"} <= names
    assert "word/media/old.png" not in names
    assert "media/old.png" not in relationships
    assert 'Target="media/figure1.svg"' in relationships
    assert 'Target="footer1.xml"' in relationships
    assert 'Extension="svg" ContentType="image/svg+xml"' in content_types

    text = "".join(node.text or "" for node in body.iter(f"{W}t"))
    assert "The abstract cites [1] before [2]." in text
    assert "[1] Second reference." in text
    assert "[2] First reference." in text
    # Title in one column, text in two, the wide figure in one, then two columns again up to the end.
    columns = [section.find(f"{W}cols").get(f"{W}num", "1") for section in body.iter(f"{W}sectPr")]
    assert columns == ["1", "2", "1", "2"]
    assert len(list(body.iter(f"{W}tbl"))) == 1
    # A table is not split between columns: every row but the last keeps with the next one.
    rows = body.find(f".//{W}tbl").findall(f"{W}tr")
    assert [row.find(f".//{W}keepNext") is not None for row in rows] == [True, False]
    assert len(body.findall(f".//{W}drawing")) == 1


def test_png_only_manuscript_has_no_svg_parts(workspace):
    manuscript.build(workspace / "paper_png.docx", png_only=True)
    with zipfile.ZipFile(workspace / "paper_png.docx") as package:
        names = package.namelist()
        document = package.read("word/document.xml").decode()
    assert [name for name in names if name.startswith("word/media/")] == ["word/media/figure1.png"]
    assert "svgBlip" not in document
