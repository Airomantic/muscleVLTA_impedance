"""Builds the Word manuscript from the plain-text sources in picture/src/manuscript/.

The sources (*.md, read in name order) hold the paper text in a small markup, references.json holds the reference
list, and the styles come from the authors' template picture/muscleVLTA_CVPR_style_manuscript.docx (two-column,
CVPR-like typography). Figures are embedded as SVG with a PNG fallback, the way Word stores an inserted SVG;
--png-only embeds the PNGs alone, for Word versions that do not display SVG:

    .venv/bin/python scripts/build_manuscript.py
    .venv/bin/python scripts/build_manuscript.py --png-only --out picture/muscleVLTA_manuscript_v2_png.docx

Markup. Blocks are separated by blank lines:

    # Title                     @authors Names               @abstract (the next paragraph is the abstract)
    ## 1. Section               ### 1.1. Subsection
    $$ a_{i} = b^{2} $$         display equation, numbered automatically
    !figure wide|column FILE    followed by a line "caption: ..." (FILE.svg and FILE.png must both exist)
    !table wide|column W1 W2 .. followed by "caption: ..." and "| cell | cell |" rows (first row = header);
                                the widths are relative. Tables are three-line tables; a row "---" draws a
                                rule above the row that follows it
    !algorithm Title            followed by one line per step
    !columns 1|2                switches the column layout;  !pagebreak;  !references (the cited entries, in order)

Inline: *italic*, **bold**, _{subscript}, ^{superscript}, [@key] or [@key1; @key2] citations, and {{text}} for a
highlighted placeholder that the authors still have to fill in or confirm.
"""

import argparse
from collections.abc import Sequence
import dataclasses
import json
import pathlib
import re
from xml.sax.saxutils import escape
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1] / "picture"
TEMPLATE = ROOT / "muscleVLTA_CVPR_style_manuscript.docx"
SOURCES = ROOT / "src" / "manuscript"
EMU_PER_INCH = 914400
COLUMN_INCHES, PAGE_INCHES = 3.25, 6.875
COLUMN_DXA, PAGE_DXA = 4680, 9900
PAGE = (
    '<w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1170" w:bottom="1620" w:left="1170" '
    'w:header="504" w:footer="936" w:gutter="0"/>'
)
IMAGE_RELATIONSHIP = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
MATH_FONT = '<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/><w:sz w:val="18"/></w:rPr>'
INLINE = re.compile(
    r"(\*\*.+?\*\*|\*[^*\s](?:[^*]*[^*\s])?\*|_\{[^{}]*\}|\^\{[^{}]*\}|\{\{.+?\}\}|\[@[^\]]+\])", flags=re.DOTALL
)


@dataclasses.dataclass
class Document:
    """State shared by the blocks of one manuscript: citation numbers, equation numbers and embedded images."""

    references: dict[str, str]
    png_only: bool = False
    cited: list[str] = dataclasses.field(default_factory=list)
    equations: int = 0
    images: list[tuple[str, bytes]] = dataclasses.field(default_factory=list)
    relationships: list[tuple[str, str]] = dataclasses.field(default_factory=list)
    placeholders: int = 0

    def cite(self, keys: str) -> str:
        numbers = []
        for key in (part.strip().removeprefix("@") for part in keys.split(";")):
            if key not in self.references:
                raise KeyError(f"[@{key}] is not in references.json")
            if key not in self.cited:
                self.cited.append(key)
            numbers.append(self.cited.index(key) + 1)
        return "[" + ", ".join(str(number) for number in sorted(numbers)) + "]"

    def add_image(self, name: str, data: bytes) -> str:
        self.images.append((name, data))
        relationship = f"rIdImage{len(self.images)}"
        self.relationships.append((relationship, f"media/{name}"))
        return relationship


RUN_PROPERTY_ORDER = ("rFonts", "b", "i", "noProof", "color", "sz", "highlight", "vertAlign")


def run(text: str, *properties: str) -> str:
    """A text run; the properties are written in the order the WordprocessingML schema prescribes."""
    unique = sorted(set(properties), key=lambda item: RUN_PROPERTY_ORDER.index(item[3:].split(" ")[0].rstrip("/>")))
    formatting = f"<w:rPr>{''.join(unique)}</w:rPr>" if unique else ""
    return f'<w:r>{formatting}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def inline(text: str, document: Document, *properties: str) -> str:
    """Runs of a text with inline markup; ``properties`` apply to all of them (e.g. a font size)."""
    runs = []
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**"):
            runs.append(inline(part[2:-2], document, *properties, "<w:b/>"))
        elif part.startswith("{{"):
            document.placeholders += 1
            runs.append(inline(part[2:-2], document, *properties, '<w:highlight w:val="yellow"/>'))
        elif part.startswith("[@"):
            runs.append(run(document.cite(part[1:-1]), *properties))
        elif part.startswith("_{"):
            runs.append(inline(part[2:-1], document, *properties, '<w:vertAlign w:val="subscript"/>'))
        elif part.startswith("^{"):
            runs.append(inline(part[2:-1], document, *properties, '<w:vertAlign w:val="superscript"/>'))
        elif part.startswith("*"):
            runs.append(inline(part[1:-1], document, *properties, "<w:i/>"))
        else:
            runs.append(run(part, *properties))
    return "".join(runs)


def paragraph(content: str, properties: str = '<w:jc w:val="both"/>') -> str:
    return f"<w:p><w:pPr>{properties}</w:pPr>{content}</w:p>"


def _math_run(text: str) -> str:
    return f'<m:r>{MATH_FONT}<m:t xml:space="preserve">{escape(text)}</m:t></m:r>'


def _math_group(text: str, start: int) -> tuple[str, int]:
    """The contents of the {...} group that starts at ``start`` and the index after it."""
    depth = 0
    for index in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[index], 0)
        if depth == 0:
            return text[start + 1 : index], index + 1
    raise ValueError(f"Unbalanced braces in equation: {text}")


def math(text: str) -> str:
    """OMML of an equation in a minimal syntax: x_{i}, x^{2}, x_{i}^{2}, \\frac{a}{b}, {groups}; other characters
    are copied (write Greek letters and operators as Unicode). A script attaches to the preceding group, or to the
    preceding symbol (the characters since the last space, operator or bracket)."""
    output: list[str] = []
    index, pending = 0, ""

    def flush() -> None:
        nonlocal pending
        if pending:
            output.append(_math_run(pending))
            pending = ""

    while index < len(text):
        character = text[index]
        if text.startswith("\\frac", index):
            flush()
            numerator, index = _math_group(text, index + 5)
            denominator, index = _math_group(text, index)
            output.append(f"<m:f><m:num>{math(numerator)}</m:num><m:den>{math(denominator)}</m:den></m:f>")
        elif character == "{":
            flush()
            group, index = _math_group(text, index)
            output.append(math(group))
        elif character in "_^":
            match = re.search(r"[^\s+\-\u2212=·,;:()\[\]{}←→≤≥<>|~]+$", pending)
            if match:
                base, pending = _math_run(match.group(0)), pending[: match.start()]
                flush()
            else:
                flush()
                base = output.pop() if output else _math_run("")
            scripts = {}
            while index < len(text) and text[index] in "_^":
                kind = text[index]
                scripts[kind], index = _math_group(text, index + 1)
            sub, sup = scripts.get("_"), scripts.get("^")
            if sub is not None and sup is not None:
                output.append(
                    f"<m:sSubSup><m:e>{base}</m:e><m:sub>{math(sub)}</m:sub><m:sup>{math(sup)}</m:sup></m:sSubSup>"
                )
            elif sub is not None:
                output.append(f"<m:sSub><m:e>{base}</m:e><m:sub>{math(sub)}</m:sub></m:sSub>")
            else:
                output.append(f"<m:sSup><m:e>{base}</m:e><m:sup>{math(sup)}</m:sup></m:sSup>")
        else:
            pending += character
            index += 1
    flush()
    return "".join(output)


def equation(text: str, document: Document) -> str:
    document.equations += 1
    number = run(f"   ({document.equations})", '<w:sz w:val="18"/>')
    return paragraph(
        f"<m:oMath>{math(text.strip())}</m:oMath>{number}", '<w:ind w:firstLine="0"/><w:jc w:val="center"/>'
    )


def figure(path: pathlib.Path, caption: str, document: Document, *, wide: bool) -> str:
    from PIL import Image

    png, svg = path.with_suffix(".png"), path.with_suffix(".svg")
    with Image.open(png) as image:
        aspect = image.height / image.width
    width = round((PAGE_INCHES if wide else COLUMN_INCHES) * EMU_PER_INCH)
    height = round(width * aspect)
    number = len(document.images) + 1
    blip = f'<a:blip r:embed="{document.add_image(f"figure{number}.png", png.read_bytes())}"'
    if document.png_only:
        blip += "/>"
    else:
        vector = document.add_image(f"figure{number}.svg", svg.read_bytes())
        blip += (
            '><a:extLst><a:ext uri="{96DAC541-7B7A-43D3-8B79-37D633B846F1}">'
            '<asvg:svgBlip xmlns:asvg="http://schemas.microsoft.com/office/drawing/2016/SVG/main" '
            f'r:embed="{vector}"/></a:ext></a:extLst></a:blip>'
        )
    drawing = (
        '<w:r><w:rPr><w:noProof/></w:rPr><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{width}" cy="{height}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>'
        f'<wp:docPr id="{number}" name="Figure {number}" descr="{escape(path.stem)}"/><wp:cNvGraphicFramePr>'
        '<a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/>'
        '</wp:cNvGraphicFramePr><a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr>'
        f'<pic:cNvPr id="{number}" name="{escape(path.name)}"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill>{blip}'
        f'<a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/>'
        f'<a:ext cx="{width}" cy="{height}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
        "</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>"
    )
    return paragraph(drawing, '<w:keepNext/><w:ind w:firstLine="0"/><w:jc w:val="center"/>') + paragraph(
        inline(caption, document), '<w:pStyle w:val="af7"/>'
    )


def table(
    weights: Sequence[float], caption: str, rows: Sequence[Sequence[str]], document: Document, *, wide: bool
) -> str:
    """A three-line table: heavy rules above and below, light rules under the header and where the source says so."""
    total = PAGE_DXA if wide else COLUMN_DXA
    widths = [round(total * weight / sum(weights)) for weight in weights]
    margins = "".join(f'<w:{side} w:w="60" w:type="dxa"/>' for side in ("top", "left", "bottom", "right"))
    content: list[Sequence[str]] = []
    ruled = {1}
    for cells in rows:
        if list(cells) == ["---"]:
            ruled.add(len(content))
        else:
            content.append(cells)
    heavy, light = 10, 4  # eighths of a point
    body = []
    for row_index, cells in enumerate(content):
        if len(cells) != len(widths):
            raise ValueError(f"Table row {cells} has {len(cells)} cells, expected {len(widths)}")
        header = row_index == 0
        last = row_index == len(content) - 1
        rules = {
            "top": heavy if header else light if row_index in ruled else 0,
            "left": 0,
            "bottom": heavy if last else light if header else 0,
            "right": 0,
        }
        border = "".join(
            f'<w:{side} w:val="single" w:sz="{size}" w:space="0" w:color="000000"/>'
            if size
            else f'<w:{side} w:val="nil"/>'
            for side, size in rules.items()
        )
        # Every row but the last stays with the next one, so a table is not split between columns or pages.
        keep = "<w:keepNext/>" if not last else ""
        row = ["<w:tr><w:trPr><w:cantSplit/>" + ("<w:tblHeader/>" if header else "") + "</w:trPr>"]
        for column, (cell, width) in enumerate(zip(cells, widths, strict=True)):
            text = inline(cell, document, *(["<w:b/>"] if header else []), '<w:sz w:val="16"/>')
            align = '<w:jc w:val="center"/>' if column else '<w:jc w:val="left"/>'
            row.append(
                f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/><w:tcBorders>{border}</w:tcBorders>'
                f"<w:tcMar>{margins}</w:tcMar></w:tcPr>"
                + paragraph(text, f'{keep}<w:spacing w:after="20"/><w:ind w:firstLine="0"/>{align}')
                + "</w:tc>"
            )
        body.append("".join(row) + "</w:tr>")
    grid = "".join(f'<w:gridCol w:w="{width}"/>' for width in widths)
    return (
        paragraph(inline(caption, document), '<w:pStyle w:val="af7"/><w:keepNext/>')
        + f'<w:tbl><w:tblPr><w:tblW w:w="{total}" w:type="dxa"/><w:tblLayout w:type="fixed"/></w:tblPr>'
        + f"<w:tblGrid>{grid}</w:tblGrid>{''.join(body)}</w:tbl>"
        + paragraph("", '<w:spacing w:after="60"/><w:ind w:firstLine="0"/>')
    )


def algorithm(title: str, steps: Sequence[str], document: Document) -> str:
    def properties(left: int, rule: str = "") -> str:
        border = f'<w:pBdr><w:{rule} w:val="single" w:sz="6" w:space="1" w:color="000000"/></w:pBdr>' if rule else ""
        return (
            f'<w:keepNext/><w:keepLines/>{border}<w:spacing w:after="20"/><w:ind w:left="{left}" w:hanging="{left}"/>'
        )

    blocks = [paragraph(inline(title, document, "<w:b/>", '<w:sz w:val="18"/>'), properties(0, "top"))]
    for index, step in enumerate(steps):
        indent = 200 * (len(step) - len(step.lstrip(" ")))
        rule = "bottom" if index == len(steps) - 1 else ""
        blocks.append(paragraph(inline(step.strip(), document, '<w:sz w:val="18"/>'), properties(260 + indent, rule)))
    return "".join(blocks) + paragraph("", '<w:spacing w:after="60"/><w:ind w:firstLine="0"/>')


def section_break(columns: int, *, first: bool) -> str:
    """Paragraph that ends a section laid out in ``columns`` columns."""
    footer = '<w:footerReference w:type="default" r:id="rId8"/>' if first else '<w:type w:val="continuous"/>'
    layout = f'<w:cols w:num="{columns}" w:space="450"/>' if columns > 1 else '<w:cols w:space="720"/>'
    return f'<w:p><w:pPr><w:sectPr>{footer}{PAGE}{layout}<w:docGrid w:linePitch="360"/></w:sectPr></w:pPr></w:p>'


def build_body(source: str, document: Document) -> str:
    blocks: list[str] = []
    columns, sections = 1, 0
    abstract = False

    def layout(wanted: int) -> None:
        nonlocal columns, sections
        if wanted != columns:
            blocks.append(section_break(columns, first=sections == 0))
            columns, sections = wanted, sections + 1

    for block in re.split(r"\n\s*\n", source.strip()):
        lines = block.strip("\n").split("\n")
        head = lines[0]
        if head.startswith("# "):
            blocks.append(paragraph(inline(head[2:], document), '<w:pStyle w:val="aa"/><w:jc w:val="center"/>'))
        elif head.startswith("@authors "):
            blocks.append(
                paragraph(
                    inline(head[9:], document),
                    '<w:pStyle w:val="aff9"/><w:spacing w:after="240"/><w:jc w:val="center"/>',
                )
            )
        elif head == "@abstract":
            layout(2)
            blocks.append(paragraph(run("Abstract"), '<w:pStyle w:val="1"/>'))
            abstract = True
        elif head.startswith("### "):
            blocks.append(paragraph(inline(head[4:], document), '<w:pStyle w:val="21"/>'))
        elif head.startswith("## "):
            abstract = False
            blocks.append(paragraph(inline(head[3:], document), '<w:pStyle w:val="1"/>'))
        elif head.startswith("$$"):
            blocks.append(equation(block.strip().strip("$"), document))
        elif head.startswith("!columns "):
            layout(int(head.split()[1]))
        elif head == "!pagebreak":
            blocks.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
        elif head.startswith(("!figure ", "!table ")):
            kind, size, *arguments = head.split()
            caption = lines[1].removeprefix("caption:").strip()
            previous = columns
            if size == "wide":
                layout(1)
            if kind == "!figure":
                blocks.append(figure(ROOT / arguments[0], caption, document, wide=size == "wide"))
            else:
                rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines[2:]]
                weights = [float(weight) for weight in arguments]
                blocks.append(table(weights, caption, rows, document, wide=size == "wide"))
            layout(previous)
        elif head.startswith("!algorithm "):
            blocks.append(algorithm(head[11:], lines[1:], document))
        elif head == "!references":
            for number, key in enumerate(document.cited, start=1):
                text = inline(f"[{number}] {document.references[key]}", document, '<w:sz w:val="18"/>')
                blocks.append(paragraph(text, '<w:ind w:left="200" w:hanging="200"/><w:jc w:val="left"/>'))
        else:
            text = " ".join(line.strip() for line in lines)
            blocks.append(paragraph(inline(text, document, *(["<w:i/>"] if abstract else []))))
    final = f'<w:cols w:num="{columns}" w:space="450"/>' if columns > 1 else '<w:cols w:space="450"/>'
    blocks.append(f'<w:sectPr><w:type w:val="continuous"/>{PAGE}{final}<w:docGrid w:linePitch="360"/></w:sectPr>')
    return "".join(blocks)


def build(out: pathlib.Path, *, png_only: bool = False) -> Document:
    references = json.loads((SOURCES / "references.json").read_text(encoding="utf-8"))
    source = "\n\n".join(path.read_text(encoding="utf-8") for path in sorted(SOURCES.glob("*.md")))
    document = Document(references, png_only=png_only)
    body = build_body(source, document)
    with zipfile.ZipFile(TEMPLATE) as template, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as package:
        for item in template.infolist():
            data = template.read(item.filename)
            if item.filename.startswith("word/media/"):
                continue
            if item.filename == "word/document.xml":
                text = data.decode("utf-8")
                data = (text[: text.index("<w:body>")] + f"<w:body>{body}</w:body></w:document>").encode("utf-8")
            elif item.filename == "word/_rels/document.xml.rels":
                text = re.sub(r'<Relationship [^>]*Type="[^"]*/image"[^>]*/>', "", data.decode("utf-8"))
                added = "".join(
                    f'<Relationship Id="{identifier}" Type="{IMAGE_RELATIONSHIP}" Target="{target}"/>'
                    for identifier, target in document.relationships
                )
                data = text.replace("</Relationships>", added + "</Relationships>").encode("utf-8")
            elif item.filename == "[Content_Types].xml" and not png_only:
                text = data.decode("utf-8")
                svg = '<Default Extension="svg" ContentType="image/svg+xml"/>'
                data = text.replace("<Default ", svg + "<Default ", 1).encode("utf-8")
            package.writestr(item, data)
        for name, data in document.images:
            package.writestr(f"word/media/{name}", data)
    return document


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "muscleVLTA_manuscript_v2.docx")
    parser.add_argument("--png-only", action="store_true", help="Embed the PNG figures without their SVG.")
    args = parser.parse_args(argv)
    document = build(args.out, png_only=args.png_only)
    print(
        f"wrote {args.out}: {len(document.cited)} references, {document.equations} equations, "
        f"{len(document.images)} image parts, {document.placeholders} highlighted placeholders for the authors"
    )


if __name__ == "__main__":
    main()
