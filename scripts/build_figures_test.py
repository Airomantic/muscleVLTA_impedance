import base64
import io
import xml.etree.ElementTree as ET

from PIL import Image

from scripts import build_figures

SVG = "{http://www.w3.org/2000/svg}"
HREF = "{http://www.w3.org/1999/xlink}href"

SOURCE = """<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="100" height="60">
  <defs>
    <style>
      .line { fill: none; stroke: #123456; stroke-width: 2px; }
      .label { font-size: 12px; fill: #111111; }
      .mid { text-anchor: middle; }
    </style>
    <marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8"
            markerUnits="userSpaceOnUse" orient="auto">
      <path d="M0 1 L10 5 L0 9 Z" fill="#654321"/>
    </marker>
    <g id="tag"><rect width="6" height="4" fill="#FF0000"/></g>
  </defs>
  <g stroke="#00FF00">
    <path class="line" d="M10 10 H50 V40" marker-end="url(#head)"/>
  </g>
  <text class="label mid" x="50" y="55" fill="#999999">label</text>
  <use xlink:href="#tag" x="70" y="20"/>
  <image x="5" y="5" width="10" height="5" href="photo.png"/>
</svg>
"""


def test_exported_figure_only_uses_what_office_displays(tmp_path):
    (tmp_path / "figure.svg").write_text(SOURCE)
    # 4 pixels per unit are kept: the 10-unit-wide image is resampled from 200 to 40 pixels.
    Image.new("RGB", (200, 100), (10, 120, 200)).save(tmp_path / "photo.png")

    root = ET.fromstring(build_figures.export(tmp_path / "figure.svg"))

    tags = {element.tag.removeprefix(SVG) for element in root.iter()}
    assert tags == {"svg", "g", "path", "text", "rect", "image"}
    assert not any({"class", "marker-end", "href"} & set(element.attrib) for element in root.iter())

    group, text, tag, image = list(root)
    line, head = list(group)
    assert (line.get("stroke"), line.get("stroke-width"), line.get("fill")) == ("#123456", "2", "none")
    # The marker's triangle, 8 units long, along the last segment (downwards): its reference point (x = 9 of 10) lies
    # on the end of the path. It keeps its own fill and does not take the stroke of the group around the path.
    assert head.get("d") == "M53.20 32.80 L50.00 40.80 L46.80 32.80 Z"
    assert (head.get("fill"), head.get("stroke")) == ("#654321", "none")
    # As in CSS, a class rule overrides the element's presentation attribute.
    assert (text.get("fill"), text.get("font-size"), text.get("text-anchor")) == ("#111111", "12", "middle")
    assert tag.get("transform") == "translate(70 20)"
    (rect,) = tag[0]
    assert (rect.tag, rect.attrib) == (f"{SVG}rect", {"width": "6", "height": "4", "fill": "#FF0000"})
    mime, data = image.get(HREF).removeprefix("data:").split(";base64,")
    assert mime == "image/png"
    assert Image.open(io.BytesIO(base64.b64decode(data))).size == (40, 20)
