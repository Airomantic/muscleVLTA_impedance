"""Exports the paper figures picture/<name>.{svg,pdf,png} from their sources picture/src/<name>.svg.

The sources are written for editing: CSS classes, <marker> arrowheads, <use> and linked image files. The exported SVG
is one self-contained file restricted to what PowerPoint, Word, browsers and Inkscape all display: linked images are
embedded as base64 data URIs (xlink:href; large ones resampled to print resolution), class rules become presentation
attributes, arrowheads become plain paths and <use> is expanded. With librsvg and cairo installed, the PDF for LaTeX
(fonts embedded) and a 600 dpi PNG (for Office versions without SVG support) are written too:

    .venv/bin/python scripts/build_figures.py
"""

import base64
import copy
import ctypes
import io
import math
import pathlib
import re
from typing import ClassVar
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parents[1] / "picture"
# Pixels per SVG unit of embedded images and of the PNG: 600 dpi when the 1000-unit-wide figure is printed 17.5 cm wide.
PIXELS_PER_UNIT = 4.0
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
SVG_NAMESPACE, XLINK_NAMESPACE = "http://www.w3.org/2000/svg", "http://www.w3.org/1999/xlink"
HREF = f"{{{XLINK_NAMESPACE}}}href"
ET.register_namespace("", SVG_NAMESPACE)
ET.register_namespace("xlink", XLINK_NAMESPACE)


def _tag(name: str) -> str:
    return f"{{{SVG_NAMESPACE}}}{name}"


def _load_image(path: pathlib.Path) -> tuple[str, bytes]:
    if path.suffix.lower() != ".svg":
        return MIME[path.suffix.lower()], path.read_bytes()
    # An SVG that only wraps one embedded raster (e.g. a cut-out photo): embed the raster itself.
    text = path.read_text(encoding="utf-8")
    rasters = re.findall(r'href="data:(image/(?:png|jpeg));base64,([^"]+)"', text)
    if len(rasters) == 1 and not re.search(r"<(?:path|rect|circle|text|g)\b", text):
        return rasters[0][0], base64.b64decode(rasters[0][1])
    raise ValueError(f"{path}: only raster images can be embedded for Office (an SVG inside an SVG is not shown)")


def _resample(mime: str, data: bytes, width_units: float) -> tuple[str, bytes]:
    try:
        from PIL import Image
    except ImportError:
        return mime, data
    image = Image.open(io.BytesIO(data))
    target = math.ceil(width_units * PIXELS_PER_UNIT)
    if image.width <= 1.25 * target:
        return mime, data
    image = image.resize((target, round(image.height * target / image.width)), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "image/png", buffer.getvalue()


def embed_images(root: ET.Element, base: pathlib.Path) -> None:
    for image in root.iter(_tag("image")):
        uri = image.attrib.pop("href", None) or image.get(HREF)
        if not uri.startswith("data:"):
            mime, data = _resample(*_load_image((base / uri).resolve()), float(image.get("width")))
            uri = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
        image.set(HREF, uri)


def expand_uses(root: ET.Element) -> set[str]:
    """Replaces every <use> by a translated copy of the element it references; returns the referenced ids."""
    targets = {element.get("id"): element for element in root.iter() if element.get("id")}
    used = set()
    for parent in list(root.iter()):
        for index, child in enumerate(list(parent)):
            if child.tag != _tag("use"):
                continue
            name = (child.get(HREF) or child.get("href")).lstrip("#")
            clone = copy.deepcopy(targets[name])
            del clone.attrib["id"]
            group = ET.Element(_tag("g"), {"transform": f"translate({child.get('x', '0')} {child.get('y', '0')})"})
            group.append(clone)
            group.tail, clone.tail = child.tail, None
            parent[index] = group
            used.add(name)
    return used


def inline_styles(root: ET.Element) -> None:
    """Applies the `.class { property: value; }` rules of the <style> sheets as presentation attributes. As in CSS, a
    rule overrides an attribute of the element, and a later rule an earlier one."""
    rules: dict[str, dict[str, str]] = {}
    for style in root.iter(_tag("style")):
        for name, body in re.findall(r"\.([\w-]+)\s*\{([^}]*)\}", style.text or ""):
            declarations = (part.split(":", 1) for part in body.split(";") if ":" in part)
            rules[name] = {key.strip(): value.strip().removesuffix("px") for key, value in declarations}
    for element in root.iter():
        classes = element.attrib.pop("class", "").split()
        if unknown := set(classes) - set(rules):
            raise ValueError(f"No style rule for the classes {sorted(unknown)}")
        for name, declarations in rules.items():
            if name in classes:
                element.attrib.update(declarations)


def _vertices(path: str) -> list[tuple[float, float]]:
    """Vertices of a path made of M, L, H, V and Z commands."""
    points, x, y, command = [], 0.0, 0.0, "M"
    tokens = iter(re.findall(r"[A-Za-z]|-?\d*\.?\d+", path))
    for token in tokens:
        if token.isalpha():
            command = token
            if command not in ("M", "L", "H", "V", "Z", "m", "l", "h", "v", "z"):
                raise ValueError(f"Unsupported path command {command!r} in {path!r}")
            continue
        value, relative = float(token), command.islower()
        if command in ("H", "h"):
            x = value + x * relative
        elif command in ("V", "v"):
            y = value + y * relative
        else:
            x, y = value + x * relative, float(next(tokens)) + y * relative
        points.append((x, y))
    return points


def draw_arrowheads(root: ET.Element) -> None:
    """Replaces every marker-end by the marker's shape, drawn at the end of the path along its last segment."""
    markers = {marker.get("id"): marker for marker in root.iter(_tag("marker"))}
    for parent in list(root.iter()):
        for child in list(parent):
            reference = child.attrib.pop("marker-end", None)
            if reference is None:
                continue
            marker = markers[reference.removeprefix("url(#").removesuffix(")")]
            if (marker.get("markerUnits"), marker.get("orient")) != ("userSpaceOnUse", "auto"):
                raise ValueError("Arrowhead markers must use markerUnits=userSpaceOnUse and orient=auto")
            _, _, width, height = (float(value) for value in marker.get("viewBox").split())
            scale_x, scale_y = float(marker.get("markerWidth")) / width, float(marker.get("markerHeight")) / height
            ref_x, ref_y = float(marker.get("refX")), float(marker.get("refY"))
            (x0, y0), (x1, y1) = _vertices(child.get("d"))[-2:]
            length = math.hypot(x1 - x0, y1 - y0)
            cos, sin = (x1 - x0) / length, (y1 - y0) / length
            points = []
            for x, y in _vertices(marker[0].get("d")):
                u, v = (x - ref_x) * scale_x, (y - ref_y) * scale_y
                points.append(f"{x1 + u * cos - v * sin:.2f} {y1 + u * sin + v * cos:.2f}")
            # stroke="none": the head must not inherit the stroke of the group around the path.
            attributes = {"d": "M" + " L".join(points) + " Z", "fill": marker[0].get("fill"), "stroke": "none"}
            head = ET.Element(_tag("path"), attributes)
            head.tail = child.tail
            parent.insert(list(parent).index(child) + 1, head)


def export(source: pathlib.Path) -> str:
    root = ET.parse(source).getroot()
    used = expand_uses(root)
    inline_styles(root)
    draw_arrowheads(root)
    embed_images(root, source.parent)
    for defs in root.findall(_tag("defs")):
        for child in list(defs):
            if child.tag in (_tag("style"), _tag("marker")) or child.get("id") in used:
                defs.remove(child)
        if not len(defs):
            root.remove(defs)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def render(svg: pathlib.Path) -> list[pathlib.Path]:
    """Writes the PDF and the PNG next to the SVG with librsvg and cairo; returns the files written."""
    try:
        rsvg, cairo = ctypes.CDLL("librsvg-2.so.2"), ctypes.CDLL("libcairo.so.2")
    except OSError:
        return []
    pointer, double = ctypes.c_void_p, ctypes.c_double

    class Rect(ctypes.Structure):
        _fields_: ClassVar = [("x", double), ("y", double), ("width", double), ("height", double)]

    rsvg.rsvg_handle_new_from_file.restype = pointer
    rsvg.rsvg_handle_new_from_file.argtypes = [ctypes.c_char_p, ctypes.POINTER(pointer)]
    rsvg.rsvg_handle_get_intrinsic_size_in_pixels.argtypes = [pointer, ctypes.POINTER(double), ctypes.POINTER(double)]
    rsvg.rsvg_handle_render_document.argtypes = [pointer, pointer, ctypes.POINTER(Rect), ctypes.POINTER(pointer)]
    cairo.cairo_pdf_surface_create.restype = pointer
    cairo.cairo_pdf_surface_create.argtypes = [ctypes.c_char_p, double, double]
    cairo.cairo_image_surface_create.restype = pointer
    cairo.cairo_image_surface_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    cairo.cairo_surface_write_to_png.argtypes = [pointer, ctypes.c_char_p]
    cairo.cairo_create.restype = pointer
    for name in ("cairo_create", "cairo_show_page", "cairo_destroy", "cairo_surface_destroy"):
        getattr(cairo, name).argtypes = [pointer]

    error = pointer()
    handle = rsvg.rsvg_handle_new_from_file(str(svg).encode(), ctypes.byref(error))
    if not handle:
        raise RuntimeError(f"librsvg could not read {svg}")
    width, height = double(), double()
    rsvg.rsvg_handle_get_intrinsic_size_in_pixels(handle, ctypes.byref(width), ctypes.byref(height))

    def draw(surface: int, scale: float) -> None:
        context = cairo.cairo_create(surface)
        page = Rect(0.0, 0.0, width.value * scale, height.value * scale)
        if not rsvg.rsvg_handle_render_document(handle, context, ctypes.byref(page), ctypes.byref(error)):
            raise RuntimeError(f"librsvg could not render {svg}")
        cairo.cairo_show_page(context)
        cairo.cairo_destroy(context)

    pdf, png = svg.with_suffix(".pdf"), svg.with_suffix(".png")
    # CSS pixels to PDF points.
    surface = cairo.cairo_pdf_surface_create(str(pdf).encode(), width.value * 0.75, height.value * 0.75)
    draw(surface, 0.75)
    cairo.cairo_surface_destroy(surface)
    size = (round(width.value * PIXELS_PER_UNIT), round(height.value * PIXELS_PER_UNIT))
    surface = cairo.cairo_image_surface_create(0, *size)
    draw(surface, PIXELS_PER_UNIT)
    cairo.cairo_surface_write_to_png(surface, str(png).encode())
    cairo.cairo_surface_destroy(surface)
    return [pdf, png]


def main() -> None:
    for source in sorted((ROOT / "src").glob("*.svg")):
        figure = ROOT / source.name
        figure.write_text(export(source), encoding="utf-8")
        outputs = [figure, *render(figure)]
        print(f"{source.relative_to(ROOT)} -> {', '.join(path.name for path in outputs)}")


if __name__ == "__main__":
    main()
