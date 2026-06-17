#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "memoria" / "figuras" / "reinicio"

CONVERGENCIA = FIG_DIR / "curva_convergencia_reinicio_cec2017_f10_shade.png"
DIVERSIDAD = FIG_DIR / "curva_diversidad_reinicio_cec2017_f10_shade.png"


def cargar_fuente(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidatos = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for ruta in candidatos:
        if Path(ruta).exists():
            return ImageFont.truetype(ruta, size)
    return ImageFont.load_default()


def etiqueta_vertical(im: Image.Image, texto: str, x: int, y_centro: int, font) -> None:
    draw = ImageDraw.Draw(im)
    bbox = draw.textbbox((0, 0), texto, font=font)
    ancho = bbox[2] - bbox[0]
    alto = bbox[3] - bbox[1]
    tmp = Image.new("RGBA", (ancho + 8, alto + 8), (255, 255, 255, 0))
    tmp_draw = ImageDraw.Draw(tmp)
    tmp_draw.text((4, 4), texto, fill=(0, 0, 0, 255), font=font)
    rotada = tmp.rotate(90, expand=True)
    y = int(y_centro - rotada.height / 2)
    im.alpha_composite(rotada, (x, y))


def editar_convergencia() -> None:
    im = Image.open(CONVERGENCIA).convert("RGBA")
    draw = ImageDraw.Draw(im)
    fuente = cargar_fuente(28)

    # Título interno.
    draw.rectangle((0, 0, im.width, 64), fill=(255, 255, 255, 255))
    # Etiqueta original del eje Y.
    draw.rectangle((0, 70, 82, im.height - 60), fill=(255, 255, 255, 255))
    etiqueta_vertical(im, "Error medio log.", 24, im.height // 2, fuente)
    # Etiqueta del eje X. La deja solo en la subfigura inferior.
    draw.rectangle((360, im.height - 55, 720, im.height), fill=(255, 255, 255, 255))

    im.convert("RGB").save(CONVERGENCIA)


def editar_diversidad() -> None:
    im = Image.open(DIVERSIDAD).convert("RGBA")
    draw = ImageDraw.Draw(im)

    # Título interno. Se mantiene el eje X porque esta es la subfigura inferior.
    draw.rectangle((0, 0, im.width, 64), fill=(255, 255, 255, 255))

    im.convert("RGB").save(DIVERSIDAD)


def main() -> None:
    editar_convergencia()
    editar_diversidad()


if __name__ == "__main__":
    main()
