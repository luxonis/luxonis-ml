"""Payloads: attach an arbitrary value to a box (the OCR motivating case).

Any box can carry a ``payload`` — a string, int, or float — that is rendered on
its label chip. The classic use is OCR: a box around a word plus the transcribed
text. Here each box shows a detected token and its transcription.
"""

from _common import gradient, save

from luxonis_ml.vizlab import BBox, Image


def main() -> None:
    """Render OCR-style boxes whose chips show the transcribed text."""
    img = Image(gradient(560, 300, hue=0.12))
    img.add(
        BBox((40, 60, 210, 70), label="word", score=0.99, payload="INVOICE")
    )
    img.add(
        BBox((300, 60, 220, 70), label="word", score=0.97, payload="#1042")
    )
    img.add(
        BBox(
            (40, 180, 480, 70),
            label="line",
            score=0.94,
            payload="Total due: $1,299.00",
        )
    )
    save(img, "04_payload.png")


if __name__ == "__main__":
    main()
