# Visual metaphor

How FoQLens is drawn. Fixed 2026-09-11.

## Why Lens

Mechanically the model does not look through anything - it distributes precision over the weights. The lens in the name is justified by the picture: a field coarsened everywhere except where sharpness is needed.

## The picture - a pixel-art field of weights

Pixel size and palette are tied to bit depth:

- **2 bits** - large squares (16px), 4 colors. Background.
- **4 bits** - smaller squares (8px), 16 colors. The edge of a zone and the isthmus.
- **8 bits** - 4px.
- **16 bits** - 2px, full color. **Only in the antinodes.**

Rules:
- 16 bits only in the centers (antinodes), then stepping down.
- The isthmus is above the background but below the centers - do not fill it with the centers' precision.
- The edge is stepped, not smooth: the shape of the falloff is a subject of measurement, the picture does not assert it.
- Different antinodes hold different knowledge (for example, a color pattern on the left, rings on the right). On the background both are coarsened beyond recognition; they are recognizable only where there are enough bits.
