# Visual metaphor

How FoQLens is drawn. Fixed 2026-09-11, updated 2026-09-12 to the lens mechanism ([lens.md](lens.md)).

## Why Lens

Mechanically the model does not look through anything - it distributes precision over the weights. The lens in the name is justified by the picture: a field coarsened everywhere except where sharpness is needed.

## The picture - a pixel-art field of weights

Pixel size and palette follow the rung of the ladder a block is read at:

| Rung | Pixel | Palette |
| --- | --- | --- |
| ZERO | none - the cell is empty | the ground shows through |
| 2 bits | 16px | 4 colors |
| 4 bits | 8px | 16 colors |
| 6 bits | 6px | 64 colors |
| 8 bits | 4px | 256 colors |
| 16 bits (bf16) | 2px | full color - only for the original model, never inside a lens |

Rules:

- The glass sets the whole field: coarse pixels for a glass at a low rung, an empty field for a ZERO glass.
- The lens centers are at the ceiling - the rung focus_strength reaches - then the rings step down to the glass.
- The edge is stepped, not smooth: the shape of the falloff is set by the profile's stops, the picture does not invent it.
- Behind an empty glass the lowest rung may show as a thin ring just outside a lens - the soft edge into emptiness.
- Where two lenses overlap their power adds up: the core of a deep overlap can be as sharp as the centers, the junction rises smoothly from the edges.
- Different centers hold different knowledge (for example, a color pattern on the left, rings on the right). Behind the glass both are coarsened beyond recognition, or gone; they are recognizable only where there are enough bits.
