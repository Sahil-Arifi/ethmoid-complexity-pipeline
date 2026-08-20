from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def build_directional_phantom(output: Path) -> None:
    shape = (48, 52, 44)
    x, y, z = np.indices(shape, dtype=np.float32)
    center = (np.asarray(shape, dtype=np.float32) - 1.0) / 2.0
    radius = (
        ((x - center[0]) / 20.0) ** 2
        + ((y - center[1]) / 22.0) ** 2
        + ((z - center[2]) / 18.0) ** 2
    )
    data = np.full(shape, -1000, dtype=np.int16)
    data[radius <= 1.0] = 40
    data[(radius >= 0.82) & (radius <= 1.0)] = 1200

    # Asymmetric air cavities make every orthogonal plane visually distinct.
    left_air = ((x - 15) / 5.0) ** 2 + ((y - 30) / 7.0) ** 2 + ((z - 24) / 8.0) ** 2
    right_air = ((x - 33) / 7.0) ** 2 + ((y - 27) / 5.0) ** 2 + ((z - 20) / 6.0) ** 2
    data[(left_air <= 1.0) | (right_air <= 1.0)] = -950

    # Direction-specific blocks: low/high X are L/R, low/high Y are P/A,
    # and low/high Z are I/S in the positive-diagonal RAS affine below.
    data[5:10, 23:29, 19:25] = 300
    data[38:43, 23:29, 19:25] = 1500
    data[21:27, 5:10, 19:25] = 500
    data[21:27, 42:47, 19:25] = 1000
    data[21:27, 23:29, 4:9] = 700
    data[21:27, 23:29, 35:40] = 1300

    affine = np.asarray(
        [
            [0.8, 0.0, 0.0, -center[0] * 0.8],
            [0.0, 0.9, 0.0, -center[1] * 0.9],
            [0.0, 0.0, 1.1, -center[2] * 1.1],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    image = nib.Nifti1Image(data, affine)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    image.header.set_xyzt_units("mm")
    image.header["descrip"] = b"synthetic directional E2E phantom"
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build_directional_phantom(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
