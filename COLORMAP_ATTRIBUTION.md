# Colormap attribution and provenance

`viewer/colormaps.cpp` is a generated 256-sample-per-row atlas of canonical
sRGB-encoded RGBA8 bytes. The bytes are uploaded unchanged to an RGBA8 UNORM
image and nearest-fetched because the viewer renders to a UNORM swapchain
attachment. The viewer has no runtime dependency on Python, Matplotlib,
cmweather, D3, or a shader compiler.

## Atlas rows

Rows 0-2 preserve the viewer's original IDs:

| ID | Viewer label | Definition |
|---:|---|---|
| 0 | Grayscale | Local linear black-to-white ramp |
| 1 | Turbo | D3 `interpolateTurbo`, sampled at `i / 255` |
| 2 | Cyclic phase | Local analytic three-phase cosine map; samples 0 and 255 are identical |

Rows 3-11 come from the color-vision-deficiency-oriented catalog in
[openradar/cmweather `_cm_colorblind.py`](https://github.com/openradar/cmweather/blob/98fabaa4391bd6c793eeb803d1e01636d57fa811/cmweather/_cm_colorblind.py),
pinned to commit `98fabaa4391bd6c793eeb803d1e01636d57fa811`:

| ID | Viewer label | Upstream identifier | Native control samples |
|---:|---|---|---:|
| 3 | cmweather balance | `balance` | 256 |
| 4 | cmweather ChaseSpectral | `ChaseSpectral` | 180 |
| 5 | cmweather SpectralExtended | `SpectralExtended` | 180 |
| 6 | cmweather plasmidis | `plasmidis` | 100 |
| 7 | cmweather bgyp | `bgyp` | 100 |
| 8 | cmweather turbone | `turbone` | 100 |
| 9 | cmweather CM_depol | `CM_depol` | 256 |
| 10 | cmweather CM_rhohv | `CM_rhohv` | 256 |
| 11 | cmweather HomeyerRainbow | `HomeyerRainbow` | 15 generated YUV anchors |

The shorter cmweather tables are linearly resampled exactly as
`matplotlib.colors.LinearSegmentedColormap.from_list` does. These maps retain
their weather/radar provenance: ChaseSpectral, SpectralExtended, and
HomeyerRainbow are reflectivity-oriented; turbone is differential-
reflectivity-oriented; balance is diverging; plasmidis, bgyp, and CM_rhohv are
correlation-oriented; CM_depol is depolarization-oriented. They are offered as
inspection palettes, not represented as SAR-specific semantics. None is used
as the cyclic phase map.

Rows 12-19 come from
[d3/d3-scale-chromatic](https://github.com/d3/d3-scale-chromatic/tree/2c52792197299346b7bdb94322bb4dff8f554fea),
pinned to commit `2c52792197299346b7bdb94322bb4dff8f554fea`
(package version 3.1.0), sampled at `i / 255`:

| ID | Viewer label | Exact D3 interpolator |
|---:|---|---|
| 12 | D3 Viridis | `interpolateViridis` |
| 13 | D3 Cividis | `interpolateCividis` |
| 14 | D3 Inferno | `interpolateInferno` |
| 15 | D3 Magma | `interpolateMagma` |
| 16 | D3 Plasma | `interpolatePlasma` |
| 17 | D3 RdBu (blue-red) | `interpolateRdBu(1 - t)` |
| 18 | D3 PuOr | `interpolatePuOr` |
| 19 | D3 Cubehelix Default | `interpolateCubehelixDefault` |

RdBu is intentionally reversed so low/negative values are blue and
high/positive values are red. D3's `Spectral` is not included: it is a
diverging ColorBrewer map and is neither cmweather's ChaseSpectral nor a cyclic
phase map.

Viridis, Magma, Inferno, and Plasma were designed for Matplotlib by Stéfan van
der Walt, Nathaniel Smith, and Eric Firing and released
[CC0](https://bids.github.io/colormap/). Cividis was designed by Jamie R.
Nuñez, Christopher R. Anderton, and Ryan S. Renslow. Turbo was designed by
Anton Mikhailov at Google. Cubehelix is based on Dave Green's scheme.

cmweather asks users of its CVD-oriented maps to cite Sherman et al. (2024),
"Effective Visualization of Radar Data for Users Impacted by Color Vision
Deficiency," BAMS 105, E1479-E1489,
<https://doi.org/10.1175/BAMS-D-23-0056.1>. For balance it additionally asks
users to cite Thyng et al. (2016), "True colors of oceanography,"
<https://doi.org/10.5670/oceanog.2016.66>.

## openradar/cmweather license

MIT License

Copyright (c) 2023 Open Radar Community

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## d3-scale-chromatic license

Copyright 2010-2024 Mike Bostock

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.

### ColorBrewer notice (applies to RdBu and PuOr)

Apache-Style Software License for ColorBrewer software and ColorBrewer Color
Schemes

Copyright 2002 Cynthia Brewer, Mark Harrower, and The Pennsylvania State
University

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at

<http://www.apache.org/licenses/LICENSE-2.0>

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations under
the License.
