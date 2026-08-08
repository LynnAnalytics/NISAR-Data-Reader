# Third-party notices

NISAR Data Reader uses the following third-party software and generated assets.
Their licenses apply to those components, not to NISAR Data Reader itself.

## Runtime and linked dependencies

- **HDF5** — 3-clause BSD-style license. Retain the HDF5 copyright, license,
  contributor, and institutional notices supplied with the exact HDF5 build.
  <https://www.hdfgroup.org/licenses/>
- **zlib** — zlib license. The source notice must be retained; modified source
  versions must be identified as modified.
  <https://github.com/madler/zlib/blob/master/README>
- **nlohmann/json** — MIT license. The single-header distribution also
  contains separately credited third-party material; see its upstream license
  and license directory.
  <https://github.com/nlohmann/json#license>
- **SDL3** — zlib license.
  <https://github.com/libsdl-org/SDL/blob/main/LICENSE.txt>
- **Dear ImGui** — MIT license. Dear ImGui also credits its bundled fonts and
  public-domain stb components.
  <https://github.com/ocornut/imgui/blob/master/LICENSE.txt>
- **Vulkan loader and headers** — Khronos Apache-style licensing and notices.
  <https://github.com/KhronosGroup/Vulkan-Loader/blob/main/LICENSE.txt>

When distributing a binary package, include the exact `copyright`, `LICENSE`,
and `NOTICE` files shipped by the resolved dependency versions, where present.

## Build-only dependency

- **glslang** — used as the build-time `glslangValidator` shader compiler. It
  has multiple component licenses, including BSD, MIT, Apache, and a GPL
  license with a special bison exception. The installed application contains
  compiled SPIR-V shaders, not the glslang executable.
  <https://github.com/KhronosGroup/glslang/blob/main/LICENSE.txt>

## Platform and vendor components

- **NVIDIA CUDA Toolkit** — proprietary NVIDIA license terms apply to the CUDA
  runtime and any other CUDA Toolkit files redistributed with the application.
  Only the components identified as redistributable by the applicable CUDA
  Toolkit EULA may be shipped.
  <https://docs.nvidia.com/cuda/eula/>
- **Microsoft Visual C++ runtime** — Microsoft Visual Studio license terms
  apply to the redistributable runtime files. They must be redistributed
  unmodified and only as permitted by the applicable Visual Studio terms.
  <https://learn.microsoft.com/en-us/cpp/windows/redistributing-visual-cpp-files>

## Generated colormap atlas

The generated colormap atlas is documented separately, including its cmweather,
D3, ColorBrewer, and CC0 provenance and notices:

- [COLORMAP_ATTRIBUTION.md](COLORMAP_ATTRIBUTION.md)
