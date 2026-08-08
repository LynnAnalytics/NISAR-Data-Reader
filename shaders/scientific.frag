#version 450

layout(set = 0, binding = 0) uniform sampler2D scientific_texture_0;
layout(set = 0, binding = 1) uniform sampler2D scientific_texture_1;
layout(set = 0, binding = 2) uniform sampler2D colormap_lut;

layout(push_constant) uniform DisplayPushConstants
{
    float low;
    float high;
    float gamma;
    uint colormap; // Stable row ID in the 256 x 20 RGBA8 LUT.
    vec2 window_uv_origin;
    vec2 window_uv_dx;
    vec2 window_uv_dy;
    uvec2 valid_extent_0;
    vec2 sample_uv_origin_0;
    vec2 sample_uv_extent_0;
    uvec2 valid_extent_1;
    vec2 sample_uv_origin_1;
    vec2 sample_uv_extent_1;
    vec2 sample_weights;
    uint active_slots;
    uint sampling_mode; // 0: smooth, 1: exact pixels.
    uint circular_phase;
    uint padding;
} display;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

const uint colormap_width = 256u;
const uint colormap_count = 20u;

bool finite_value(float value)
{
    return !isnan(value) && !isinf(value);
}

float fetch_scientific(uint slot, ivec2 texel)
{
    return slot == 0u
        ? texelFetch(scientific_texture_0, texel, 0).r
        : texelFetch(scientific_texture_1, texel, 0).r;
}

ivec2 allocation_extent(uint slot)
{
    return slot == 0u
        ? textureSize(scientific_texture_0, 0)
        : textureSize(scientific_texture_1, 0);
}

void accumulate_linear(
    float value,
    float weight,
    inout float weighted_value,
    inout float finite_weight)
{
    if (finite_value(value))
    {
        weighted_value += weight * value;
        finite_weight += weight;
    }
}

void accumulate_phase(
    float value,
    float weight,
    inout vec2 weighted_phase,
    inout float finite_weight)
{
    if (finite_value(value))
    {
        weighted_phase += weight * vec2(cos(value), sin(value));
        finite_weight += weight;
    }
}

bool sample_scientific(
    uint slot,
    uvec2 valid_extent,
    vec2 sample_uv_origin,
    vec2 sample_uv_extent,
    vec2 window_uv,
    out float value)
{
    ivec2 texture_extent = allocation_extent(slot);
    if (any(equal(valid_extent, uvec2(0))) ||
        any(greaterThan(valid_extent, uvec2(texture_extent))))
    {
        return false;
    }

    vec2 sample_uv =
        sample_uv_origin + window_uv * sample_uv_extent;
    if (any(lessThan(sample_uv, vec2(0.0))) ||
        any(greaterThan(sample_uv, vec2(1.0))))
    {
        return false;
    }

    vec2 logical = sample_uv * vec2(valid_extent);
    ivec2 maximum_texel = ivec2(valid_extent) - ivec2(1);
    ivec2 central_texel = min(ivec2(floor(logical)), maximum_texel);

    if (display.sampling_mode != 0u)
    {
        value = fetch_scientific(slot, central_texel);
        return finite_value(value);
    }

    // Align the interpolation lattice to texel centers. The nearest texel is
    // authoritative for validity, so smoothing never crosses a masked pixel.
    vec2 sample_position = logical - vec2(0.5);
    ivec2 base = ivec2(floor(sample_position));
    vec2 fraction = fract(sample_position);
    ivec2 p00 = clamp(base, ivec2(0), maximum_texel);
    ivec2 p10 = clamp(base + ivec2(1, 0), ivec2(0), maximum_texel);
    ivec2 p01 = clamp(base + ivec2(0, 1), ivec2(0), maximum_texel);
    ivec2 p11 = clamp(base + ivec2(1, 1), ivec2(0), maximum_texel);
    float v00 = fetch_scientific(slot, p00);
    float v10 = fetch_scientific(slot, p10);
    float v01 = fetch_scientific(slot, p01);
    float v11 = fetch_scientific(slot, p11);

    value = v11;
    if (all(equal(central_texel, p00)))
    {
        value = v00;
    }
    else if (all(equal(central_texel, p10)))
    {
        value = v10;
    }
    else if (all(equal(central_texel, p01)))
    {
        value = v01;
    }
    if (!finite_value(value))
    {
        return false;
    }

    float w00 = (1.0 - fraction.x) * (1.0 - fraction.y);
    float w10 = fraction.x * (1.0 - fraction.y);
    float w01 = (1.0 - fraction.x) * fraction.y;
    float w11 = fraction.x * fraction.y;
    float finite_weight = 0.0;

    if (display.circular_phase != 0u)
    {
        // Interpolate directions rather than radians so the -pi/+pi seam
        // follows the short arc. Opposing phases retain the nearest value.
        vec2 weighted_phase = vec2(0.0);
        accumulate_phase(v00, w00, weighted_phase, finite_weight);
        accumulate_phase(v10, w10, weighted_phase, finite_weight);
        accumulate_phase(v01, w01, weighted_phase, finite_weight);
        accumulate_phase(v11, w11, weighted_phase, finite_weight);
        float phase_length_squared = dot(weighted_phase, weighted_phase);
        if (finite_weight > 0.0 &&
            phase_length_squared >
                1.0e-12 * finite_weight * finite_weight)
        {
            value = atan(weighted_phase.y, weighted_phase.x);
        }
    }
    else
    {
        float weighted_value = 0.0;
        accumulate_linear(v00, w00, weighted_value, finite_weight);
        accumulate_linear(v10, w10, weighted_value, finite_weight);
        accumulate_linear(v01, w01, weighted_value, finite_weight);
        accumulate_linear(v11, w11, weighted_value, finite_weight);
        if (finite_weight > 0.0)
        {
            value = weighted_value / finite_weight;
        }
    }
    return true;
}

vec3 colorize(float value)
{
    float window_width = display.high - display.low;
    float normalized =
        abs(window_width) > 1.0e-20
            ? clamp((value - display.low) / window_width, 0.0, 1.0)
            : 0.0;
    normalized = pow(normalized, 1.0 / max(display.gamma, 1.0e-4));

    int sample_index = int(normalized * float(colormap_width - 1u) + 0.5);
    int row = int(min(display.colormap, colormap_count - 1u));
    return texelFetch(colormap_lut, ivec2(sample_index, row), 0).rgb;
}

void main()
{
    vec2 window_uv = display.window_uv_origin +
        v_uv.x * display.window_uv_dx +
        v_uv.y * display.window_uv_dy;
    const float edge_tolerance = 1.0e-5;
    if (any(lessThan(window_uv, vec2(-edge_tolerance))) ||
        any(greaterThan(window_uv, vec2(1.0 + edge_tolerance))))
    {
        discard;
    }
    window_uv = clamp(window_uv, 0.0, 1.0);

    float value_0 = 0.0;
    float value_1 = 0.0;
    bool valid_0 = (display.active_slots & 1u) != 0u &&
        sample_scientific(
            0u,
            display.valid_extent_0,
            display.sample_uv_origin_0,
            display.sample_uv_extent_0,
            window_uv,
            value_0);
    bool valid_1 = (display.active_slots & 2u) != 0u &&
        sample_scientific(
            1u,
            display.valid_extent_1,
            display.sample_uv_origin_1,
            display.sample_uv_extent_1,
            window_uv,
            value_1);

    if (!valid_0 && !valid_1)
    {
        discard;
    }
    if (valid_0 && valid_1)
    {
        float weight_0 = max(display.sample_weights.x, 0.0);
        float weight_1 = max(display.sample_weights.y, 0.0);
        float total_weight = weight_0 + weight_1;
        float blend = total_weight > 1.0e-8
            ? clamp(weight_1 / total_weight, 0.0, 1.0)
            : 0.5;
        out_color = vec4(
            mix(colorize(value_0), colorize(value_1), blend), 1.0);
        return;
    }

    // Exclusive coverage remains fully opaque throughout the transition.
    // This prevents dark seams where a regional native page and its overview
    // do not have identical source coverage.
    out_color = vec4(colorize(valid_0 ? value_0 : value_1), 1.0);
}
