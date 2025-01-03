@group(0) @binding(6) var u_colormap_texture : texture_1d<f32>;
@group(0) @binding(7) var u_colormap_sampler : sampler;

fn getColor(value: f32) -> vec4<f32> {
    let v = (value - u_function.colormap.x) / (u_function.colormap.y - u_function.colormap.x);
    return textureSample(u_colormap_texture, u_colormap_sampler, v);
}

