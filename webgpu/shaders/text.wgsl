#import font
#import camera

@group(0) @binding(30) var<storage> u_text : Texts;

struct OverlayUniforms {
    corner: vec2f,
    scale: f32,
    padding: f32,
};
@group(0) @binding(31) var<uniform> u_overlay: OverlayUniforms;

struct Texts {
  n_texts: u32,
  data: array<u32>,
};

struct TextData {
  pos: vec3f,
  normal: vec3f,
  color: vec4f,
  shift: vec2f,
  length: u32,
  ichar: u32,
  char: u32,
  apply_camera: u32,
};

fn textLoadData(i: u32) -> TextData {
    // Char data starts after n_texts * 8 u32s of text data
    let offset = u_text.n_texts * 8u + i * 2u;
    let itext = u_text.data[ offset ];
    let char_data = u_text.data[ offset + 1u ];
    let ichar = extractBits(char_data, 0u, 16u);
    let char = extractBits(char_data, 16u, 16u);

    let offset_text = itext * 8u;
    let pos = vec3f(
        bitcast<f32>(u_text.data[offset_text]),
        bitcast<f32>(u_text.data[offset_text + 1u]),
        bitcast<f32>(u_text.data[offset_text + 2u])
    );
    let text_packed = u_text.data[offset_text + 3u];
    let length = extractBits(text_packed, 0u, 16u);
    let apply_camera = extractBits(text_packed, 16u, 8u);
    let x_align = f32(extractBits(text_packed, 24u, 2u));
    let y_align = f32(extractBits(text_packed, 26u, 2u));

    let normal = vec3f(
        bitcast<f32>(u_text.data[offset_text + 4u]),
        bitcast<f32>(u_text.data[offset_text + 5u]),
        bitcast<f32>(u_text.data[offset_text + 6u])
    );

    // Color stored as u8x4 packed in one u32
    let color_packed = u_text.data[offset_text + 7u];
    let color = vec4f(
        f32(extractBits(color_packed, 0u, 8u)) / 255.0,
        f32(extractBits(color_packed, 8u, 8u)) / 255.0,
        f32(extractBits(color_packed, 16u, 8u)) / 255.0,
        f32(extractBits(color_packed, 24u, 8u)) / 255.0
    );

    let shift = vec2<f32>(-0.5 * x_align - 0.278 / f32(length), -0.5 * y_align - 0.20);

    return TextData(pos, normal, color, shift, length, ichar, char, apply_camera);
}

struct TextVertexOutput {
    @builtin(position) fragPosition: vec4f,
    @location(0) tex_coord: vec2f,
    @location(1) @interpolate(flat) color: vec4f,
};

@vertex
fn vertexText(@builtin(vertex_index) vertexId: u32, @builtin(instance_index) charId: u32) -> TextVertexOutput {
    let text = textLoadData(charId);
    var out: TextVertexOutput;
    out.color = text.color;

    var position: vec4f;
    if (text.apply_camera == 2u) {
        // Overlay/gizmo mode: rotation only + corner positioning
        let rotated = (u_camera.rot_mat * vec4f(text.pos, 0.0)).xyz;
        let rot_normal = (u_camera.rot_mat * vec4f(text.normal, 0.0)).xyz;

        // Visibility: hide if normal points away from camera
        let nlen = length(text.normal);
        if (nlen > 0.01 && rot_normal.z < 0.15) {
            out.fragPosition = vec4f(0.0, 0.0, -2.0, 1.0);
            out.tex_coord = vec2f(0.0);
            return out;
        }

        var gizmo_offset = rotated.xy * u_overlay.scale;
        gizmo_offset.x /= u_camera.aspect;
        let ndc = u_overlay.corner + gizmo_offset;
        position = vec4f(ndc, 0.005, 1.0);
    } else if (text.apply_camera == 1u) {
        position = cameraMapPoint(text.pos);
    } else {
        position = vec4f(text.pos, 1.0);
    }

    let char_size = fontGetSizeOnScreen();
    position.x += f32(text.ichar) * char_size.z * position.w;

    let shift = text.shift;
    position.x += char_size.x * shift.x * f32(text.length) * position.w;
    position.y += char_size.y * shift.y * position.w;

    // snap position to pixel grid
    let resolution = vec2f(f32(u_camera.width), f32(u_camera.height));
    let ndc = position.xy / position.w;
    let screen = (ndc * 0.5 + vec2f(0.5)) * resolution;
    let snapped_screen = floor(screen) + 0.5;
    let snapped_ndc = (snapped_screen / resolution - vec2f(0.5)) * 2.0;
    position.x = snapped_ndc.x * position.w;
    position.y = snapped_ndc.y * position.w;

    let fi = fontCalc(text.char, position, vertexId);
    out.fragPosition = fi.fragPosition;
    out.tex_coord = fi.tex_coord;
    return out;
}

@fragment
fn fragmentFontColor(input: TextVertexOutput) -> @location(0) vec4f {
    let v = textureSample(u_font_texture, u_font_sampler, input.tex_coord);
    var dist = max(min(v.x, v.y), min(max(v.x, v.y), v.z));
    if (dist == 0.0) { discard; }

    let width = max(1.0, 4.0 * u_font.size / u_font.font_size);
    let u_in_bias = 0.2 * smoothstep(0.0, 20.0, u_font.size) - 0.2;
    let e = width * (dist - 0.5 - u_in_bias) + 0.5;
    let alpha = smoothstep(0.0, 1.0, e);
    if (alpha < 0.01) { discard; }

    return vec4f(input.color.rgb * alpha, alpha);
}
