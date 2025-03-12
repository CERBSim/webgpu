@group(0) @binding(30) var<storage> u_text : Texts;

struct Texts {
  n_texts: u32,
  data: array<u32>, // check in python code of label.py on how the data is stored
};

struct TextData {
  pos: vec3f,
  shift: vec2f,
  length: u32,
  ichar: u32,
  char: u32,
  apply_camera: u32,
};

fn textLoadData(i: u32) -> TextData {
    let offset = u_text.n_texts * 4 + i * 2;
    let itext = u_text.data[ offset ];
    let char_data = u_text.data[ offset + 1 ];
    let ichar = extractBits(char_data, 0u, 16u);
    let char = extractBits(char_data, 16u, 8u);

    let offset_text = itext * 4;
    let pos = vec3f(bitcast<f32>(u_text.data[offset_text]), bitcast<f32>(u_text.data[offset_text + 1]), bitcast<f32>(u_text.data[offset_text + 2]));
    let text_data = u_text.data[offset_text + 3];
    let length = extractBits(text_data, 0u, 16u);
    let apply_camera = extractBits(text_data, 16u, 8u);

    let x_align = f32(extractBits(text_data, 24u, 2u));
    let y_align = f32(extractBits(text_data, 26u, 2u));

    let shift = vec2<f32>(-0.5 * x_align, -0.5 * y_align);

  return TextData(pos, shift, length, ichar, char, apply_camera);
}

@vertex
fn vertexText(@builtin(vertex_index) vertexId: u32, @builtin(instance_index) charId: u32) -> FontFragmentInput {
    let text = textLoadData(charId);

    var position = vec4f(text.pos, 1.0);
    if (text.apply_camera != 0) {
        position = cameraMapPoint(text.pos);
    }

    let w: f32 = u_font.width_normalized * position.w;
    let h: f32 = u_font.height_normalized * position.w;

    position.x += f32(text.ichar) * w;

    let shift = text.shift;
    position.x += w * shift.x * f32(text.length);
    position.y += h * shift.y;

    return fontCalc(text.char, position, vertexId);
}
