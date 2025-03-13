// general uniform structures
struct ClippingUniforms {
  plane: vec4<f32>,
  sphere: vec4<f32>,
  mode: u32, // 0: disabled, 1: plane, 2: sphere, 3: both

  padding0: u32,
  padding1: u32,
  padding2: u32,
};

@group(0) @binding(1) var<uniform> u_clipping : ClippingUniforms;

fn calcClipping(p: vec3<f32>) -> bool {
    var result: bool = true;
    if( (u_clipping.mode & 0x01u) == 1u) {
        if dot(u_clipping.plane, vec4<f32>(p, 1.0)) > 0. {
            result = false;
        }
    }
    if ((u_clipping.mode & 0x02u) == 0x02u){
        let d = distance(p, u_clipping.sphere.xyz);
        if d > u_clipping.sphere.w {
            result = false;
        }
    }
    return result;
}

fn checkClipping(p: vec3<f32>) {
    if calcClipping(p) == false {
    discard;
    }
}

