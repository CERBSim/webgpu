#import camera

struct LightUniforms {
  direction: vec3f,
  ambient: f32,
  diffuse: f32,
  specular: f32,
  shininess: f32,

  padding0: u32,
};

@group(0) @binding(8) var<uniform> u_light : LightUniforms;

fn lightCalcBrightness(p: vec3f, normal: vec3f) -> vec2f {
    let lightPos4 = u_camera.model_view_projection*vec4f(u_light.direction, 0.0);
    let n = normalize(cameraMapNormal(normal.xyz).xyz);
    let view = -normalize((u_camera.model_view * vec4(p, 1.0)).xyz);
    let r = reflect( -normalize(u_light.direction), n );

    let dim = 1.0;
    var spec = pow( max( dot(r,view) , 0.0 ), u_light.shininess );
    let sDotN = max( dot( normalize(u_light.direction), n ), 0.0 );
    let diffuse = u_light.diffuse * sDotN;
    if(diffuse==0.0 || u_light.shininess ==0.0) {
      spec = 0.0;
    }

    return vec2f(u_light.ambient + diffuse, spec*u_light.specular);
}

fn lightCalcColor(p: vec3f, n: vec3f, color: vec4f) -> vec4f {
    let brightness = lightCalcBrightness(p, n);
    return vec4(color.xyz*brightness.x + brightness.y*vec3(1,1,1), color.w);
}
