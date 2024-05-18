//Vertex shader

struct Camera {
    view_pos : vec4 < f32>,
    view_proj : mat4x4 < f32>,
}
@group(1) @binding(0)
var<uniform> camera : Camera;

struct VertexInput {
    @location(0) position : vec3 < f32>,
}
struct InstanceInput {
    @location(1) model_matrix_0 : vec4 < f32>,
    @location(2) model_matrix_1 : vec4 < f32>,
    @location(3) model_matrix_2 : vec4 < f32>,
    @location(4) model_matrix_3 : vec4 < f32>,
    @location(5) color : vec4 < f32>,
    @location(6) instance_state : f32,
    @location(7) instance_fade_level : f32,
}

struct VertexOutput {
    @builtin(position) clip_position : vec4 < f32>,
    @location(0) color : vec4 < f32>,
}

@vertex
fn vs_main(
model : VertexInput,
instance : InstanceInput,
) -> VertexOutput {
    let model_matrix = mat4x4 < f32 > (
    instance.model_matrix_0,
    instance.model_matrix_1,
    instance.model_matrix_2,
    instance.model_matrix_3,
    );
    var out : VertexOutput;
    let world_position = model_matrix * vec4 < f32 > (model.position, 1.0);
    out.clip_position = camera.view_proj * (camera.view_pos + world_position);
    if instance.instance_state == 0.0 {
        //Dead cell
        out.color = vec4 < f32 > (0.0, 0.0, 0.0, 0.0);
    } else if instance.instance_state == 1.0 {
        //Alive cell
        out.color = instance.color;
    } else {
        //Transitioning cell apply the fade level as alpha
        if instance.instance_fade_level < 0.0 {
            out.color = vec4 < f32 > (0.0, 0.0, 0.0, 0.0);
        } else {
            out.color = vec4 < f32 > (instance.color.r, instance.color.g, instance.color.b, instance.instance_fade_level);
        }
    }
    return out;
}

//Fragment shader
@fragment
fn fs_main(vertex : VertexOutput) -> @location(0) vec4 < f32> {
    return vertex.color;
}

@fragment
fn fs_wire(vertex : VertexOutput) -> @location(0) vec4 < f32> {
    return vec4 < f32 > (0.0, 0.5, 0.0, 0.5);
}
