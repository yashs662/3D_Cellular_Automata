//implement a world grid for easy navigation, it should fade towards the edges and will be drawn before anything else
struct Camera {
    view_pos : vec4 < f32>,
    view_proj : mat4x4 < f32>,
}

const PI : f32 = 3.1415926;
const GRID_SIZE : f32 = 0.5;
const GRID_THICCNESS : f32 = 0.01;

@group(1) @binding(0)
var<uniform> camera : Camera;

struct VertexInput {
    @location(0) position : vec3 < f32>,
}

struct VertexOutput {
    @builtin(position) clip_position : vec4 < f32>,
    @location(0) uv : vec2 < f32>,
}

@vertex
fn vs_main(model : VertexInput) -> VertexOutput {
    var out : VertexOutput;
    let world_position = vec4<f32>(model.position, 1.0);
    out.clip_position = camera.view_proj * (camera.view_pos + world_position);

    // Generate UV coordinates based on object space position
    out.uv = model.position.xz + 1.0;

    return out;
}

@fragment
fn fs_main(vertex : VertexOutput) -> @location(0) vec4<f32> {
    // Calculate the grid pattern
    let grid = fract(GRID_SIZE * vertex.uv) - 0.5;

    // Calculate the distance from the world center
    let distance = length(vertex.uv - vec2<f32>(1.0, 1.0)) / 100;

    // Adjust the anti-aliasing amount based on the distance
    let adjusted_aa = smoothstep(-0.2, 2.0, distance / 2);

    let mask_x = smoothstep(GRID_THICCNESS, GRID_THICCNESS + adjusted_aa, abs(grid.x));
    let mask_y = smoothstep(GRID_THICCNESS, GRID_THICCNESS + adjusted_aa, abs(grid.y));
    let mask = 1.0 - min(mask_x, mask_y);

    // Adjust the alpha value based on the distance
    let alpha = 1.0 - smoothstep(0.0, 1.0, distance);

    // Render the grid pattern with adjusted transparency and anti-aliasing
    return vec4<f32>(mask, mask, mask, mask * alpha);
}
