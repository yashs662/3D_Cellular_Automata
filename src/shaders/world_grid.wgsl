struct Camera {
    view_pos : vec4 < f32>,
    view_proj : mat4x4 < f32>,
}

const GRID_SIZE : f32 = 0.12;
const GRID_THICCNESS : f32 = 0.1;

@group(1) @binding(0)
var<uniform> camera : Camera;

struct VertexInput {
    @location(0) position : vec4 < f32>,
    @location(1) texture_coords : vec2 < f32>,
    @location(2) fade_distance : f32,
}

struct VertexOutput {
    @builtin(position) clip_position : vec4 < f32>,
    @location(0) uv : vec2 < f32>,
    @location(1) fade_distance : f32,
    @location(2) camera_pos : vec4 < f32>,
    @location(3) world_pos : vec4 < f32>,
    @location(4) alpha : f32,
}

@vertex
fn vs_main(model : VertexInput) -> VertexOutput {
    var out : VertexOutput;
    out.clip_position = camera.view_proj * (camera.view_pos + model.position);
    out.fade_distance = model.fade_distance;
    out.camera_pos = camera.view_pos;
    //Generate UV coordinates based on object space position
    out.uv = model.position.xz;
    let distance_from_camera = length(out.camera_pos.xz - model.position.xz) / 1000;
    // Fade out grid behind the center of the screen
    out.alpha = 1 - smoothstep(0.1, 0.3, distance_from_camera);
    return out;
}

@fragment
fn fs_main(vertex : VertexOutput) -> @location(0) vec4 < f32> {
    //Calculate the distance from the world center
    let distance_from_center = length(vertex.uv - vec2 < f32 > (0.0, 0.0)) / 30.0;
    let distance_from_camera = length(vertex.camera_pos.xz - vertex.world_pos.xz) / 200.0;

    //Calculate the grid pattern
    let grid = fract(GRID_SIZE * vertex.uv / distance_from_camera);

    //Calculate the gradient effect on both sides
    let mask_x = smoothstep(0.0, GRID_THICCNESS, abs(grid.x - 0.5) * 2.0) * smoothstep(0.0, GRID_THICCNESS, abs(grid.x + 0.5) * 2.0);
    let mask_y = smoothstep(0.0, GRID_THICCNESS, abs(grid.y - 0.5) * 2.0) * smoothstep(0.0, GRID_THICCNESS, abs(grid.y + 0.5) * 2.0);
    var mask = 1.0 - min(mask_x, mask_y);

    // Circular fade effect
    let alpha = 1.0 - smoothstep(0.0, 1.0, distance_from_center / (vertex.fade_distance * 6.0));

    //Render the grid pattern with adjusted transparency and anti-aliasing
    return vec4 < f32 > (mask, mask, mask, mask * alpha * distance_from_center * vertex.alpha);
}
