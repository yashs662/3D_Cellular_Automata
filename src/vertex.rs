use bytemuck::{Pod, Zeroable};
use std::mem;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vertex {
    pub position: [f32; 4],
    pub texture_coord: [f32; 2],
}

unsafe impl Pod for Vertex {}
unsafe impl Zeroable for Vertex {}

impl Vertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            }],
        }
    }

    pub fn new(pos: [f32; 3], tc: [f32; 2]) -> Vertex {
        Vertex {
            position: [pos[0], pos[1], pos[2], 1.0],
            texture_coord: [tc[0], tc[1]],
        }
    }

    /// Create a cube with the given size.
    pub fn create_vertices(cube_size: f32) -> (Vec<Vertex>, Vec<u16>) {
        let vertex_data = [
            // top (0, 0, 1)
            Vertex::new([-cube_size, -cube_size, cube_size], [0.0, 0.0]),
            Vertex::new([cube_size, -cube_size, cube_size], [1.0, 0.0]),
            Vertex::new([cube_size, cube_size, cube_size], [1.0, 1.0]),
            Vertex::new([-cube_size, cube_size, cube_size], [0.0, 1.0]),
            // bottom (0, 0, -1)
            Vertex::new([-cube_size, cube_size, -cube_size], [1.0, 0.0]),
            Vertex::new([cube_size, cube_size, -cube_size], [0.0, 0.0]),
            Vertex::new([cube_size, -cube_size, -cube_size], [0.0, 1.0]),
            Vertex::new([-cube_size, -cube_size, -cube_size], [1.0, 1.0]),
            // right (1, 0, 0)
            Vertex::new([cube_size, -cube_size, -cube_size], [0.0, 0.0]),
            Vertex::new([cube_size, cube_size, -cube_size], [1.0, 0.0]),
            Vertex::new([cube_size, cube_size, cube_size], [1.0, 1.0]),
            Vertex::new([cube_size, -cube_size, cube_size], [0.0, 1.0]),
            // left (-1, 0, 0)
            Vertex::new([-cube_size, -cube_size, cube_size], [1.0, 0.0]),
            Vertex::new([-cube_size, cube_size, cube_size], [0.0, 0.0]),
            Vertex::new([-cube_size, cube_size, -cube_size], [0.0, 1.0]),
            Vertex::new([-cube_size, -cube_size, -cube_size], [1.0, 1.0]),
            // front (0, 1, 0)
            Vertex::new([cube_size, cube_size, -cube_size], [1.0, 0.0]),
            Vertex::new([-cube_size, cube_size, -cube_size], [0.0, 0.0]),
            Vertex::new([-cube_size, cube_size, cube_size], [0.0, 1.0]),
            Vertex::new([cube_size, cube_size, cube_size], [1.0, 1.0]),
            // back (0, -1, 0)
            Vertex::new([cube_size, -cube_size, cube_size], [0.0, 0.0]),
            Vertex::new([-cube_size, -cube_size, cube_size], [1.0, 0.0]),
            Vertex::new([-cube_size, -cube_size, -cube_size], [1.0, 1.0]),
            Vertex::new([cube_size, -cube_size, -cube_size], [0.0, 1.0]),
        ];

        let index_data: &[u16] = &[
            0, 1, 2, 2, 3, 0, // top
            4, 5, 6, 6, 7, 4, // bottom
            8, 9, 10, 10, 11, 8, // right
            12, 13, 14, 14, 15, 12, // left
            16, 17, 18, 18, 19, 16, // front
            20, 21, 22, 22, 23, 20, // back
        ];

        (vertex_data.to_vec(), index_data.to_vec())
    }

    pub fn create_vertices_for_world_grid(size: f32) -> Vec<Vertex> {
        vec![
            Vertex::new([-size, 0.0, -size], [0.0, 0.0]),
            Vertex::new([size, 0.0, -size], [1.0, 0.0]),
            Vertex::new([-size, 0.0, size], [0.0, 1.0]),
            Vertex::new([size, 0.0, -size], [1.0, 0.0]),
            Vertex::new([size, 0.0, size], [1.0, 1.0]),
            Vertex::new([-size, 0.0, size], [0.0, 1.0]),
        ]
    }
}
