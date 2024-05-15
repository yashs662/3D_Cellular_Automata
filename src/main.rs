use bytemuck::{Pod, Zeroable};
use cellular_automata_3d::{
    camera::{Camera, CameraController, CameraUniform, Projection},
    constants::DEPTH_FORMAT,
    neighbors::NeighborMethod,
    texture::{self, Texture},
};
use cgmath::{EuclideanSpace, InnerSpace, Rotation3};
use std::{
    borrow::Cow,
    f32::consts,
    fmt::{Display, Formatter},
    mem,
};
use wgpu::{util::DeviceExt, PipelineCompilationOptions};
use winit::{
    event::{DeviceEvent, ElementState, KeyEvent, MouseButton, WindowEvent},
    keyboard::{Key, NamedKey},
};

pub struct UpdateQueue {
    queue: Vec<UpdateEnum>,
}

impl UpdateQueue {
    fn new() -> Self {
        UpdateQueue { queue: Vec::new() }
    }

    fn add(&mut self, update: UpdateEnum) {
        self.queue.push(update);
    }

    fn reset(&mut self) {
        self.queue.clear();
    }
}

enum UpdateEnum {
    Transparency,
    NumInstancesIncreased,
    NumInstancesDecreased,
    SpaceBetweenInstances,
    SortInstances,
}

pub struct Settings {
    wireframe_overlay: bool,
    num_instances_per_row: u32,
    cube_size: f32,
    space_between_instances: f32,
    transparency: f32,
    num_instances_step_size: u32,
    transparency_step_size: f32,
    space_between_step_size: f32,
    angle_threshold_for_sort: f32,
    fade_time: f32,
    simulation_tick_rate: u16,
    last_simulation_tick: std::time::Instant,
    neighbor_method: NeighborMethod,
}

impl Default for Settings {
    fn default() -> Self {
        Self::new()
    }
}

impl Settings {
    pub fn new() -> Self {
        let wireframe_overlay = false;
        let num_instances_per_row = 10;
        let cube_size = 1.0;
        let space_between_instances = (cube_size * 2.0) + 0.1;

        Settings {
            wireframe_overlay,
            num_instances_per_row,
            cube_size,
            space_between_instances,
            transparency: 0.2,
            num_instances_step_size: 2,
            transparency_step_size: 0.1,
            space_between_step_size: 0.05,
            angle_threshold_for_sort: 0.10,
            fade_time: 2.0,
            simulation_tick_rate: 100,
            last_simulation_tick: std::time::Instant::now(),
            neighbor_method: NeighborMethod::Moore,
        }
    }

    pub fn toggle_wireframe_overlay(&mut self) {
        self.wireframe_overlay = !self.wireframe_overlay;
        log::info!("Wireframe overlay set to: {}", self.wireframe_overlay);
    }

    pub fn set_num_instances_per_row(
        &mut self,
        num_instances_per_row: u32,
        update_queue: &mut UpdateQueue,
    ) {
        if num_instances_per_row < 2 {
            log::error!("Number of instances per row cannot be less than 2");
            self.num_instances_per_row = 2;
        } else if num_instances_per_row > 100 {
            log::error!("Number of instances per row cannot be more than 100");
            self.num_instances_per_row = 100;
        } else {
            match num_instances_per_row.cmp(&self.num_instances_per_row) {
                std::cmp::Ordering::Less => {
                    update_queue.add(UpdateEnum::NumInstancesDecreased);
                }
                std::cmp::Ordering::Greater => {
                    update_queue.add(UpdateEnum::NumInstancesIncreased);
                }
                std::cmp::Ordering::Equal => {
                    // No change
                }
            }
            self.num_instances_per_row = num_instances_per_row;
            log::info!(
                "Number of instances per row set to: {}",
                self.num_instances_per_row
            );
        }
    }

    pub fn set_transparency(&mut self, transparency: f32, update_queue: &mut UpdateQueue) {
        if transparency < 0.0 {
            log::error!("Transparency cannot be less than 0.0");
            self.transparency = 0.0;
        } else if transparency > 1.0 {
            log::error!("Transparency cannot be more than 1.0");
            self.transparency = 1.0;
        } else {
            self.transparency = (transparency * 10.0).round() / 10.0;
        }
        log::info!("Transparency set to: {}", self.transparency);
        update_queue.add(UpdateEnum::Transparency);
    }

    pub fn set_space_between_instances(
        &mut self,
        space_between_instances: f32,
        update_queue: &mut UpdateQueue,
    ) {
        if space_between_instances < (self.cube_size * 2.0) {
            log::error!("Space between instances cannot be less than 0.0");
            // add a very small value to avoid overlapping instances
            self.space_between_instances = (self.cube_size * 2.0) + 0.001;
        } else if space_between_instances > ((self.cube_size * 2.0) + 10.0) {
            log::error!("Space between instances cannot be more than 10.0");
            self.space_between_instances = (self.cube_size * 2.0) + 10.0;
        } else {
            self.space_between_instances = (space_between_instances * 100.0).round() / 100.0;
        }
        log::info!(
            "Space between instances set to: {}",
            ((self.space_between_instances - (self.cube_size * 2.0)) * 100.0).round() / 100.0
        );
        update_queue.add(UpdateEnum::SpaceBetweenInstances);
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    position: [f32; 4],
    texture_coord: [f32; 2],
}

unsafe impl Pod for Vertex {}
unsafe impl Zeroable for Vertex {}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
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

    fn new(pos: [f32; 3], tc: [f32; 2]) -> Vertex {
        Vertex {
            position: [pos[0], pos[1], pos[2], 1.0],
            texture_coord: [tc[0], tc[1]],
        }
    }
}

#[derive(Clone, Copy, Default, Debug, PartialEq)]
enum CellStateEnum {
    #[default]
    Alive,
    Fading,
    Dead,
}

impl CellStateEnum {
    fn to_int(self) -> u8 {
        match self {
            CellStateEnum::Alive => 1,
            CellStateEnum::Fading => 2,
            CellStateEnum::Dead => 0,
        }
    }
}

impl Display for CellStateEnum {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            CellStateEnum::Alive => write!(f, "Alive"),
            CellStateEnum::Fading => write!(f, "Fading"),
            CellStateEnum::Dead => write!(f, "Dead"),
        }
    }
}

#[derive(Clone, Copy, Default, Debug, PartialEq)]
struct CellState {
    state: CellStateEnum,
    killed_at: Option<std::time::Instant>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Instance {
    position: cgmath::Vector3<f32>,
    rotation: cgmath::Quaternion<f32>,
    color: cgmath::Vector4<f32>,
    instance_state: CellState,
}

impl Instance {
    fn to_raw(self, fade_time: f32, current_transparency_setting: f32) -> InstanceRaw {
        let transform =
            cgmath::Matrix4::from_translation(self.position) * cgmath::Matrix4::from(self.rotation);
        InstanceRaw {
            transform: transform.into(),
            color: self.color.into(),
            instance_state: self.instance_state.state.to_int(),
            instance_death_transparency: self.instance_state.killed_at.map_or(
                current_transparency_setting,
                |t| {
                    ((fade_time - t.elapsed().as_secs_f32()) / fade_time)
                        * current_transparency_setting
                },
            ),
        }
    }

    fn create_instance_at_pos(settings: &Settings, x: f32, y: f32, z: f32) -> Instance {
        let position = cgmath::Vector3 { x, y, z };

        let rotation =
            cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0));

        // random color
        // let color = cgmath::Vector4 {
        //     x: rand::random::<f32>(),
        //     y: rand::random::<f32>(),
        //     z: rand::random::<f32>(),
        //     w: settings.transparency,
        // };

        // give a color based on position
        let color = cgmath::Vector4 {
            x: (x + settings.space_between_instances)
                / (settings.space_between_instances * settings.num_instances_per_row as f32),
            y: (y + settings.space_between_instances)
                / (settings.space_between_instances * settings.num_instances_per_row as f32),
            z: (z + settings.space_between_instances)
                / (settings.space_between_instances * settings.num_instances_per_row as f32),
            w: settings.transparency,
        };

        Instance {
            position,
            rotation,
            color,
            instance_state: CellState::default(),
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct InstanceRaw {
    transform: [[f32; 4]; 4],
    color: [f32; 4],
    instance_state: u8,
    instance_death_transparency: f32,
}

unsafe impl Pod for InstanceRaw {}
unsafe impl Zeroable for InstanceRaw {}

impl InstanceRaw {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 2 * mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 3 * mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 4 * mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 5 * mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32,
                },
                wgpu::VertexAttribute {
                    offset: ((5 * mem::size_of::<[f32; 4]>()) + mem::size_of::<f32>())
                        as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

struct InstanceManager {
    instances: Vec<Vec<Vec<Instance>>>,
    flattened: Vec<Instance>,
    sort_map: Vec<usize>,
}

impl InstanceManager {
    fn prepare_initial_instances(settings: &Settings) -> InstanceManager {
        let instances: Vec<Vec<Vec<Instance>>> = (0..settings.num_instances_per_row)
            .map(|x| {
                (0..settings.num_instances_per_row)
                    .map(|y| {
                        (0..settings.num_instances_per_row)
                            .map(|z| {
                                let x = settings.space_between_instances * x as f32;
                                let y = settings.space_between_instances * y as f32;
                                let z = settings.space_between_instances * z as f32;
                                Instance::create_instance_at_pos(settings, x, y, z)
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let flattened = instances.iter().flatten().flatten().cloned().collect();
        InstanceManager {
            instances,
            flattened,
            sort_map: Vec::new(),
        }
    }

    fn prepare_raw_instance_data(&self, settings: &Settings) -> Vec<InstanceRaw> {
        self.flattened
            .iter()
            .map(|instance| Instance::to_raw(*instance, settings.fade_time, settings.transparency))
            .collect::<Vec<_>>()
    }

    fn create_new_buffer(&self, settings: &Settings, device: &wgpu::Device) -> wgpu::Buffer {
        let instance_data = self.prepare_raw_instance_data(settings);
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        })
    }

    fn sort_by_distance_to_camera(&mut self, camera: &Camera) {
        let start = std::time::Instant::now();
        let camera_position = camera.position.to_vec();
        self.sort_map = (0..self.flattened.len()).collect();
        self.sort_map.sort_by(|&i, &j| {
            let a_dist = (self.flattened[i].position - camera_position).magnitude();
            let b_dist = (self.flattened[j].position - camera_position).magnitude();
            b_dist.partial_cmp(&a_dist).unwrap()
        });
        self.flattened = self.sort_map.iter().map(|&i| self.flattened[i]).collect();
        log::info!("Time to sort: {:?}", start.elapsed());
    }

    fn flatten(&mut self) {
        self.flattened = self.instances.iter().flatten().flatten().cloned().collect();
    }

    fn simulate(
        &mut self,
        settings: &Settings,
        queue: &wgpu::Queue,
        instance_buffer: &wgpu::Buffer,
    ) {
        let start = std::time::Instant::now();
        if self.sort_map.is_empty() {
            log::error!("Sort map is empty");
            return;
        }
        for (z, layer) in self.instances.iter_mut().enumerate() {
            for (y, row) in layer.iter_mut().enumerate() {
                for (x, _) in row.iter_mut().enumerate() {
                    let index = (z * settings.num_instances_per_row.pow(2) as usize)
                        + (y * settings.num_instances_per_row as usize)
                        + x;
                    let sorted_index = self.sort_map[index];
                    let flattened_copy = self.flattened.clone();
                    let instance = &mut self.flattened[sorted_index];
                    let mut alive_neighbors = 0;
                    let neighbors = settings.neighbor_method.get_neighbor_iter();
                    for (dx, dy, dz) in neighbors {
                        let x = x as i32 + dx;
                        let y = y as i32 + dy;
                        let z = z as i32 + dz;
                        if x < 0
                            || y < 0
                            || z < 0
                            || x >= settings.num_instances_per_row as i32
                            || y >= settings.num_instances_per_row as i32
                            || z >= settings.num_instances_per_row as i32
                        {
                            continue;
                        }
                        let index = (z * settings.num_instances_per_row.pow(2) as i32)
                            + (y * settings.num_instances_per_row as i32)
                            + x;
                        let neighbor = &flattened_copy[index as usize];
                        if neighbor.instance_state.state == CellStateEnum::Alive {
                            alive_neighbors += 1;
                        }
                    }
                    match instance.instance_state.state {
                        CellStateEnum::Alive => {
                            if alive_neighbors < 2 || alive_neighbors > 7 {
                                instance.instance_state.state = CellStateEnum::Fading;
                                instance.instance_state.killed_at = Some(std::time::Instant::now());
                            }
                        }
                        CellStateEnum::Fading => {
                            if alive_neighbors == 3
                                || alive_neighbors == 4
                                || alive_neighbors == 5
                                || alive_neighbors == 6
                            {
                                instance.instance_state.state = CellStateEnum::Alive;
                                instance.instance_state.killed_at = None;
                            } else if instance
                                .instance_state
                                .killed_at
                                .map_or(false, |t| t.elapsed().as_secs_f32() >= settings.fade_time)
                            {
                                instance.instance_state.state = CellStateEnum::Dead;
                            }
                        }
                        CellStateEnum::Dead => {
                            if alive_neighbors == 3
                                || alive_neighbors == 4
                                || alive_neighbors == 5
                                || alive_neighbors == 6
                            {
                                instance.instance_state.state = CellStateEnum::Alive;
                                instance.instance_state.killed_at = None;
                            }
                        }
                    }
                }
            }
        }
        self.update_buffer(settings, queue, instance_buffer);
        log::info!("Time to simulate: {:?}", start.elapsed());
    }

    fn update_buffer(
        &mut self,
        settings: &Settings,
        queue: &wgpu::Queue,
        instance_buffer: &wgpu::Buffer,
    ) {
        // update the instance buffer
        let instance_data = self.prepare_raw_instance_data(settings);
        queue.write_buffer(instance_buffer, 0, bytemuck::cast_slice(&instance_data));
    }

    fn update_transparency(
        &mut self,
        settings: &Settings,
        queue: &wgpu::Queue,
        instance_buffer: &wgpu::Buffer,
    ) {
        // go throughout the 3d instances and update the transparency then flatten the instances and update the buffer
        for z in 0..settings.num_instances_per_row {
            for y in 0..settings.num_instances_per_row {
                for x in 0..settings.num_instances_per_row {
                    let index = (z as usize * settings.num_instances_per_row.pow(2) as usize)
                        + (y as usize * settings.num_instances_per_row as usize)
                        + x as usize;
                    let sorted_index = self.sort_map[index];
                    let instance = &mut self.flattened[sorted_index];
                    instance.color.w = settings.transparency;
                    let instance = &mut self.instances[z as usize][y as usize][x as usize];
                    instance.color.w = settings.transparency;
                }
            }
        }
        self.update_buffer(settings, queue, instance_buffer)
    }

    fn update_space_between_instances(
        &mut self,
        settings: &Settings,
        queue: &wgpu::Queue,
        instance_buffer: &wgpu::Buffer,
    ) {
        // go throughout the 3d instances and update the position then flatten the instances and update the buffer
        for z in 0..settings.num_instances_per_row {
            for y in 0..settings.num_instances_per_row {
                for x in 0..settings.num_instances_per_row {
                    let index = (z as usize * settings.num_instances_per_row.pow(2) as usize)
                        + (y as usize * settings.num_instances_per_row as usize)
                        + x as usize;
                    let sorted_index = self.sort_map[index];
                    let instance = &mut self.flattened[sorted_index];
                    let ix = settings.space_between_instances
                        * (instance.position.x / settings.space_between_instances).round();
                    let iy = settings.space_between_instances
                        * (instance.position.y / settings.space_between_instances).round();
                    let iz = settings.space_between_instances
                        * (instance.position.z / settings.space_between_instances).round();
                    instance.position = cgmath::Vector3 {
                        x: ix,
                        y: iy,
                        z: iz,
                    };
                    let instance = &mut self.instances[z as usize][y as usize][x as usize];
                    instance.position = cgmath::Vector3 {
                        x: ix,
                        y: iy,
                        z: iz,
                    };
                }
            }
        }
        self.update_buffer(settings, queue, instance_buffer)
    }

    fn increase_num_instances_per_row(
        &mut self,
        settings: &Settings,
        device: &wgpu::Device,
        camera: &Camera,
    ) -> wgpu::Buffer {
        let mut new_instances: Vec<Vec<Vec<Instance>>> = Vec::new();

        for x in 0..settings.num_instances_per_row {
            let mut y_instances: Vec<Vec<Instance>> = Vec::new();
            for y in 0..settings.num_instances_per_row {
                let mut z_instances: Vec<Instance> = Vec::new();
                for z in 0..settings.num_instances_per_row {
                    let instance = if x == 0
                        || y == 0
                        || z == 0
                        || x == settings.num_instances_per_row - 1
                        || y == settings.num_instances_per_row - 1
                        || z == settings.num_instances_per_row - 1
                    {
                        // Create a new instance
                        let x = settings.space_between_instances * x as f32;
                        let y = settings.space_between_instances * y as f32;
                        let z = settings.space_between_instances * z as f32;
                        Instance::create_instance_at_pos(settings, x, y, z)
                    } else {
                        let mut cached_instance = self.instances[x.saturating_sub(1) as usize]
                            [y.saturating_sub(1) as usize]
                            [z.saturating_sub(1) as usize];
                        cached_instance.position.x += settings.space_between_instances;
                        cached_instance.position.y += settings.space_between_instances;
                        cached_instance.position.z += settings.space_between_instances;
                        cached_instance
                    };
                    z_instances.push(instance);
                }
                y_instances.push(z_instances);
            }
            new_instances.push(y_instances);
        }
        self.instances = new_instances;
        self.flatten();
        self.sort_by_distance_to_camera(camera);
        self.create_new_buffer(settings, device)
    }

    fn decrease_num_instances_per_row(
        &mut self,
        settings: &Settings,
        device: &wgpu::Device,
        camera: &Camera,
    ) -> wgpu::Buffer {
        let mut new_instances: Vec<Vec<Vec<Instance>>> = Vec::new();

        // remove the outer layer of instances and keep the cube inside eg initially 12x12x12 then 10x10x10 remove 1 layer from each side
        for x in 1..settings.num_instances_per_row + 1 {
            let mut y_instances: Vec<Vec<Instance>> = Vec::new();
            for y in 1..settings.num_instances_per_row + 1 {
                let mut z_instances: Vec<Instance> = Vec::new();
                for z in 1..settings.num_instances_per_row + 1 {
                    let mut instance = self.instances[x as usize][y as usize][z as usize];
                    instance.position.x -= settings.space_between_instances;
                    instance.position.y -= settings.space_between_instances;
                    instance.position.z -= settings.space_between_instances;
                    z_instances.push(instance);
                }
                y_instances.push(z_instances);
            }
            new_instances.push(y_instances);
        }
        self.instances = new_instances;
        self.flatten();
        self.sort_by_distance_to_camera(camera);
        self.create_new_buffer(settings, device)
    }
}

fn create_vertices(cube_size: f32) -> (Vec<Vertex>, Vec<u16>) {
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

struct App {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: usize,
    bind_group: wgpu::BindGroup,
    uniform_buf: wgpu::Buffer,
    pipeline: wgpu::RenderPipeline,
    pipeline_wire: Option<wgpu::RenderPipeline>,
    instance_manager: InstanceManager,
    instance_buffer: wgpu::Buffer,
    num_indices: u32,
    camera: Camera,
    last_sort_camera: Camera,
    projection: Projection,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    mouse_pressed: bool,
    settings: Settings,
    depth_texture: Texture,
    update_queue: UpdateQueue,
}

impl App {
    fn generate_matrix(aspect_ratio: f32) -> glam::Mat4 {
        let projection = glam::Mat4::perspective_rh(consts::FRAC_PI_4, aspect_ratio, 1.0, 10.0);
        let view = glam::Mat4::look_at_rh(
            glam::Vec3::new(1.5f32, -5.0, 3.0),
            glam::Vec3::ZERO,
            glam::Vec3::Z,
        );
        projection * view
    }

    fn update_camera(&mut self, dt: f32, queue: &wgpu::Queue) {
        self.camera_controller.update_camera(&mut self.camera, dt);
        self.camera_uniform
            .update_view_proj(&self.camera, &self.projection);
        queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
    }

    fn transparency_update(&mut self) {
        let yaw_diff = (self.camera.yaw - self.last_sort_camera.yaw).0.abs();
        let pitch_diff = (self.camera.pitch - self.last_sort_camera.pitch).0.abs();
        if yaw_diff > self.settings.angle_threshold_for_sort
            || pitch_diff > self.settings.angle_threshold_for_sort
        {
            self.update_queue.add(UpdateEnum::SortInstances);
        }
    }

    fn simulation_update(&mut self, queue: &wgpu::Queue) {
        if self.settings.last_simulation_tick.elapsed().as_millis()
            >= self.settings.simulation_tick_rate.into()
        {
            let start = std::time::Instant::now();
            self.instance_manager
                .simulate(&self.settings, queue, &self.instance_buffer);
            self.settings.last_simulation_tick = std::time::Instant::now();
            log::debug!("Time to simulate: {:?}", start.elapsed());
        }
    }

    fn queue_update(&mut self, gpu_queue: &wgpu::Queue, device: &wgpu::Device) {
        for update in self.update_queue.queue.iter() {
            match update {
                UpdateEnum::Transparency => {
                    self.instance_manager.update_transparency(
                        &self.settings,
                        gpu_queue,
                        &self.instance_buffer,
                    );
                }
                UpdateEnum::SpaceBetweenInstances => {
                    self.instance_manager.update_space_between_instances(
                        &self.settings,
                        gpu_queue,
                        &self.instance_buffer,
                    );
                }
                UpdateEnum::SortInstances => {
                    self.instance_manager
                        .sort_by_distance_to_camera(&self.camera);
                    self.last_sort_camera = self.camera;
                    self.instance_manager.update_buffer(
                        &self.settings,
                        gpu_queue,
                        &self.instance_buffer,
                    );
                }
                UpdateEnum::NumInstancesIncreased => {
                    self.instance_buffer = self.instance_manager.increase_num_instances_per_row(
                        &self.settings,
                        device,
                        &self.camera,
                    );
                }
                UpdateEnum::NumInstancesDecreased => {
                    self.instance_buffer = self.instance_manager.decrease_num_instances_per_row(
                        &self.settings,
                        device,
                        &self.camera,
                    );
                }
            }
        }
        self.update_queue.reset();
    }
}

impl cellular_automata_3d::framework::App for App {
    fn optional_features() -> wgpu::Features {
        wgpu::Features::POLYGON_MODE_LINE
    }

    fn init(
        config: &wgpu::SurfaceConfiguration,
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
    ) -> Self {
        let settings = Settings::new();
        // Create the vertex and index buffers
        let (vertex_data, index_data) = create_vertices(settings.cube_size);

        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertex_data),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&index_data),
            usage: wgpu::BufferUsages::INDEX,
        });

        let mut instance_manager = InstanceManager::prepare_initial_instances(&settings);

        let (
            camera,
            projection,
            camera_controller,
            camera_uniform,
            camera_buffer,
            camera_bind_group_layout,
            camera_bind_group,
        ) = setup_camera(config, device);

        instance_manager.sort_by_distance_to_camera(&camera);
        let instance_buffer = instance_manager.create_new_buffer(&settings, device);

        // Create pipeline layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Render Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(64),
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout, &camera_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create other resources
        let mx_total = Self::generate_matrix(config.width as f32 / config.height as f32);
        let mx_ref: &[f32; 16] = mx_total.as_ref();
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(mx_ref),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let depth_texture = texture::Texture::create_depth_texture(device, config, "depth_texture");

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
            label: None,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        let vertex_buffers = &[Vertex::desc(), InstanceRaw::desc()];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: vertex_buffers,
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                // Setting this to anything other than Fill requires Features::POLYGON_MODE_LINE
                // or Features::POLYGON_MODE_POINT
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less, // 1.
                stencil: wgpu::StencilState::default(),     // 2.
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            // If the pipeline will be used with a multiview render pass, this
            // indicates how many array layers the attachments will have.
            multiview: None,
        });

        let pipeline_wire = if device
            .features()
            .contains(wgpu::Features::POLYGON_MODE_LINE)
        {
            let pipeline_wire = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: vertex_buffers,
                    compilation_options: PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_wire",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.view_formats[0],
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                operation: wgpu::BlendOperation::Add,
                                src_factor: wgpu::BlendFactor::SrcAlpha,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            },
                            alpha: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    polygon_mode: wgpu::PolygonMode::Line,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less, // 1.
                    stencil: wgpu::StencilState::default(),     // 2.
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });
            Some(pipeline_wire)
        } else {
            None
        };

        // Done
        App {
            vertex_buf,
            index_buf,
            index_count: index_data.len(),
            bind_group,
            uniform_buf,
            pipeline,
            pipeline_wire,
            instance_manager,
            instance_buffer,
            num_indices: index_data.len() as u32,
            camera,
            last_sort_camera: camera,
            projection,
            camera_controller,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            mouse_pressed: false,
            settings,
            depth_texture,
            update_queue: UpdateQueue::new(),
        }
    }

    fn update_window_event(&mut self, event: winit::event::WindowEvent) {
        match event {
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    logical_key, state, ..
                },
                ..
            } => {
                self.camera_controller
                    .process_keyboard(logical_key.clone(), state);
                if state == ElementState::Pressed {
                    if let Key::Character(s) = logical_key {
                        if s.as_str() == "p" {
                            self.settings.toggle_wireframe_overlay();
                        } else if s.as_str() == "n" {
                            self.settings.set_transparency(
                                self.settings.transparency - self.settings.transparency_step_size,
                                &mut self.update_queue,
                            );
                        } else if s.as_str() == "m" {
                            self.settings.set_transparency(
                                self.settings.transparency + self.settings.transparency_step_size,
                                &mut self.update_queue,
                            );
                        } else if s.as_str() == "k" {
                            self.settings.set_space_between_instances(
                                self.settings.space_between_instances
                                    - self.settings.space_between_step_size,
                                &mut self.update_queue,
                            );
                        } else if s.as_str() == "l" {
                            self.settings.set_space_between_instances(
                                self.settings.space_between_instances
                                    + self.settings.space_between_step_size,
                                &mut self.update_queue,
                            );
                        }
                    } else if let Key::Named(key) = logical_key {
                        if key == NamedKey::PageUp {
                            self.settings.set_num_instances_per_row(
                                self.settings.num_instances_per_row
                                    + self.settings.num_instances_step_size,
                                &mut self.update_queue,
                            );
                        } else if key == NamedKey::PageDown {
                            self.settings.set_num_instances_per_row(
                                self.settings
                                    .num_instances_per_row
                                    .saturating_sub(self.settings.num_instances_step_size),
                                &mut self.update_queue,
                            );
                        }
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.camera_controller.process_scroll(&delta);
            }
            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state,
                ..
            } => {
                self.mouse_pressed = state == ElementState::Pressed;
            }
            _ => {}
        }
    }

    fn update_device_event(&mut self, event: winit::event::DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            if self.mouse_pressed {
                self.camera_controller.process_mouse(delta.0, delta.1)
            }
        }
    }

    fn update(&mut self, dt: f32, queue: &wgpu::Queue, device: &wgpu::Device) {
        self.update_camera(dt, queue);
        self.transparency_update();
        self.queue_update(queue, device);
        self.simulation_update(queue);
    }

    fn resize(
        &mut self,
        config: &wgpu::SurfaceConfiguration,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let mx_total = Self::generate_matrix(config.width as f32 / config.height as f32);
        let mx_ref: &[f32; 16] = mx_total.as_ref();
        self.projection.resize(config.width, config.height);
        self.depth_texture =
            texture::Texture::create_depth_texture(device, config, "depth_texture");
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::cast_slice(mx_ref));
    }

    fn render(&mut self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            render_pass.push_debug_group("Prepare data for draw.");
            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(
                0..self.num_indices,
                0,
                0..self.instance_manager.flattened.len() as _,
            );
            render_pass.pop_debug_group();
            render_pass.insert_debug_marker("Draw!");
            if let Some(ref pipe) = self.pipeline_wire {
                if self.settings.wireframe_overlay {
                    render_pass.set_pipeline(pipe);
                    render_pass.draw_indexed(
                        0..self.index_count as u32,
                        0,
                        0..self.instance_manager.flattened.len() as _,
                    );
                }
            }
        }

        queue.submit(Some(encoder.finish()));
    }
}

fn setup_camera(
    config: &wgpu::SurfaceConfiguration,
    device: &wgpu::Device,
) -> (
    Camera,
    Projection,
    CameraController,
    CameraUniform,
    wgpu::Buffer,
    wgpu::BindGroupLayout,
    wgpu::BindGroup,
) {
    let camera = Camera::new((10.0, 20.0, 40.0), cgmath::Deg(-90.0), cgmath::Deg(-30.0));
    let projection = Projection::new(config.width, config.height, cgmath::Deg(45.0), 0.1, 100.0);
    let camera_controller = CameraController::new(10.0, 0.6);

    let mut camera_uniform = CameraUniform::new();
    camera_uniform.update_view_proj(&camera, &projection);
    let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Camera Buffer"),
        contents: bytemuck::cast_slice(&[camera_uniform]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let camera_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("camera_and_time_bind_group_layout"),
        });

    let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &camera_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: camera_buffer.as_entire_binding(),
        }],
        label: Some("camera_and_time_bind_group"),
    });
    (
        camera,
        projection,
        camera_controller,
        camera_uniform,
        camera_buffer,
        camera_bind_group_layout,
        camera_bind_group,
    )
}

pub fn main() {
    cellular_automata_3d::framework::run::<App>("cube");
}
