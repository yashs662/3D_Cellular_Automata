use crate::{
    camera::Camera,
    framework::Settings,
    simulation::{CellState, CellStateEnum, ColorMethod, SimulationState},
    utils::{Color, UpdateEnum, UpdateQueue},
    vertex::Vertex,
};
use bytemuck::{Pod, Zeroable};
use cgmath::EuclideanSpace;
use cgmath::{InnerSpace, Rotation3};
use rand::{rngs::ThreadRng, Rng};
use std::mem;
use std::time::Duration;
use wgpu::util::DeviceExt;

#[cfg(feature = "multithreading")]
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Instance {
    position: cgmath::Vector3<f32>,
    rotation: cgmath::Quaternion<f32>,
    color: cgmath::Vector4<f32>,
    fade_to_color: cgmath::Vector4<f32>,
    instance_state: CellState,
}

impl Instance {
    pub fn to_raw(self) -> InstanceRaw {
        let transform =
            cgmath::Matrix4::from_translation(self.position) * cgmath::Matrix4::from(self.rotation);
        InstanceRaw {
            transform: transform.into(),
            color: self.color.into(),
            instance_state: self.instance_state.state.to_int() as f32,
        }
    }

    pub fn create_bounding_box(settings: &Settings) -> Instance {
        Instance::create_instance_at_pos(
            settings,
            -settings.space_between_instances,
            -settings.space_between_instances,
            -settings.space_between_instances,
            CellState::default(),
        )
    }

    pub fn calculate_bounding_box_size(settings: &Settings) -> f32 {
        (settings.cube_size as f64
            + ((settings.space_between_instances as f64 + (settings.cube_size * 2.0) as f64)
                * (settings.domain_size as f64 / 2.0))) as f32
    }

    pub fn scale_x_y_z(x: u32, y: u32, z: u32, settings: &Settings) -> (f32, f32, f32) {
        let total_domain_size =
            Self::calculate_bounding_box_size(settings) - (settings.cube_size * 2.0);
        let x_scaled = x as f32 * (settings.cube_size * 2.0 + settings.space_between_instances)
            - total_domain_size;
        let y_scaled = y as f32 * (settings.cube_size * 2.0 + settings.space_between_instances)
            - total_domain_size;
        let z_scaled = z as f32 * (settings.cube_size * 2.0 + settings.space_between_instances)
            - total_domain_size;
        (x_scaled, y_scaled, z_scaled)
    }

    fn create_instance_at_pos(
        settings: &Settings,
        x: f32,
        y: f32,
        z: f32,
        instance_state: CellState,
    ) -> Instance {
        let position = cgmath::Vector3 { x, y, z };

        let rotation =
            cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0));
        match settings.color_method {
            ColorMethod::Single(color) => Instance {
                position,
                rotation,
                color,
                fade_to_color: color,
                instance_state,
            },
            ColorMethod::StateLerp(color_1, color_2) => Instance {
                position,
                rotation,
                color: color_1,
                fade_to_color: color_2,
                instance_state,
            },
            ColorMethod::DistToCenter(color_1, color_2) => {
                let lerp_amount = (position.magnitude() / settings.domain_magnitude).min(1.0);
                let color = Color::lerp_color(color_2, color_1, lerp_amount);
                Instance {
                    position,
                    rotation,
                    color,
                    fade_to_color: color,
                    instance_state,
                }
            }
            ColorMethod::Neighbor(color_1, color_2) => Instance {
                position,
                rotation,
                color: color_1,
                fade_to_color: color_2,
                instance_state,
            },
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct InstanceRaw {
    transform: [[f32; 4]; 4],
    color: [f32; 4],
    instance_state: f32,
}

unsafe impl Pod for InstanceRaw {}
unsafe impl Zeroable for InstanceRaw {}

impl InstanceRaw {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
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
            ],
        }
    }
}

pub struct InstanceManager {
    instances: Vec<Vec<Vec<Instance>>>,
    flattened: Vec<Instance>,
    bounding_box_instance: Instance,
}

impl InstanceManager {
    fn default(settings: &Settings) -> Self {
        InstanceManager {
            instances: Vec::new(),
            flattened: Vec::new(),
            bounding_box_instance: Instance::create_bounding_box(settings),
        }
    }

    pub fn new(settings: &Settings) -> Self {
        let mut instance_manager = InstanceManager::default(settings);
        instance_manager.prepare_initial_instances(settings);
        // skipping scheduling to create a new buffer as it is handled externally
        instance_manager.flatten(|| ());
        instance_manager
    }

    pub fn generate_offset(&self, rng: &mut ThreadRng, settings: &Settings) -> f32 {
        if settings.noise_amount > 0 {
            let rand_offset = rng.gen_range(0..settings.noise_amount);
            if rand::random::<f32>() > 0.5 {
                rand_offset as f32
            } else {
                rand_offset as f32 * -1.0
            }
        } else {
            0.0
        }
    }

    pub fn spawn_condition(&self, pos: f32, offset: f32, half_total: f32, half_spawn: f32) -> bool {
        (pos >= (half_total - half_spawn + offset)) && (pos < (half_total + half_spawn + offset))
    }

    pub fn prepare_initial_instances(&mut self, settings: &Settings) {
        let half_total = settings.domain_size as f32 / 2.0;
        let half_spawn = settings.spawn_size as f32 / 2.0;
        let mut rng: ThreadRng = rand::thread_rng();
        self.instances = (0..settings.domain_size)
            .map(|x| {
                (0..settings.domain_size)
                    .map(|y| {
                        (0..settings.domain_size)
                            .map(|z| {
                                let x_rand_offset = self.generate_offset(&mut rng, settings);
                                let y_rand_offset = self.generate_offset(&mut rng, settings);
                                let z_rand_offset = self.generate_offset(&mut rng, settings);
                                let (x_scaled, y_scaled, z_scaled) =
                                    Instance::scale_x_y_z(x, y, z, settings);
                                // randomize the spawn conditions to create a random initial state
                                if self.spawn_condition(
                                    x as f32,
                                    x_rand_offset,
                                    half_total,
                                    half_spawn,
                                ) && self.spawn_condition(
                                    y as f32,
                                    y_rand_offset,
                                    half_total,
                                    half_spawn,
                                ) && self.spawn_condition(
                                    z as f32,
                                    z_rand_offset,
                                    half_total,
                                    half_spawn,
                                ) {
                                    Instance::create_instance_at_pos(
                                        settings,
                                        x_scaled,
                                        y_scaled,
                                        z_scaled,
                                        CellState::default(),
                                    )
                                } else {
                                    Instance::create_instance_at_pos(
                                        settings,
                                        x_scaled,
                                        y_scaled,
                                        z_scaled,
                                        CellState::dead(),
                                    )
                                }
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
    }

    pub fn prepare_raw_instance_data(&self) -> Vec<InstanceRaw> {
        self.flattened
            .iter()
            .map(|instance| Instance::to_raw(*instance))
            .collect::<Vec<_>>()
    }

    pub fn create_new_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        let instance_data = self.prepare_raw_instance_data();
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        })
    }

    #[cfg(feature = "multithreading")]
    pub fn sort_by_distance_to_camera(&mut self, camera: &Camera) -> Duration {
        let start = std::time::Instant::now();
        let camera_position = camera.position.to_vec();
        self.flattened.par_sort_by(|a, b| {
            let a_dist = (a.position - camera_position).magnitude();
            let b_dist = (b.position - camera_position).magnitude();
            b_dist.partial_cmp(&a_dist).unwrap()
        });
        // SimulationMode::Gpu => {
        //     // Maybe use the gpu to sort the instances
        // }
        return start.elapsed();
    }

    #[cfg(not(feature = "multithreading"))]
    pub fn sort_by_distance_to_camera(&mut self, camera: &Camera) -> Duration {
        let start = std::time::Instant::now();
        let camera_position = camera.position.to_vec();
        self.flattened.sort_by(|a, b| {
            let a_dist = (a.position - camera_position).magnitude();
            let b_dist = (b.position - camera_position).magnitude();
            b_dist.partial_cmp(&a_dist).unwrap()
        });
        return start.elapsed();
    }

    pub fn flatten<F: FnOnce()>(&mut self, schedule_create_new_buffer: F) {
        let mut new_flattened = Vec::new();
        for x in 0..self.instances.len() {
            for y in 0..self.instances[x].len() {
                for z in 0..self.instances[x][y].len() {
                    if self.instances[x][y][z].instance_state.state != CellStateEnum::Dead {
                        new_flattened.push(self.instances[x][y][z]);
                    }
                }
            }
        }
        let adjust_buffer_size_required =
            self.check_if_new_buffer_needs_to_be_created(new_flattened.len());

        if adjust_buffer_size_required {
            schedule_create_new_buffer();
        }
        self.flattened = new_flattened;
    }

    /// Do this before Updating self.flattened or it will not work!!!
    pub fn check_if_new_buffer_needs_to_be_created(&self, new_flattened_length: usize) -> bool {
        self.flattened.len() < new_flattened_length
    }

    pub fn handle_simulation_result(
        &mut self,
        did_something: bool,
        update_queue: &mut UpdateQueue,
    ) -> Option<SimulationState> {
        if did_something {
            self.flatten(|| {
                update_queue.add(UpdateEnum::SortInstances);
                update_queue.add(UpdateEnum::CreateNewInstanceBuffer)
            });
            if update_queue.last() != Some(&UpdateEnum::CreateNewInstanceBuffer) {
                update_queue.add(UpdateEnum::SortInstances);
                update_queue.add(UpdateEnum::UpdateInstanceBuffer);
            }
            None
        } else {
            log::warn!("Simulation has reached a stable state, pausing simulation");
            Some(SimulationState::Stable)
        }
    }

    #[cfg(feature = "multithreading")]
    pub fn simulate(
        &mut self,
        settings: &Settings,
        update_queue: &mut UpdateQueue,
    ) -> Option<SimulationState> {
        let instance_cache = self.instances.clone();
        let did_something = self
            .instances
            .par_iter_mut()
            .enumerate()
            .map(|(x, layer)| Self::per_thread_simulate(layer, settings, x, &instance_cache))
            .reduce(|| false, |a, b| a || b);

        self.handle_simulation_result(did_something, update_queue)
    }

    #[cfg(not(feature = "multithreading"))]
    pub fn simulate(
        &mut self,
        settings: &mut Settings,
        update_queue: &mut UpdateQueue,
    ) -> Option<SimulationState> {
        let instance_cache = self.instances.clone();
        let did_anything = self
            .instances
            .iter_mut()
            .enumerate()
            .map(|(x, layer)| Self::per_thread_simulate(layer, settings, x, &instance_cache))
            .fold(false, |a, b| a || b);

        self.handle_simulation_result(did_anything, update_queue)
    }

    pub fn update_buffer(&mut self, queue: &wgpu::Queue, instance_buffer: &wgpu::Buffer) {
        // update the instance buffer
        let instance_data = self.prepare_raw_instance_data();
        queue.write_buffer(instance_buffer, 0, bytemuck::cast_slice(&instance_data));
    }

    pub fn update_bounding_box_instance_buffer(
        &self,
        queue: &wgpu::Queue,
        bounding_box_instance_buffer: &wgpu::Buffer,
    ) {
        let instance_data = vec![self.bounding_box_instance.to_raw()];
        queue.write_buffer(
            bounding_box_instance_buffer,
            0,
            bytemuck::cast_slice(&instance_data),
        );
    }

    pub fn update_bounding_box_vertex_buffer(
        &self,
        settings: &Settings,
        queue: &wgpu::Queue,
        bounding_box_vertex_buffer: &wgpu::Buffer,
    ) {
        let (vertices, _) =
            Vertex::create_vertices(Instance::calculate_bounding_box_size(settings));
        queue.write_buffer(
            bounding_box_vertex_buffer,
            0,
            bytemuck::cast_slice(&vertices),
        );
    }

    pub fn update_transparency(&mut self, settings: &Settings) -> bool {
        let mut new_flattened = Vec::new();
        for layer in self.instances.iter_mut() {
            for row in layer.iter_mut() {
                for instance in row.iter_mut() {
                    instance.color.w = settings.transparency;
                    if instance.instance_state.state != CellStateEnum::Dead {
                        new_flattened.push(*instance);
                    }
                }
            }
        }
        let adjust_buffer_size_required =
            self.check_if_new_buffer_needs_to_be_created(new_flattened.len());
        self.flattened = new_flattened;
        adjust_buffer_size_required
    }

    pub fn update_space_between_instances(&mut self, settings: &Settings) -> bool {
        let mut new_flattened = Vec::new();
        for (x, layer) in self.instances.iter_mut().enumerate() {
            for (y, row) in layer.iter_mut().enumerate() {
                for (z, instance) in row.iter_mut().enumerate() {
                    let (ix, iy, iz) =
                        Instance::scale_x_y_z(x as u32, y as u32, z as u32, settings);
                    instance.position = cgmath::Vector3 {
                        x: ix,
                        y: iy,
                        z: iz,
                    };
                    if instance.instance_state.state != CellStateEnum::Dead {
                        new_flattened.push(*instance);
                    }
                }
            }
        }
        let adjust_buffer_size_required =
            self.check_if_new_buffer_needs_to_be_created(new_flattened.len());
        self.flattened = new_flattened;
        adjust_buffer_size_required
    }

    pub fn increase_domain_size(&mut self, settings: &Settings) -> Option<UpdateEnum> {
        let mut new_instances: Vec<Vec<Vec<Instance>>> = Vec::new();
        let mut return_update_enum = None;
        for x in 0..settings.domain_size {
            let mut y_instances: Vec<Vec<Instance>> = Vec::new();
            for y in 0..settings.domain_size {
                let mut z_instances: Vec<Instance> = Vec::new();
                for z in 0..settings.domain_size {
                    let instance = if x == 0
                        || y == 0
                        || z == 0
                        || x == settings.domain_size - 1
                        || y == settings.domain_size - 1
                        || z == settings.domain_size - 1
                    {
                        // Create a new instance
                        let (x_scaled, y_scaled, z_scaled) =
                            Instance::scale_x_y_z(x, y, z, settings);
                        Instance::create_instance_at_pos(
                            settings,
                            x_scaled,
                            y_scaled,
                            z_scaled,
                            CellState::dead(),
                        )
                    } else {
                        self.instances[(x - 1) as usize][(y - 1) as usize][(z - 1) as usize]
                    };
                    z_instances.push(instance);
                }
                y_instances.push(z_instances);
            }
            new_instances.push(y_instances);
        }
        self.instances = new_instances;
        self.flatten(|| return_update_enum = Some(UpdateEnum::CreateNewInstanceBuffer));
        return_update_enum
    }

    pub fn decrease_domain_size(&mut self, settings: &Settings) -> Option<UpdateEnum> {
        let mut new_instances: Vec<Vec<Vec<Instance>>> = Vec::new();
        let mut return_update_enum = None;
        // remove the outer layer of instances and keep the cube inside eg initially 12x12x12 then 10x10x10 remove 1 layer from each side
        for x in 1..settings.domain_size + 1 {
            let mut y_instances: Vec<Vec<Instance>> = Vec::new();
            for y in 1..settings.domain_size + 1 {
                let mut z_instances: Vec<Instance> = Vec::new();
                for z in 1..settings.domain_size + 1 {
                    z_instances.push(self.instances[x as usize][y as usize][z as usize]);
                }
                y_instances.push(z_instances);
            }
            new_instances.push(y_instances);
        }
        self.instances = new_instances;
        self.flatten(|| return_update_enum = Some(UpdateEnum::CreateNewInstanceBuffer));
        return_update_enum
    }

    pub fn per_thread_simulate(
        layer: &mut [Vec<Instance>],
        settings: &Settings,
        x: usize,
        instance_cache: &[Vec<Vec<Instance>>],
    ) -> bool {
        let mut thread_did_something = false;
        for (y, row) in layer.iter_mut().enumerate() {
            for (z, instance) in row.iter_mut().enumerate() {
                let mut alive_neighbors = 0;
                let neighbors = settings
                    .simulation_rules
                    .neighbor_method
                    .get_neighbor_iter();
                for (dx, dy, dz) in neighbors {
                    let ix = ((x as i32 + dx) % settings.domain_size as i32
                        + settings.domain_size as i32)
                        % settings.domain_size as i32;
                    let iy = ((y as i32 + dy) % settings.domain_size as i32
                        + settings.domain_size as i32)
                        % settings.domain_size as i32;
                    let iz = ((z as i32 + dz) % settings.domain_size as i32
                        + settings.domain_size as i32)
                        % settings.domain_size as i32;

                    let neighbor = &instance_cache[ix as usize][iy as usize][iz as usize];
                    if neighbor.instance_state.state == CellStateEnum::Alive {
                        alive_neighbors += 1;
                    }
                }

                match instance.instance_state.state {
                    CellStateEnum::Alive => {
                        if !settings
                            .simulation_rules
                            .survival
                            .contains(&alive_neighbors)
                        {
                            if settings.simulation_rules.num_states > 2 {
                                instance.instance_state.state = CellStateEnum::Fading;
                            } else {
                                instance.instance_state.state = CellStateEnum::Dead;
                            }
                            thread_did_something = true;
                        }
                    }
                    CellStateEnum::Fading => {
                        if instance.instance_state.fade_level
                            >= settings.simulation_rules.num_states
                        {
                            instance.instance_state.state = CellStateEnum::Dead;
                        } else if instance.instance_state.fade_level
                            < settings.simulation_rules.num_states
                        {
                            instance.instance_state.fade_level += 1;
                            instance.color = match settings.color_method {
                                ColorMethod::Single(c) => {
                                    let instance_transparency =
                                        if settings.simulation_rules.num_states > 2 {
                                            if instance.instance_state.fade_level == 0 {
                                                settings.transparency
                                            } else if instance.instance_state.fade_level
                                                == settings.simulation_rules.num_states - 1
                                            {
                                                0.0
                                            } else {
                                                settings.transparency
                                                    - (instance.instance_state.fade_level as f32
                                                        / (settings.simulation_rules.num_states - 1)
                                                            as f32)
                                            }
                                        } else {
                                            settings.transparency
                                        };
                                    cgmath::Vector4::new(c.x, c.y, c.z, instance_transparency)
                                }
                                ColorMethod::StateLerp(c1, c2) => {
                                    let dt = (instance.instance_state.fade_level
                                        / (settings.simulation_rules.num_states - 1))
                                        as f32;
                                    Color::lerp_color(c2, c1, dt)
                                }
                                ColorMethod::DistToCenter(c1, c2) => {
                                    let lerp_amount = (instance.position.magnitude()
                                        / settings.domain_magnitude)
                                        .min(1.0);
                                    Color::lerp_color(c2, c1, lerp_amount)
                                }
                                ColorMethod::Neighbor(c1, c2) => {
                                    let dt = alive_neighbors as f32
                                        / settings
                                            .simulation_rules
                                            .neighbor_method
                                            .total_num_neighbors()
                                            as f32;
                                    Color::lerp_color(c2, c1, dt)
                                }
                            }
                        }
                        thread_did_something = true;
                    }
                    CellStateEnum::Dead => {
                        if settings.simulation_rules.birth.contains(&alive_neighbors) {
                            instance.instance_state.state = CellStateEnum::Alive;
                            instance.instance_state.fade_level = 0;
                            thread_did_something = true;
                        }
                    }
                }
            }
        }
        thread_did_something
    }

    pub fn num_flattened_instances(&self) -> usize {
        self.flattened.len()
    }
}
