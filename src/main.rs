use bytemuck::{Pod, Zeroable};
use cellular_automata_3d::{
    camera::{setup_camera, Camera, CameraController, CameraUniform, Projection},
    constants::DEPTH_FORMAT,
    simulation::{CellState, CellStateEnum, SimulationRules},
    texture::{self, Texture},
    utils::{init_logger, CommandLineArgs, Validator},
    vertex::Vertex,
};
use cgmath::{EuclideanSpace, InnerSpace, Rotation3};
use clap::Parser;
use rand::{rngs::ThreadRng, Rng};
use std::{borrow::Cow, f32::consts, mem};
use wgpu::{util::DeviceExt, PipelineCompilationOptions};
use winit::{
    event::{DeviceEvent, ElementState, KeyEvent, MouseButton, WindowEvent},
    keyboard::{Key, NamedKey},
};

#[cfg(feature = "multithreading")]
use rayon::prelude::*;

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

    fn schedule_updates(&mut self, new_updates: Vec<UpdateEnum>) {
        self.queue.extend(new_updates);
    }
}

#[derive(PartialEq)]
enum UpdateEnum {
    Transparency,
    NumInstancesIncreased,
    NumInstancesDecreased,
    SpaceBetweenInstances,
    SortInstances,
    UpdateBoundingBox,
    UpdateBuffer,
    CreateNewBuffer,
}

#[derive(Debug, PartialEq)]
enum SimulationMode {
    SingleThreaded,
    #[cfg(feature = "multithreading")]
    MultiThreaded,
    // Gpu,
}

// Allow Dead Code as multithreading uses simulation mode but when rayon is not enabled it is not used
#[allow(dead_code)]
pub struct Settings {
    bounding_box_active: bool,
    world_grid_active: bool,
    domain_size: u32,
    cube_size: f32,
    space_between_instances: f32,
    transparency: f32,
    num_instances_step_size: u32,
    transparency_step_size: f32,
    space_between_step_size: f32,
    angle_threshold_for_sort: f32,
    translation_threshold_for_sort: f32,
    simulation_tick_rate: u16,
    last_simulation_tick: std::time::Instant,
    spawn_size: u8,
    noise_amount: u8,
    simulation_rules: SimulationRules,
    simulation_paused: bool,
    simulation_mode: SimulationMode,
}

impl Default for Settings {
    fn default() -> Self {
        Self::new(CommandLineArgs::default())
    }
}

impl Settings {
    pub fn new(command_line_args: CommandLineArgs) -> Self {
        let simulation_rules = SimulationRules::parse_rules(command_line_args.rules.as_deref());
        let simulation_tick_rate = Validator::validate_simulation_tick_rate(
            command_line_args.simulation_tick_rate.unwrap_or(10),
        );
        let domain_size =
            Validator::validate_domain_size(command_line_args.domain_size.unwrap_or(20));
        let spawn_size = Validator::validate_spawn_size(
            command_line_args.initial_spawn_size.unwrap_or(10),
            domain_size as u8,
        );
        // Add 1 to avoid no noise on setting noise to 1 as it will calculate random offset from 0 to 1
        // which is not visible in the grid, hence we need at least 2 to see the noise
        let noise_amount =
            (Validator::validate_noise_amount(command_line_args.noise_level.unwrap_or(0)) + 1)
                .min(10);

        let bounding_box_active = false;
        let world_grid_active = !command_line_args.disable_world_grid;
        let cube_size = 1.0;
        let space_between_instances = 0.1;
        let transparency = 1.0;
        let num_instances_step_size = 2;
        let transparency_step_size = 0.1;
        let space_between_step_size = 0.05;
        let angle_threshold_for_sort = 0.10;
        let translation_threshold_for_sort = 10.0;
        let last_simulation_tick = std::time::Instant::now();
        let simulation_paused = true; // Start simulation paused

        #[cfg(feature = "multithreading")]
        let simulation_mode = SimulationMode::MultiThreaded;

        #[cfg(not(feature = "multithreading"))]
        let simulation_mode = SimulationMode::SingleThreaded;

        log::debug!("Setting simulation tick rate to: {}", simulation_tick_rate);
        log::debug!("Setting domain size to: {}", domain_size);
        log::debug!("Setting initial spawn size to: {}", spawn_size);
        log::debug!("Setting noise level to: {}", noise_amount);

        Settings {
            bounding_box_active,
            world_grid_active,
            domain_size,
            cube_size,
            space_between_instances,
            transparency,
            num_instances_step_size,
            transparency_step_size,
            space_between_step_size,
            angle_threshold_for_sort,
            translation_threshold_for_sort,
            simulation_tick_rate,
            last_simulation_tick,
            spawn_size,
            noise_amount,
            simulation_rules,
            simulation_paused,
            simulation_mode,
        }
    }

    #[cfg(feature = "multithreading")]
    pub fn next_simulation_mode(&mut self) {
        self.simulation_mode = match self.simulation_mode {
            SimulationMode::SingleThreaded => SimulationMode::MultiThreaded,
            SimulationMode::MultiThreaded => SimulationMode::SingleThreaded,
            // SimulationMode::Gpu => SimulationMode::SingleThreaded,
        };
        log::info!("Simulation mode set to: {:?}", self.simulation_mode);
    }

    #[cfg(not(feature = "multithreading"))]
    pub fn next_simulation_mode(&mut self) {
        log::warn!("Multithreading feature is not enabled, cannot switch to multi-threaded mode");
    }

    pub fn toggle_bounding_box(&mut self) {
        self.bounding_box_active = !self.bounding_box_active;
        log::info!("Wireframe overlay set to: {}", self.bounding_box_active);
    }

    pub fn toggle_world_grid(&mut self) {
        self.world_grid_active = !self.world_grid_active;
        log::info!("World grid display set to: {}", self.world_grid_active);
    }

    pub fn toggle_pause_simulation(&mut self) {
        self.simulation_paused = !self.simulation_paused;
        if self.simulation_paused {
            log::info!("Simulation paused");
        } else {
            log::info!("Simulation resumed");
        }
    }

    pub fn set_domain_size(&mut self, domain_size: u32, update_queue: &mut UpdateQueue) {
        let validated_domain_size = Validator::validate_domain_size(domain_size);

        match validated_domain_size.cmp(&self.domain_size) {
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
        self.domain_size = validated_domain_size;
        log::info!("Domain size set to: {}", self.domain_size);
    }

    pub fn set_transparency(&mut self, transparency: f32, update_queue: &mut UpdateQueue) {
        self.transparency = Validator::validate_transparency(transparency);
        log::info!("Transparency set to: {}", self.transparency);
        update_queue.add(UpdateEnum::Transparency);
    }

    pub fn set_space_between_instances(
        &mut self,
        space_between_instances: f32,
        update_queue: &mut UpdateQueue,
    ) {
        self.space_between_instances =
            Validator::validate_space_between_instances(space_between_instances);
        log::info!(
            "Space between instances set to: {}",
            (self.space_between_instances * 100.0).round() / 100.0
        );
        update_queue.add(UpdateEnum::SpaceBetweenInstances);
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Instance {
    position: cgmath::Vector3<f32>,
    rotation: cgmath::Quaternion<f32>,
    color: cgmath::Vector4<f32>,
    instance_state: CellState,
}

impl Instance {
    fn to_raw(self, settings: &Settings) -> InstanceRaw {
        let transform =
            cgmath::Matrix4::from_translation(self.position) * cgmath::Matrix4::from(self.rotation);
        InstanceRaw {
            transform: transform.into(),
            color: self.color.into(),
            instance_state: self.instance_state.state.to_int() as f32,
            instance_death_transparency: if settings.simulation_rules.num_states > 2 {
                if self.instance_state.state == CellStateEnum::Fading {
                    if self.instance_state.fade_level == 0 {
                        settings.transparency
                    } else if self.instance_state.fade_level
                        == settings.simulation_rules.num_states - 1
                    {
                        0.0
                    } else {
                        settings.transparency
                            - (self.instance_state.fade_level as f32
                                / (settings.simulation_rules.num_states - 1) as f32)
                    }
                } else {
                    settings.transparency
                }
            } else {
                settings.transparency
            },
        }
    }

    fn create_instance_at_pos(settings: &Settings, x: f32, y: f32, z: f32) -> Instance {
        let position = cgmath::Vector3 { x, y, z };

        let rotation =
            cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0));

        // TODO: use a Color Strategy pattern to determine the color
        // give colors based on position
        let color = cgmath::Vector4 {
            x: (x + settings.domain_size as f32) / (settings.domain_size as f32 * 2.0),
            y: (y + settings.domain_size as f32) / (settings.domain_size as f32 * 2.0),
            z: (z + settings.domain_size as f32) / (settings.domain_size as f32 * 2.0),
            w: settings.transparency,
        };

        Instance {
            position,
            rotation,
            color,
            instance_state: CellState::default(),
        }
    }

    fn create_dead_instance_at_pos(settings: &Settings, x: f32, y: f32, z: f32) -> Instance {
        let position = cgmath::Vector3 { x, y, z };

        let rotation =
            cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0));

        // give colors based on position
        let color = cgmath::Vector4 {
            x: (x + settings.domain_size as f32) / (settings.domain_size as f32 * 2.0),
            y: (y + settings.domain_size as f32) / (settings.domain_size as f32 * 2.0),
            z: (z + settings.domain_size as f32) / (settings.domain_size as f32 * 2.0),
            w: settings.transparency,
        };

        Instance {
            position,
            rotation,
            color,
            instance_state: CellState {
                state: CellStateEnum::Dead,
                fade_level: 0,
            },
        }
    }

    fn create_bounding_box(settings: &Settings) -> Instance {
        Instance::create_dead_instance_at_pos(
            settings,
            -settings.space_between_instances,
            -settings.space_between_instances,
            -settings.space_between_instances,
        )
    }

    fn calculate_bounding_box_size(settings: &Settings) -> f32 {
        (settings.cube_size as f64
            + ((settings.space_between_instances as f64 + (settings.cube_size * 2.0) as f64)
                * (settings.domain_size as f64 / 2.0))) as f32
    }

    // Not required anymore but keeping it for reference
    // fn convert_3d_to_flattened_index(x: u32, y: u32, z: u32, domain_size: u32) -> usize {
    //     ((x * domain_size.pow(2)) + (y * domain_size) + z) as usize
    // }

    fn scale_x_y_z(x: u32, y: u32, z: u32, settings: &Settings) -> (f32, f32, f32) {
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
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct InstanceRaw {
    transform: [[f32; 4]; 4],
    color: [f32; 4],
    instance_state: f32,
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

    fn new(settings: &Settings) -> Self {
        let mut instance_manager = InstanceManager::default(settings);
        instance_manager.prepare_initial_instances(settings);
        // skipping scheduling to create a new buffer as it is handled externally
        instance_manager.flatten(|| ());
        instance_manager
    }

    fn generate_offset(&self, rng: &mut ThreadRng, settings: &Settings) -> f32 {
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

    fn spawn_condition(&self, pos: f32, offset: f32, half_total: f32, half_spawn: f32) -> bool {
        (pos >= (half_total - half_spawn + offset)) && (pos < (half_total + half_spawn + offset))
    }

    fn prepare_initial_instances(&mut self, settings: &Settings) {
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
                                        settings, x_scaled, y_scaled, z_scaled,
                                    )
                                } else {
                                    Instance::create_dead_instance_at_pos(
                                        settings, x_scaled, y_scaled, z_scaled,
                                    )
                                }
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
    }

    fn prepare_raw_instance_data(&self, settings: &Settings) -> Vec<InstanceRaw> {
        self.flattened
            .iter()
            .map(|instance| Instance::to_raw(*instance, settings))
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

    #[cfg(feature = "multithreading")]
    fn sort_by_distance_to_camera(&mut self, camera: &Camera) {
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
        log::debug!("Time to sort: {:?}", start.elapsed());
    }

    #[cfg(not(feature = "multithreading"))]
    fn sort_by_distance_to_camera(&mut self, camera: &Camera) {
        let start = std::time::Instant::now();
        let camera_position = camera.position.to_vec();
        self.flattened.sort_by(|a, b| {
            let a_dist = (a.position - camera_position).magnitude();
            let b_dist = (b.position - camera_position).magnitude();
            b_dist.partial_cmp(&a_dist).unwrap()
        });
        log::debug!("Time to sort: {:?}", start.elapsed());
    }

    fn flatten<F: FnOnce()>(&mut self, schedule_create_new_buffer: F) {
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
    fn check_if_new_buffer_needs_to_be_created(&self, new_flattened_length: usize) -> bool {
        self.flattened.len() < new_flattened_length
    }

    fn handle_simulation_result(&mut self, did_something: bool, settings: &mut Settings, update_queue: &mut UpdateQueue) {
        if did_something {
            self.flatten(|| {
                update_queue.add(UpdateEnum::SortInstances);
                update_queue.add(UpdateEnum::CreateNewBuffer)
            });
            if update_queue.queue.last() != Some(&UpdateEnum::CreateNewBuffer) {
                update_queue.add(UpdateEnum::SortInstances);
                update_queue.add(UpdateEnum::UpdateBuffer);
            }
        } else {
            log::warn!("Simulation has reached a stable state, pausing simulation");
            settings.simulation_paused = true;
        }
    }
    
    #[cfg(feature = "multithreading")]
    fn simulate(&mut self, settings: &mut Settings, update_queue: &mut UpdateQueue) {
        let instance_cache = self.instances.clone();
        let did_something = self
            .instances
            .par_iter_mut()
            .enumerate()
            .map(|(x, layer)| per_thread_simulate(layer, settings, x, &instance_cache))
            .reduce(|| false, |a, b| a || b);
    
        self.handle_simulation_result(did_something, settings, update_queue);
    }
    
    #[cfg(not(feature = "multithreading"))]
    fn simulate(&mut self, settings: &mut Settings, update_queue: &mut UpdateQueue) {
        let instance_cache = self.instances.clone();
        let did_anything = self
            .instances
            .iter_mut()
            .enumerate()
            .map(|(x, layer)| per_thread_simulate(layer, settings, x, &instance_cache))
            .fold(false, |a, b| a || b);
    
        self.handle_simulation_result(did_anything, settings, update_queue);
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

    fn update_bounding_box_instance_buffer(
        &self,
        settings: &Settings,
        queue: &wgpu::Queue,
        bounding_box_instance_buffer: &wgpu::Buffer,
    ) {
        let instance_data = vec![self.bounding_box_instance.to_raw(settings)];
        queue.write_buffer(
            bounding_box_instance_buffer,
            0,
            bytemuck::cast_slice(&instance_data),
        );
    }

    fn update_bounding_box_vertex_buffer(
        &self,
        settings: &Settings,
        queue: &wgpu::Queue,
        bounding_box_vertex_buffer: &wgpu::Buffer,
    ) {
        let (vertices, _) = Vertex::create_vertices_for_bounding_box(
            Instance::calculate_bounding_box_size(settings),
        );
        queue.write_buffer(
            bounding_box_vertex_buffer,
            0,
            bytemuck::cast_slice(&vertices),
        );
    }

    fn update_transparency(&mut self, settings: &Settings) -> bool {
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

    fn update_space_between_instances(&mut self, settings: &Settings) -> bool {
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

    fn increase_domain_size(&mut self, settings: &Settings, camera: &Camera) -> Option<UpdateEnum> {
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
                        Instance::create_dead_instance_at_pos(
                            settings, x_scaled, y_scaled, z_scaled,
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
        self.flatten(|| return_update_enum = Some(UpdateEnum::CreateNewBuffer));
        self.sort_by_distance_to_camera(camera);
        return_update_enum
    }

    fn decrease_domain_size(&mut self, settings: &Settings, camera: &Camera) -> Option<UpdateEnum> {
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
        self.flatten(|| return_update_enum = Some(UpdateEnum::CreateNewBuffer));
        self.sort_by_distance_to_camera(camera);
        return_update_enum
    }
}

fn per_thread_simulate(
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
                    if instance.instance_state.fade_level >= settings.simulation_rules.num_states {
                        instance.instance_state.state = CellStateEnum::Dead;
                    } else if instance.instance_state.fade_level
                        < settings.simulation_rules.num_states
                    {
                        instance.instance_state.fade_level += 1;
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

struct App {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
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
    bounding_box_vertex_buf: wgpu::Buffer,
    bounding_box_index_buf: wgpu::Buffer,
    bounding_box_num_indices: u32,
    bounding_box_instance_buffer: wgpu::Buffer,
    world_grid_pipeline: wgpu::RenderPipeline,
    world_grid_pipeline_depth: wgpu::RenderPipeline,
    world_grid_vertices: Vec<Vertex>,
    world_grid_buffer: wgpu::Buffer,
}

impl App {
    fn generate_matrix(aspect_ratio: f32) -> glam::Mat4 {
        let projection = glam::Mat4::perspective_rh(consts::FRAC_PI_4, aspect_ratio, 1.0, 1.0);
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
        let translation_diff = self.camera.position - self.last_sort_camera.position;
        if translation_diff.magnitude() > self.settings.translation_threshold_for_sort
            || yaw_diff > self.settings.angle_threshold_for_sort
            || pitch_diff > self.settings.angle_threshold_for_sort
        {
            self.update_queue.add(UpdateEnum::SortInstances);
        }
    }

    fn simulation_update(&mut self) {
        if (self.settings.last_simulation_tick.elapsed().as_millis()
            >= self.settings.simulation_tick_rate.into())
            && !self.settings.simulation_paused
        {
            let start = std::time::Instant::now();
            self.instance_manager
                .simulate(&mut self.settings, &mut self.update_queue);
            self.settings.last_simulation_tick = std::time::Instant::now();
            log::debug!("Time to simulate: {:?}", start.elapsed());
        }
    }

    fn queue_update(&mut self, gpu_queue: &wgpu::Queue, device: &wgpu::Device) {
        let mut scheduled_updates = Vec::new();
        if self.update_queue.queue.is_empty() {
            return;
        }
        for update in self.update_queue.queue.iter() {
            match update {
                UpdateEnum::Transparency => {
                    if self.instance_manager.update_transparency(&self.settings) {
                        scheduled_updates.push(UpdateEnum::CreateNewBuffer);
                    }
                    scheduled_updates.push(UpdateEnum::SortInstances);
                    scheduled_updates.push(UpdateEnum::UpdateBuffer);
                }
                UpdateEnum::SpaceBetweenInstances => {
                    if self
                        .instance_manager
                        .update_space_between_instances(&self.settings)
                    {
                        scheduled_updates.push(UpdateEnum::CreateNewBuffer);
                    }
                    scheduled_updates.push(UpdateEnum::UpdateBoundingBox);
                    scheduled_updates.push(UpdateEnum::SortInstances);
                    scheduled_updates.push(UpdateEnum::UpdateBuffer);
                }
                UpdateEnum::SortInstances => {
                    self.instance_manager
                        .sort_by_distance_to_camera(&self.camera);
                    self.last_sort_camera = self.camera;
                }
                UpdateEnum::NumInstancesIncreased => {
                    if let Some(update_enum) = self
                        .instance_manager
                        .increase_domain_size(&self.settings, &self.camera)
                    {
                        scheduled_updates.push(update_enum);
                    };
                    scheduled_updates.push(UpdateEnum::UpdateBoundingBox);
                }
                UpdateEnum::NumInstancesDecreased => {
                    if let Some(update_enum) = self
                        .instance_manager
                        .decrease_domain_size(&self.settings, &self.camera)
                    {
                        scheduled_updates.push(update_enum);
                    };
                    scheduled_updates.push(UpdateEnum::UpdateBoundingBox);
                }
                UpdateEnum::UpdateBoundingBox => {
                    self.instance_manager.update_bounding_box_vertex_buffer(
                        &self.settings,
                        gpu_queue,
                        &self.bounding_box_vertex_buf,
                    );
                    self.instance_manager.update_bounding_box_instance_buffer(
                        &self.settings,
                        gpu_queue,
                        &self.bounding_box_instance_buffer,
                    );
                }
                UpdateEnum::UpdateBuffer => {
                    self.instance_manager.update_buffer(
                        &self.settings,
                        gpu_queue,
                        &self.instance_buffer,
                    );
                }
                UpdateEnum::CreateNewBuffer => {
                    self.instance_buffer = self
                        .instance_manager
                        .create_new_buffer(&self.settings, device);
                }
            }
        }
        self.update_queue.reset();
        self.update_queue.schedule_updates(scheduled_updates);
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
        command_line_args: &CommandLineArgs,
    ) -> Self {
        let settings = Settings::new(command_line_args.clone());
        // Create the vertex and index buffers
        let (vertex_data, index_data) = Vertex::create_vertices(settings.cube_size);
        let (bounding_box_vertex_data, bounding_box_index_data) =
            Vertex::create_vertices_for_bounding_box(Instance::calculate_bounding_box_size(
                &settings,
            ));
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

        let bounding_box_vertex_buf =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Bounding Box Vertex Buffer"),
                contents: bytemuck::cast_slice(&bounding_box_vertex_data),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });

        let bounding_box_index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bounding Box Index Buffer"),
            contents: bytemuck::cast_slice(&bounding_box_index_data),
            usage: wgpu::BufferUsages::INDEX,
        });

        let mut instance_manager = InstanceManager::new(&settings);

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

        let bounding_box_instance = Instance::create_bounding_box(&settings);
        let bounding_box_instance_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Bounding Box Instance Buffer"),
                contents: bytemuck::cast_slice(&[bounding_box_instance.to_raw(&settings)]),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });

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
        let bounding_box_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Bounding Box Pipeline Layout"),
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

        let cube_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/shader.wgsl"))),
        });
        let world_grid_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "shaders/world_grid.wgsl"
            ))),
        });
        let world_grid_vertices = Vertex::create_vertices_for_world_grid(200.0);
        let world_grid_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("World Grid Vertex Buffer"),
            contents: bytemuck::cast_slice(&world_grid_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let vertex_buffers = &[Vertex::desc(), InstanceRaw::desc()];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &cube_shader,
                entry_point: "vs_main",
                buffers: vertex_buffers,
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &cube_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
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
                layout: Some(&bounding_box_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &cube_shader,
                    entry_point: "vs_main",
                    buffers: vertex_buffers,
                    compilation_options: PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &cube_shader,
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
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Line,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });
            Some(pipeline_wire)
        } else {
            None
        };
        let world_grid_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("World Grid Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &world_grid_shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &world_grid_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList, // Changed from LineList to TriangleList
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });
        let world_grid_pipeline_depth =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("World Grid Depth Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &world_grid_shader,
                    entry_point: "vs_main",
                    buffers: &[Vertex::desc()],
                    compilation_options: PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &world_grid_shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::SrcAlpha,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::Zero,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
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
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less, // Ensure correct depth test
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            });

        // Done
        App {
            vertex_buf,
            index_buf,
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
            bounding_box_vertex_buf,
            bounding_box_index_buf,
            bounding_box_num_indices: bounding_box_index_data.len() as u32,
            bounding_box_instance_buffer,
            world_grid_buffer,
            world_grid_pipeline,
            world_grid_pipeline_depth,
            world_grid_vertices,
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
                // TODO: Limit how fast a user can spam the keys to avoid trying to access for e.g. self.instances before increase_num_instances has finished
                if state == ElementState::Pressed {
                    if let Key::Character(s) = logical_key {
                        if s.as_str() == "p" {
                            self.settings.toggle_bounding_box();
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
                        } else if s.as_str() == "o" {
                            self.settings.toggle_pause_simulation();
                        } else if s.as_str() == "i" {
                            self.settings.toggle_world_grid();
                        } else if s.as_str() == "u" {
                            self.settings.next_simulation_mode();
                        }
                    } else if let Key::Named(key) = logical_key {
                        if key == NamedKey::PageUp {
                            self.settings.set_domain_size(
                                self.settings.domain_size + self.settings.num_instances_step_size,
                                &mut self.update_queue,
                            );
                        } else if key == NamedKey::PageDown {
                            self.settings.set_domain_size(
                                self.settings
                                    .domain_size
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
        self.simulation_update();
        // For any Scheduled updates
        self.queue_update(queue, device);
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
        let mut main_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut render_pass = main_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
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
        }

        if self.settings.world_grid_active {
            let mut world_grid_encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut render_pass =
                    world_grid_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                render_pass.set_pipeline(&self.world_grid_pipeline);
                render_pass.set_vertex_buffer(0, self.world_grid_buffer.slice(..));
                render_pass.set_bind_group(0, &self.bind_group, &[]);
                render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
                render_pass.draw(0..self.world_grid_vertices.len() as u32, 0..1);
            }

            let mut world_grid_2nd_pass_encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut render_pass =
                    world_grid_2nd_pass_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &self.depth_texture.view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                render_pass.set_pipeline(&self.world_grid_pipeline_depth);
                render_pass.set_vertex_buffer(0, self.world_grid_buffer.slice(..));
                render_pass.set_bind_group(0, &self.bind_group, &[]);
                render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
                render_pass.draw(0..self.world_grid_vertices.len() as u32, 0..1);
            }

            queue.submit(Some(world_grid_encoder.finish()));
            queue.submit(Some(main_encoder.finish()));
            queue.submit(Some(world_grid_2nd_pass_encoder.finish()));
        } else {
            queue.submit(Some(main_encoder.finish()));
        }

        if let Some(ref pipeline_wire) = self.pipeline_wire {
            if self.settings.bounding_box_active {
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut wireframe_render_pass =
                        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: None,
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        });

                    wireframe_render_pass.set_pipeline(pipeline_wire);
                    wireframe_render_pass.set_bind_group(0, &self.bind_group, &[]);
                    wireframe_render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
                    wireframe_render_pass
                        .set_vertex_buffer(0, self.bounding_box_vertex_buf.slice(..));
                    wireframe_render_pass
                        .set_vertex_buffer(1, self.bounding_box_instance_buffer.slice(..));
                    wireframe_render_pass.set_index_buffer(
                        self.bounding_box_index_buf.slice(..),
                        wgpu::IndexFormat::Uint16,
                    );
                    wireframe_render_pass.draw_indexed(0..self.bounding_box_num_indices, 0, 0..1);
                }

                queue.submit(Some(encoder.finish()));
            }
        }
    }
}

pub fn main() {
    // human_panic::setup_panic!();
    let args: CommandLineArgs = CommandLineArgs::parse();
    init_logger(args.debug);
    cellular_automata_3d::framework::run::<App>("cube", args);
}
