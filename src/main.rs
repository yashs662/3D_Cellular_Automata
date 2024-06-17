use cellular_automata_3d::{
    camera::{Camera, CameraController, CameraUniform, Projection},
    constants::{
        ANGLE_THRESHOLD_FOR_SORTING, DOMAIN_SIZE_STEP_SIZE, FONT_SIZE_STEP_SIZE,
        LINE_HEIGHT_MULTIPLIER_STEP_SIZE, MAX_DOMAIN_SIZE, MAX_FONT_SIZE,
        MAX_LINE_HEIGHT_MULTIPLIER, MIN_FONT_SIZE, MIN_LINE_HEIGHT_MULTIPLIER,
        MIN_USER_INPUT_INTERVAL, SPACE_BETWEEN_INSTANCES_STEP_SIZE,
        TRANSLATION_THRESHOLD_FOR_SORTING, TRANSPARENCY_STEP_SIZE, WORLD_GRID_SIZE_MULTIPLIER,
    },
    framework::Settings,
    graphics::{AppBuffers, AppRenderLayouts, AppRenderPipelines, AppShaders},
    instance::{Instance, InstanceManager, WorldGridRaw},
    simulation::SimulationState,
    text_renderer::TextRenderManager,
    texture::{self, Texture},
    utils::{generate_matrix, init_logger, Color, CommandLineArgs, UpdateEnum, UpdateQueue},
    vertex::Vertex,
};
use cgmath::InnerSpace;
use clap::Parser;
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;
use winit::{
    event::{DeviceEvent, ElementState, KeyEvent, MouseButton, WindowEvent},
    keyboard::{Key, NamedKey},
};

struct App {
    bind_group: wgpu::BindGroup,
    instance_manager: InstanceManager,
    cube_num_indices: u32,
    camera: Camera,
    last_sort_camera: Camera,
    projection: Projection,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_bind_group: wgpu::BindGroup,
    mouse_pressed: bool,
    settings: Settings,
    depth_texture: Texture,
    update_queue: UpdateQueue,
    bounding_box_num_indices: u32,
    world_grid_vertices: Vec<Vertex>,
    buffers: AppBuffers,
    pipelines: AppRenderPipelines,
    text_renderer: TextRenderManager,
    last_sort_time: Duration,
    simulation_state: SimulationState,
    last_simulation_time: Duration,
    last_cpu_time: Duration,
    last_user_input_instant: Instant,
}

impl App {
    fn update_camera(&mut self, dt: f32, queue: &wgpu::Queue) {
        self.camera_controller.update_camera(&mut self.camera, dt);
        self.camera_uniform
            .update_view_proj(&self.camera, &self.projection);
        queue.write_buffer(
            &self.buffers.camera,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
    }

    fn transparency_update(&mut self) {
        let yaw_diff = (self.camera.yaw - self.last_sort_camera.yaw).0.abs();
        let pitch_diff = (self.camera.pitch - self.last_sort_camera.pitch).0.abs();
        let translation_diff = self.camera.position - self.last_sort_camera.position;
        if translation_diff.magnitude() > TRANSLATION_THRESHOLD_FOR_SORTING
            || yaw_diff > ANGLE_THRESHOLD_FOR_SORTING
            || pitch_diff > ANGLE_THRESHOLD_FOR_SORTING
        {
            self.update_queue.add(UpdateEnum::SortInstances);
        }
    }

    fn simulation_update(&mut self) -> Option<Duration> {
        if (self.settings.last_simulation_tick.elapsed().as_millis()
            >= self.settings.simulation_tick_rate.into())
            && self.simulation_state == SimulationState::Active
        {
            let start = std::time::Instant::now();
            if let Some(simulation_state) = self
                .instance_manager
                .simulate(&self.settings, &mut self.update_queue)
            {
                self.simulation_state = simulation_state;
            }
            self.settings.last_simulation_tick = std::time::Instant::now();
            Some(start.elapsed())
        } else {
            None
        }
    }

    fn queue_update(&mut self, gpu_queue: &wgpu::Queue, device: &wgpu::Device) {
        let mut scheduled_updates = Vec::new();
        if self.update_queue.is_empty() {
            return;
        }
        for update in self.update_queue.iter() {
            match update {
                UpdateEnum::Transparency => {
                    self.instance_manager.update_transparency(&self.settings);
                    scheduled_updates.push(UpdateEnum::SortInstances);
                    scheduled_updates.push(UpdateEnum::UpdateInstanceBuffer);
                }
                UpdateEnum::SpaceBetweenInstances => {
                    self.instance_manager
                        .update_space_between_instances(&self.settings);
                    scheduled_updates.push(UpdateEnum::UpdateBoundingBox);
                    scheduled_updates.push(UpdateEnum::UpdateWorldGrid);
                    scheduled_updates.push(UpdateEnum::SortInstances);
                    scheduled_updates.push(UpdateEnum::UpdateInstanceBuffer);
                }
                UpdateEnum::SortInstances => {
                    self.last_sort_time = self
                        .instance_manager
                        .sort_by_distance_to_camera(&self.camera);
                    self.last_sort_camera = self.camera;
                }
                UpdateEnum::NumInstancesIncreased => {
                    if let Some(update_enum) =
                        self.instance_manager.increase_domain_size(&self.settings)
                    {
                        scheduled_updates.push(update_enum);
                        scheduled_updates.push(UpdateEnum::SortInstances);
                    } else {
                        scheduled_updates.push(UpdateEnum::SortInstances);
                        scheduled_updates.push(UpdateEnum::UpdateInstanceBuffer);
                    };
                    scheduled_updates.push(UpdateEnum::SortInstances);
                    scheduled_updates.push(UpdateEnum::UpdateWorldGrid);
                    scheduled_updates.push(UpdateEnum::UpdateBoundingBox);
                }
                UpdateEnum::NumInstancesDecreased => {
                    if let Some(update_enum) =
                        self.instance_manager.decrease_domain_size(&self.settings)
                    {
                        scheduled_updates.push(update_enum);
                        scheduled_updates.push(UpdateEnum::SortInstances);
                    } else {
                        scheduled_updates.push(UpdateEnum::SortInstances);
                        scheduled_updates.push(UpdateEnum::UpdateInstanceBuffer);
                    };
                    scheduled_updates.push(UpdateEnum::UpdateWorldGrid);
                    scheduled_updates.push(UpdateEnum::UpdateBoundingBox);
                }
                UpdateEnum::UpdateBoundingBox => {
                    self.instance_manager.update_bounding_box_vertex_buffer(
                        &self.settings,
                        gpu_queue,
                        &self.buffers.bounding_box_vertex,
                    );
                    self.instance_manager.update_bounding_box_instance_buffer(
                        gpu_queue,
                        &self.buffers.bounding_box_instance,
                    );
                }
                UpdateEnum::UpdateInstanceBuffer => {
                    self.instance_manager
                        .update_buffer(gpu_queue, &self.buffers.simulation_instances);
                }
                UpdateEnum::CreateNewInstanceBuffer => {
                    self.buffers.simulation_instances =
                        self.instance_manager.create_new_buffer(device);
                }
                UpdateEnum::UpdateWorldGrid => {
                    let new_size = Instance::calculate_bounding_box_size(&self.settings)
                        * WORLD_GRID_SIZE_MULTIPLIER;
                    self.world_grid_vertices = Vertex::create_vertices_for_world_grid(new_size);
                    let world_grid_raw = WorldGridRaw::new(
                        &self.world_grid_vertices,
                        self.settings.domain_size as f32 / MAX_DOMAIN_SIZE as f32,
                    );
                    let new_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("World Grid Vertex Buffer"),
                        contents: bytemuck::cast_slice(&world_grid_raw),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    self.buffers.world_grid = new_buffer;
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
        queue: &wgpu::Queue,
        command_line_args: &CommandLineArgs,
        scale_factor: f64,
    ) -> Self {
        let settings = Settings::new(command_line_args.clone());
        let world_grid_size =
            Instance::calculate_bounding_box_size(&settings) * WORLD_GRID_SIZE_MULTIPLIER;
        let world_grid_vertices = Vertex::create_vertices_for_world_grid(world_grid_size);
        let mut instance_manager = InstanceManager::new(&settings);
        let (
            camera,
            projection,
            camera_controller,
            camera_uniform,
            camera_buffer,
            camera_bind_group_layout,
            camera_bind_group,
        ) = Camera::setup_camera(config, device);
        let initial_sort_time = instance_manager.sort_by_distance_to_camera(&camera);
        let depth_texture = Texture::create_depth_texture(device, config, "depth_texture");
        let (buffers, cube_num_indices, bounding_box_num_indices) = AppBuffers::new(
            device,
            &settings,
            config,
            &instance_manager,
            camera_buffer,
            &world_grid_vertices,
        );
        let shaders = AppShaders::new(device);
        let render_layouts = AppRenderLayouts::new(device, &camera_bind_group_layout);
        let pipelines = AppRenderPipelines::new(device, config, &shaders, &render_layouts);
        let text_renderer = TextRenderManager::new(
            device,
            config,
            queue,
            scale_factor,
            settings.font_size,
            settings.line_height_multiplier,
            &settings.initial_color_type,
            &settings.color_method,
        );

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &render_layouts.bind_group,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffers.uniform.as_entire_binding(),
            }],
            label: None,
        });

        App {
            bind_group,
            instance_manager,
            cube_num_indices,
            camera,
            last_sort_camera: camera,
            projection,
            camera_controller,
            camera_uniform,
            camera_bind_group,
            mouse_pressed: false,
            settings,
            depth_texture,
            update_queue: UpdateQueue::default(),
            bounding_box_num_indices,
            world_grid_vertices,
            buffers,
            pipelines,
            text_renderer,
            last_sort_time: initial_sort_time,
            simulation_state: SimulationState::Paused,
            last_simulation_time: Duration::default(),
            last_cpu_time: Duration::default(),
            last_user_input_instant: Instant::now(),
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
                if self.last_user_input_instant.elapsed().as_millis()
                    < MIN_USER_INPUT_INTERVAL.into()
                {
                    log::debug!(
                        "User input too frequent. Ignoring {:?}, took: {:?},",
                        logical_key,
                        self.last_user_input_instant.elapsed(),
                    );
                    return;
                }
                self.camera_controller
                    .process_keyboard(logical_key.clone(), state);
                if state == ElementState::Pressed {
                    if let Key::Character(s) = logical_key {
                        if s.as_str() == "p" {
                            self.settings.toggle_bounding_box();
                        } else if s.as_str() == "n" {
                            if self.simulation_state == SimulationState::Active {
                                log::warn!("Cannot alter transparency when simulation is active");
                            } else {
                                self.settings.set_transparency(
                                    self.settings.transparency - TRANSPARENCY_STEP_SIZE,
                                    &mut self.update_queue,
                                );
                            }
                        } else if s.as_str() == "m" {
                            if self.simulation_state == SimulationState::Active {
                                log::warn!("Cannot alter transparency when simulation is active");
                            } else {
                                self.settings.set_transparency(
                                    self.settings.transparency + TRANSPARENCY_STEP_SIZE,
                                    &mut self.update_queue,
                                );
                            }
                        } else if s.as_str() == "k" {
                            if self.simulation_state == SimulationState::Active {
                                log::warn!("Cannot alter space between instances when simulation is active");
                            } else {
                                self.settings.set_space_between_instances(
                                    self.settings.space_between_instances
                                        - SPACE_BETWEEN_INSTANCES_STEP_SIZE,
                                    &mut self.update_queue,
                                );
                            }
                        } else if s.as_str() == "l" {
                            if self.simulation_state == SimulationState::Active {
                                log::warn!("Cannot alter space between instances when simulation is active");
                            } else {
                                self.settings.set_space_between_instances(
                                    self.settings.space_between_instances
                                        + SPACE_BETWEEN_INSTANCES_STEP_SIZE,
                                    &mut self.update_queue,
                                );
                            }
                        } else if s.as_str() == "o" {
                            match self.simulation_state {
                                SimulationState::Active => {
                                    self.simulation_state = SimulationState::Paused;
                                    log::info!("Simulation paused");
                                }
                                SimulationState::Paused => {
                                    self.simulation_state = SimulationState::Active;
                                    log::info!("Simulation active");
                                }
                                SimulationState::Stable => {
                                    log::warn!(
                                        "Simulation has reached a stable state. Cannot be resumed. Nothing to simulate."
                                    )
                                }
                            }
                        } else if s.as_str() == "i" {
                            self.settings.toggle_world_grid();
                        } else if s.as_str() == "u" {
                            self.settings.next_simulation_mode();
                        }
                    } else if let Key::Named(key) = logical_key {
                        if key == NamedKey::PageUp {
                            if self.simulation_state == SimulationState::Active {
                                log::warn!("Cannot alter domain size when simulation is active");
                            } else {
                                self.settings.set_domain_size(
                                    self.settings.domain_size + DOMAIN_SIZE_STEP_SIZE,
                                    &mut self.update_queue,
                                );
                            }
                        } else if key == NamedKey::PageDown {
                            if self.simulation_state == SimulationState::Active {
                                log::warn!("Cannot alter domain size when simulation is active");
                            } else {
                                self.settings.set_domain_size(
                                    self.settings.domain_size - DOMAIN_SIZE_STEP_SIZE,
                                    &mut self.update_queue,
                                );
                            }
                        } else if key == NamedKey::Home {
                            let current_font_size = self.text_renderer.font_metrics.font_size;
                            if current_font_size < MAX_FONT_SIZE {
                                let new_font_size = current_font_size + FONT_SIZE_STEP_SIZE;
                                self.text_renderer.update_font_size(new_font_size);
                                log::info!("Font size increased to: {}", new_font_size);
                            } else {
                                log::warn!("Font size is already at maximum: {}", MAX_FONT_SIZE);
                            }
                        } else if key == NamedKey::End {
                            let current_font_size = self.text_renderer.font_metrics.font_size;
                            if current_font_size > MIN_FONT_SIZE {
                                let new_font_size = current_font_size - FONT_SIZE_STEP_SIZE;
                                self.text_renderer.update_font_size(new_font_size);
                                log::info!("Font size decreased to: {}", new_font_size);
                            } else {
                                log::warn!("Font size is already at minimum: {}", MIN_FONT_SIZE);
                            }
                        } else if key == NamedKey::Insert {
                            if self.text_renderer.line_height_multiplier
                                < MAX_LINE_HEIGHT_MULTIPLIER
                            {
                                let new_line_height_multiplier =
                                    self.text_renderer.line_height_multiplier
                                        + LINE_HEIGHT_MULTIPLIER_STEP_SIZE;
                                self.text_renderer
                                    .update_line_height_multiplier(new_line_height_multiplier);
                                log::info!(
                                    "Line height multiplier increased to: {}",
                                    new_line_height_multiplier
                                );
                            } else {
                                log::warn!(
                                    "Line height multiplier is already at maximum: {}",
                                    MAX_LINE_HEIGHT_MULTIPLIER
                                );
                            }
                        } else if key == NamedKey::Delete {
                            if self.text_renderer.line_height_multiplier
                                > MIN_LINE_HEIGHT_MULTIPLIER
                            {
                                let new_line_height_multiplier =
                                    self.text_renderer.line_height_multiplier
                                        - LINE_HEIGHT_MULTIPLIER_STEP_SIZE;
                                self.text_renderer
                                    .update_line_height_multiplier(new_line_height_multiplier);
                                log::info!(
                                    "Line height multiplier decreased to: {}",
                                    new_line_height_multiplier
                                );
                            } else {
                                log::warn!(
                                    "Line height multiplier is already at minimum: {}",
                                    MIN_LINE_HEIGHT_MULTIPLIER
                                );
                            }
                        }
                    }
                }
                self.last_user_input_instant = Instant::now();
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

    fn update(&mut self, dt: f32, avg_fps: f32, queue: &wgpu::Queue, device: &wgpu::Device) {
        let start = std::time::Instant::now();
        self.update_camera(dt, queue);
        self.transparency_update();
        self.queue_update(queue, device);
        if let Some(simulation_time) = self.simulation_update() {
            self.last_cpu_time = simulation_time + start.elapsed();
            self.last_simulation_time = simulation_time;
        };
        if self.simulation_state != SimulationState::Active {
            self.last_cpu_time = start.elapsed();
        }
        // For any Scheduled updates
        self.queue_update(queue, device);
        self.text_renderer.update(
            dt,
            avg_fps,
            self.last_cpu_time,
            self.last_simulation_time,
            self.last_sort_time,
            self.simulation_state.clone(),
            self.settings.simulation_tick_rate,
            &self.settings.simulation_rules,
        );
    }

    fn resize(
        &mut self,
        config: &wgpu::SurfaceConfiguration,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let mx_total = generate_matrix(config.width as f32 / config.height as f32);
        let mx_ref: &[f32; 16] = mx_total.as_ref();
        self.projection.resize(config.width, config.height);
        self.depth_texture =
            texture::Texture::create_depth_texture(device, config, "depth_texture");
        self.text_renderer.update_viewport_size(config, queue);
        queue.write_buffer(&self.buffers.uniform, 0, bytemuck::cast_slice(mx_ref));
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
            render_pass.set_pipeline(&self.pipelines.main_simulation);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.buffers.cube_vertex.slice(..));
            render_pass.set_vertex_buffer(1, self.buffers.simulation_instances.slice(..));
            render_pass
                .set_index_buffer(self.buffers.cube_index.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(
                0..self.cube_num_indices,
                0,
                0..self.instance_manager.num_flattened_instances() as _,
            );
            render_pass.pop_debug_group();
            render_pass.insert_debug_marker("Draw!");

            // render text
            self.text_renderer
                .render(device, queue, &mut render_pass, self.settings.debug_mode);
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
                render_pass.set_pipeline(&self.pipelines.world_grid);
                render_pass.set_vertex_buffer(0, self.buffers.world_grid.slice(..));
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
                render_pass.set_pipeline(&self.pipelines.world_grid_overlay);
                render_pass.set_vertex_buffer(0, self.buffers.world_grid.slice(..));
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

        if let Some(ref pipeline_wire) = self.pipelines.bounding_box {
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
                        .set_vertex_buffer(0, self.buffers.bounding_box_vertex.slice(..));
                    wireframe_render_pass
                        .set_vertex_buffer(1, self.buffers.bounding_box_instance.slice(..));
                    wireframe_render_pass.set_index_buffer(
                        self.buffers.bounding_box_index.slice(..),
                        wgpu::IndexFormat::Uint16,
                    );
                    wireframe_render_pass.draw_indexed(0..self.bounding_box_num_indices, 0, 0..1);
                }

                queue.submit(Some(encoder.finish()));
            }
        }
    }

    fn trim_atlas(&mut self) {
        self.text_renderer.trim_atlas();
    }
}

pub fn main() {
    let args: CommandLineArgs = CommandLineArgs::parse();
    if args.subcommand.is_some() {
        // no need to check what it is as only one subcommand exists
        Color::print_color_help();
        return;
    }
    init_logger(args.debug);
    pollster::block_on(cellular_automata_3d::framework::start::<App>(
        "3D Cellular Automata",
        args,
    ));
}
