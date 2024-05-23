use cellular_automata_3d::{
    camera::{setup_camera, Camera, CameraController, CameraUniform, Projection},
    constants::DEPTH_FORMAT,
    framework::Settings,
    instance::{Instance, InstanceManager, InstanceRaw},
    texture::{self, Texture},
    utils::{init_logger, Color, CommandLineArgs, UpdateEnum, UpdateQueue},
    vertex::Vertex,
};
use cgmath::InnerSpace;
use clap::Parser;
use colored::Colorize;
use std::{borrow::Cow, f32::consts, mem};
use strum::IntoEnumIterator;
use wgpu::{util::DeviceExt, PipelineCompilationOptions};
use winit::{
    event::{DeviceEvent, ElementState, KeyEvent, MouseButton, WindowEvent},
    keyboard::{Key, NamedKey},
};

struct AppBuffers {
    pub cube_vertex: wgpu::Buffer,
    pub cube_index: wgpu::Buffer,
    pub uniform: wgpu::Buffer,
    pub instances: wgpu::Buffer,
    pub camera: wgpu::Buffer,
    pub bounding_box_vertex: wgpu::Buffer,
    pub bounding_box_index: wgpu::Buffer,
    pub bounding_box_instance: wgpu::Buffer,
    pub world_grid: wgpu::Buffer,
}

struct AppRenderPipelines {
    pub main_simulation: wgpu::RenderPipeline,
    pub bounding_box: Option<wgpu::RenderPipeline>,
    pub world_grid: wgpu::RenderPipeline,
    pub world_grid_overlay: wgpu::RenderPipeline,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct WorldGridRaw {
    pub position: [f32; 4],
    pub texture_coord: [f32; 2],
    pub fade_distance: f32,
}

unsafe impl bytemuck::Pod for WorldGridRaw {}
unsafe impl bytemuck::Zeroable for WorldGridRaw {}

impl WorldGridRaw {
    pub fn new(vertex: Vertex, fade_distance: f32) -> Self {
        WorldGridRaw {
            position: vertex.position,
            texture_coord: vertex.texture_coord,
            fade_distance,
        }
    }

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<WorldGridRaw>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

struct App {
    bind_group: wgpu::BindGroup,
    instance_manager: InstanceManager,
    num_indices: u32,
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
            &self.buffers.camera,
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
        if self.update_queue.is_empty() {
            return;
        }
        for update in self.update_queue.iter() {
            match update {
                UpdateEnum::Transparency => {
                    if self.instance_manager.update_transparency(&self.settings) {
                        scheduled_updates.push(UpdateEnum::CreateNewInstanceBuffer);
                    }
                    scheduled_updates.push(UpdateEnum::SortInstances);
                    scheduled_updates.push(UpdateEnum::UpdateInstanceBuffer);
                }
                UpdateEnum::SpaceBetweenInstances => {
                    if self
                        .instance_manager
                        .update_space_between_instances(&self.settings)
                    {
                        scheduled_updates.push(UpdateEnum::CreateNewInstanceBuffer);
                    }
                    scheduled_updates.push(UpdateEnum::UpdateBoundingBox);
                    scheduled_updates.push(UpdateEnum::UpdateWorldGrid);
                    scheduled_updates.push(UpdateEnum::SortInstances);
                    scheduled_updates.push(UpdateEnum::UpdateInstanceBuffer);
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
                    scheduled_updates.push(UpdateEnum::UpdateWorldGrid);
                    scheduled_updates.push(UpdateEnum::UpdateBoundingBox);
                }
                UpdateEnum::NumInstancesDecreased => {
                    if let Some(update_enum) = self
                        .instance_manager
                        .decrease_domain_size(&self.settings, &self.camera)
                    {
                        scheduled_updates.push(update_enum);
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
                        .update_buffer(gpu_queue, &self.buffers.instances);
                }
                UpdateEnum::CreateNewInstanceBuffer => {
                    self.buffers.instances = self.instance_manager.create_new_buffer(device);
                }
                UpdateEnum::UpdateWorldGrid => {
                    let new_size = Instance::calculate_bounding_box_size(&self.settings) + 20.0;
                    self.world_grid_vertices = Vertex::create_vertices_for_world_grid(new_size);
                    let world_grid_raw = self
                        .world_grid_vertices
                        .iter()
                        .map(|vertex| {
                            WorldGridRaw::new(
                                *vertex,
                                self.settings.domain_size as f32
                                    / self.settings.max_domain_size as f32,
                            )
                        })
                        .collect::<Vec<_>>();
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
        command_line_args: &CommandLineArgs,
    ) -> Self {
        let settings = Settings::new(command_line_args.clone());
        // Create the vertex and index buffers
        let (vertex_data, index_data) = Vertex::create_vertices(settings.cube_size);
        let (bounding_box_vertex_data, bounding_box_index_data) =
            Vertex::create_vertices(Instance::calculate_bounding_box_size(&settings));
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
        let instance_buffer = instance_manager.create_new_buffer(device);

        let bounding_box_instance = Instance::create_bounding_box(&settings);
        let bounding_box_instance_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Bounding Box Instance Buffer"),
                contents: bytemuck::cast_slice(&[bounding_box_instance.to_raw()]),
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
        let world_grid_size = Instance::calculate_bounding_box_size(&settings) + 20.0;
        let world_grid_vertices = Vertex::create_vertices_for_world_grid(world_grid_size);
        let world_grid_raw = world_grid_vertices
            .iter()
            .map(|vertex| {
                WorldGridRaw::new(
                    *vertex,
                    settings.domain_size as f32 / settings.max_domain_size as f32,
                )
            })
            .collect::<Vec<_>>();
        let world_grid_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("World Grid Vertex Buffer"),
            contents: bytemuck::cast_slice(&world_grid_raw),
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
                buffers: &[WorldGridRaw::desc()],
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
                    buffers: &[WorldGridRaw::desc()],
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

        let buffers = AppBuffers {
            cube_vertex: vertex_buf,
            cube_index: index_buf,
            uniform: uniform_buf,
            instances: instance_buffer,
            camera: camera_buffer,
            bounding_box_vertex: bounding_box_vertex_buf,
            bounding_box_index: bounding_box_index_buf,
            bounding_box_instance: bounding_box_instance_buffer,
            world_grid: world_grid_buffer,
        };

        let pipelines = AppRenderPipelines {
            main_simulation: pipeline,
            bounding_box: pipeline_wire,
            world_grid: world_grid_pipeline,
            world_grid_overlay: world_grid_pipeline_depth,
        };

        // Done
        App {
            bind_group,
            instance_manager,
            num_indices: index_data.len() as u32,
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
            bounding_box_num_indices: bounding_box_index_data.len() as u32,
            world_grid_vertices,
            buffers,
            pipelines,
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
            render_pass.set_vertex_buffer(1, self.buffers.instances.slice(..));
            render_pass
                .set_index_buffer(self.buffers.cube_index.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(
                0..self.num_indices,
                0,
                0..self.instance_manager.num_flattened_instances() as _,
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
}

pub fn main() {
    let args: CommandLineArgs = CommandLineArgs::parse();
    if args.subcommand.is_some() {
        // no need to check what it is as only one subcommand exists
        let color_iterator = Color::iter();
        println!(
            "{:^12} - {:^18} - {:^15} - {:^7}",
            "Color", "Float  (0.0 - 1.1)", "Int (0 - 255)", "Hex"
        );
        for color in color_iterator {
            let color_value = color.value();
            let color_0_255 = color.to_rgb_0_255();
            let color_hex = color.to_hex();
            println!(
                "{}",
                format!(
                    "{:12} - ({:4}, {:4}, {:4}) - ({:3}, {:3}, {:3}) - {}",
                    color,
                    color_value[0],
                    color_value[1],
                    color_value[2],
                    color_0_255[0],
                    color_0_255[1],
                    color_0_255[2],
                    color_hex
                )
                .on_truecolor(color_0_255[0], color_0_255[1], color_0_255[2])
            );
        }
        return;
    }
    init_logger(args.debug);
    cellular_automata_3d::framework::run::<App>("3D Cellular Automata", args);
}
