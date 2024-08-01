use crate::{
    constants::{
        DEPTH_FORMAT, FRAGMENT_SHADER_ENTRY_POINT, MAX_DOMAIN_SIZE, VERTEX_SHADER_ENTRY_POINT,
    },
    instance::{Instance, InstanceManager, InstanceRaw, WorldGridRaw},
    settings::Settings,
    utils::generate_matrix,
    vertex::Vertex,
};
use std::borrow::Cow;
use wgpu::{util::DeviceExt, PipelineCompilationOptions};

pub struct AppBuffers {
    pub cube_vertex: wgpu::Buffer,
    pub cube_index: wgpu::Buffer,
    pub uniform: wgpu::Buffer,
    pub simulation_instances: wgpu::Buffer,
    pub camera: wgpu::Buffer,
    pub bounding_box_vertex: wgpu::Buffer,
    pub bounding_box_index: wgpu::Buffer,
    pub bounding_box_instance: wgpu::Buffer,
    pub world_grid: wgpu::Buffer,
}

impl AppBuffers {
    pub fn new(
        device: &wgpu::Device,
        settings: &Settings,
        config: &wgpu::SurfaceConfiguration,
        instance_manager: &InstanceManager,
        camera_buffer: wgpu::Buffer,
        world_grid_vertices: &[Vertex],
    ) -> (Self, u32, u32) {
        let (cube_vertex_data, cube_index_data) = Vertex::create_vertices(settings.cube_size);
        let (bounding_box_vertex_data, bounding_box_index_data) =
            Vertex::create_vertices(Instance::calculate_bounding_box_size(settings));
        let world_grid_raw = WorldGridRaw::new(
            world_grid_vertices,
            settings.domain_size as f32 / MAX_DOMAIN_SIZE as f32,
        );
        let aspect_ratio = config.width as f32 / config.height as f32;
        let mx_total = generate_matrix(aspect_ratio);
        let mx_ref: &[f32; 16] = mx_total.as_ref();

        let cube_vertex = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&cube_vertex_data),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let cube_index = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&cube_index_data),
            usage: wgpu::BufferUsages::INDEX,
        });

        let bounding_box_vertex = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bounding Box Vertex Buffer"),
            contents: bytemuck::cast_slice(&bounding_box_vertex_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let bounding_box_index = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bounding Box Index Buffer"),
            contents: bytemuck::cast_slice(&bounding_box_index_data),
            usage: wgpu::BufferUsages::INDEX,
        });

        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(mx_ref),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let simulation_instance = instance_manager.create_new_buffer(device);

        let bounding_box_instance = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bounding Box Instance Buffer"),
            contents: bytemuck::cast_slice(&[instance_manager.bounding_box_instance.to_raw()]),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let world_grid = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("World Grid Vertex Buffer"),
            contents: bytemuck::cast_slice(&world_grid_raw),
            usage: wgpu::BufferUsages::VERTEX,
        });

        (
            Self {
                cube_vertex,
                cube_index,
                uniform,
                simulation_instances: simulation_instance,
                camera: camera_buffer,
                bounding_box_vertex,
                bounding_box_index,
                bounding_box_instance,
                world_grid,
            },
            cube_index_data.len() as u32,
            bounding_box_index_data.len() as u32,
        )
    }
}

pub struct AppShaders {
    simulation: wgpu::ShaderModule,
    world_grid: wgpu::ShaderModule,
}

impl AppShaders {
    pub fn new(device: &wgpu::Device) -> Self {
        let simulation = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/shader.wgsl"))),
        });
        let world_grid = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "shaders/world_grid.wgsl"
            ))),
        });

        Self {
            simulation,
            world_grid,
        }
    }
}

pub struct AppRenderLayouts {
    pub pipeline: wgpu::PipelineLayout,
    pub bind_group: wgpu::BindGroupLayout,
}

impl AppRenderLayouts {
    pub fn new(device: &wgpu::Device, camera_bind_group_layout: &wgpu::BindGroupLayout) -> Self {
        let bind_group = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        let pipeline = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&bind_group, camera_bind_group_layout],
            push_constant_ranges: &[],
        });

        Self {
            pipeline,
            bind_group,
        }
    }
}

pub struct AppRenderPipelines {
    pub main_simulation: wgpu::RenderPipeline,
    pub bounding_box: Option<wgpu::RenderPipeline>,
    pub world_grid: wgpu::RenderPipeline,
    pub world_grid_overlay: wgpu::RenderPipeline,
}

impl AppRenderPipelines {
    pub fn new(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        shaders: &AppShaders,
        render_layouts: &AppRenderLayouts,
    ) -> Self {
        let cube_vertex_buffers = &[Vertex::desc(), InstanceRaw::desc()];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_layouts.pipeline),
            vertex: wgpu::VertexState {
                module: &shaders.simulation,
                entry_point: VERTEX_SHADER_ENTRY_POINT,
                buffers: cube_vertex_buffers,
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shaders.simulation,
                entry_point: FRAGMENT_SHADER_ENTRY_POINT,
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
            cache: None,
        });

        let pipeline_wire = if device
            .features()
            .contains(wgpu::Features::POLYGON_MODE_LINE)
        {
            let pipeline_wire = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&render_layouts.pipeline),
                vertex: wgpu::VertexState {
                    module: &shaders.simulation,
                    entry_point: VERTEX_SHADER_ENTRY_POINT,
                    buffers: cube_vertex_buffers,
                    compilation_options: PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shaders.simulation,
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
                cache: None,
            });
            Some(pipeline_wire)
        } else {
            None
        };

        let world_grid_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("World Grid Pipeline"),
            layout: Some(&render_layouts.pipeline),
            vertex: wgpu::VertexState {
                module: &shaders.world_grid,
                entry_point: VERTEX_SHADER_ENTRY_POINT,
                buffers: &[WorldGridRaw::desc()],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shaders.world_grid,
                entry_point: FRAGMENT_SHADER_ENTRY_POINT,
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
            cache: None,
        });

        let world_grid_pipeline_depth =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("World Grid Depth Pipeline"),
                layout: Some(&render_layouts.pipeline),
                vertex: wgpu::VertexState {
                    module: &shaders.world_grid,
                    entry_point: VERTEX_SHADER_ENTRY_POINT,
                    buffers: &[WorldGridRaw::desc()],
                    compilation_options: PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shaders.world_grid,
                    entry_point: FRAGMENT_SHADER_ENTRY_POINT,
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
                cache: None,
            });

        Self {
            main_simulation: pipeline,
            bounding_box: pipeline_wire,
            world_grid: world_grid_pipeline,
            world_grid_overlay: world_grid_pipeline_depth,
        }
    }
}
