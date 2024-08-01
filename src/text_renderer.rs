use std::time::Duration;

use crate::{
    constants::DEPTH_FORMAT,
    simulation::{ColorMethod, ColorType, SimulationRules, SimulationState},
    utils::Color,
};

enum TextRenderLevel {
    Debug,
    Info,
    Warning,
    Error,
}

impl TextRenderLevel {
    pub fn glyphon_color(&self) -> glyphon::Color {
        match self {
            TextRenderLevel::Debug => Color::Green.to_glyphon_color(),
            TextRenderLevel::Info => Color::White.to_glyphon_color(),
            TextRenderLevel::Warning => Color::Orange.to_glyphon_color(),
            TextRenderLevel::Error => Color::Red.to_glyphon_color(),
        }
    }
}

struct TextRenderBuffers {
    pub fps_buffer: glyphon::Buffer,
    pub info_buffer: glyphon::Buffer,
    pub simulation_state_text_buffer: glyphon::Buffer,
    pub debug_buffer: glyphon::Buffer,
    pub calculated_offsets: Vec<f32>,
    pub num_line_list: Vec<u8>,
}

impl TextRenderBuffers {
    pub fn new(
        font_system: &mut glyphon::FontSystem,
        font_metrics: glyphon::Metrics,
        physical_width: f32,
        physical_height: f32,
    ) -> Self {
        // The number of lines in each buffer, currently 1 fps, 4 + 4 for survival info, 1 simulation state, 3 debug lines
        // TODO: make this more intuitive to work on
        let num_line_list = vec![1, 9, 1, 3];
        Self {
            fps_buffer: Self::create_buffer(
                font_system,
                font_metrics,
                physical_width,
                physical_height,
                "Loading...",
            ),
            info_buffer: Self::create_buffer(
                font_system,
                font_metrics,
                physical_width,
                physical_height,
                "Loading...",
            ),
            debug_buffer: Self::create_buffer(
                font_system,
                font_metrics,
                physical_width,
                physical_height,
                "Loading...",
            ),
            simulation_state_text_buffer: Self::create_buffer(
                font_system,
                font_metrics,
                physical_width,
                physical_height,
                "Loading...",
            ),
            calculated_offsets: Vec::new(),
            num_line_list,
        }
    }

    fn calculate_vertical_offsets(&mut self, font_metrics: glyphon::Metrics) {
        let top_padding = 10.0;
        let mut calculated_offsets = Vec::new();
        let mut current_offset = top_padding;
        for i in 0..self.num_line_list.len() {
            calculated_offsets.push(current_offset);
            current_offset += font_metrics.line_height * self.num_line_list[i] as f32;
        }
        self.calculated_offsets = calculated_offsets;
    }

    fn update_font_metrics(
        &mut self,
        font_system: &mut glyphon::FontSystem,
        font_metrics: glyphon::Metrics,
    ) {
        self.fps_buffer.set_metrics(font_system, font_metrics);
        self.info_buffer.set_metrics(font_system, font_metrics);
        self.simulation_state_text_buffer
            .set_metrics(font_system, font_metrics);
        self.debug_buffer.set_metrics(font_system, font_metrics);
        self.calculate_vertical_offsets(font_metrics);
    }

    fn create_buffer(
        font_system: &mut glyphon::FontSystem,
        font_metrics: glyphon::Metrics,
        physical_width: f32,
        physical_height: f32,
        default_text: &str,
    ) -> glyphon::Buffer {
        let mut buffer = glyphon::Buffer::new(font_system, font_metrics);
        buffer.set_size(font_system, Some(physical_width), Some(physical_height));
        buffer.set_text(
            font_system,
            default_text,
            glyphon::Attrs::new().family(glyphon::Family::Monospace),
            glyphon::Shaping::Advanced,
        );
        buffer
    }
}

pub struct TextRenderManager {
    font_system: glyphon::FontSystem,
    swash_cache: glyphon::SwashCache,
    viewport: glyphon::Viewport,
    text_renderer: glyphon::TextRenderer,
    atlas: glyphon::TextAtlas,
    fps_level: TextRenderLevel,
    simulation_state_level: TextRenderLevel,
    buffers: TextRenderBuffers,
    pub font_metrics: glyphon::Metrics,
    pub line_height_multiplier: f32,
    color_method: String,
    // TODO: This messes up lifetimes make it work
    // text_areas: Vec<glyphon::TextArea>,
    // text_attributes: glyphon::Attrs,
}

impl TextRenderManager {
    pub fn new(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        queue: &wgpu::Queue,
        scale_factor: f64,
        font_size: f32,
        line_height_multiplier: f32,
        initial_color_type: &ColorType,
        color_method: &ColorMethod,
    ) -> Self {
        let mut font_system = glyphon::FontSystem::new();
        let swash_cache = glyphon::SwashCache::new();
        let cache = glyphon::Cache::new(device);
        let mut viewport = glyphon::Viewport::new(device, &cache);
        let mut atlas = glyphon::TextAtlas::new(device, queue, &cache, config.format);

        let text_renderer = glyphon::TextRenderer::new(
            &mut atlas,
            device,
            wgpu::MultisampleState::default(),
            Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
        );

        let font_metrics = glyphon::Metrics::new(font_size, font_size * line_height_multiplier);

        let physical_width = (config.width as f64 * scale_factor) as f32;
        let physical_height = (config.height as f64 * scale_factor) as f32;

        viewport.update(
            queue,
            glyphon::Resolution {
                width: config.width,
                height: config.height,
            },
        );

        let mut buffers = TextRenderBuffers::new(
            &mut font_system,
            font_metrics,
            physical_width,
            physical_height,
        );
        buffers.calculate_vertical_offsets(font_metrics);

        Self {
            font_system,
            swash_cache,
            viewport,
            text_renderer,
            atlas,
            fps_level: TextRenderLevel::Warning,
            simulation_state_level: TextRenderLevel::Info,
            buffers,
            font_metrics,
            line_height_multiplier,
            color_method: color_method.to_formatted_string(initial_color_type),
        }
    }

    pub fn update_viewport_size(
        &mut self,
        config: &wgpu::SurfaceConfiguration,
        queue: &wgpu::Queue,
    ) {
        self.viewport.update(
            queue,
            glyphon::Resolution {
                width: config.width,
                height: config.height,
            },
        );
    }

    pub fn render<'a>(
        &'a mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        render_pass: &mut wgpu::RenderPass<'a>,
        debug_mode: bool,
    ) {
        let fps_text_area = glyphon::TextArea {
            buffer: &self.buffers.fps_buffer,
            left: 10.0,
            top: self.buffers.calculated_offsets[0],
            scale: 1.0,
            bounds: glyphon::TextBounds::default(),
            default_color: self.fps_level.glyphon_color(),
        };
        let info_text_area = glyphon::TextArea {
            buffer: &self.buffers.info_buffer,
            left: 10.0,
            top: self.buffers.calculated_offsets[1],
            scale: 1.0,
            bounds: glyphon::TextBounds::default(),
            default_color: TextRenderLevel::Info.glyphon_color(),
        };
        let simulation_state_text_area = glyphon::TextArea {
            buffer: &self.buffers.simulation_state_text_buffer,
            left: 10.0,
            top: self.buffers.calculated_offsets[2],
            scale: 1.0,
            bounds: glyphon::TextBounds::default(),
            default_color: self.simulation_state_level.glyphon_color(),
        };
        let debug_text_area = glyphon::TextArea {
            buffer: &self.buffers.debug_buffer,
            left: 10.0,
            top: self.buffers.calculated_offsets[3],
            scale: 1.0,
            bounds: glyphon::TextBounds::default(),
            default_color: TextRenderLevel::Debug.glyphon_color(),
        };

        let text_areas = if debug_mode {
            vec![
                fps_text_area,
                info_text_area,
                simulation_state_text_area,
                debug_text_area,
            ]
        } else {
            vec![fps_text_area, info_text_area, simulation_state_text_area]
        };

        self.text_renderer
            .prepare(
                device,
                queue,
                &mut self.font_system,
                &mut self.atlas,
                &self.viewport,
                text_areas,
                &mut self.swash_cache,
            )
            .unwrap();

        self.text_renderer
            .render(&self.atlas, &self.viewport, render_pass)
            .unwrap();
    }

    pub fn trim_atlas(&mut self) {
        self.atlas.trim();
    }

    pub fn update_font_size(&mut self, new_font_size: f32) {
        self.font_metrics =
            glyphon::Metrics::new(new_font_size, self.line_height_multiplier * new_font_size);
        self.buffers
            .update_font_metrics(&mut self.font_system, self.font_metrics);
    }

    pub fn update_line_height_multiplier(&mut self, new_line_height_multiplier: f32) {
        // round to one decimal place
        let new_line_height_multiplier = (new_line_height_multiplier * 10.0).round() / 10.0;
        self.font_metrics = glyphon::Metrics::new(
            self.font_metrics.font_size,
            self.font_metrics.font_size * new_line_height_multiplier,
        );
        self.buffers
            .update_font_metrics(&mut self.font_system, self.font_metrics);
        self.line_height_multiplier = new_line_height_multiplier;
    }

    pub fn update(
        &mut self,
        frame_time: f32,
        fps: f32,
        cpu_time: Duration,
        last_simulation_time: Duration,
        last_sort_time: Duration,
        simulation_state: SimulationState,
        simulation_tick_rate: u16,
        simulation_rules: &SimulationRules,
    ) {
        if fps < 30.0 {
            self.fps_level = TextRenderLevel::Error;
        } else if fps < 60.0 {
            self.fps_level = TextRenderLevel::Warning;
        } else {
            self.fps_level = TextRenderLevel::Info;
        }

        match simulation_state {
            SimulationState::Active => self.simulation_state_level = TextRenderLevel::Info,
            SimulationState::Paused => self.simulation_state_level = TextRenderLevel::Warning,
            SimulationState::Stable => self.simulation_state_level = TextRenderLevel::Debug,
        }

        let fps_text = if frame_time < 1.0 {
            format!("Frame time {:05.3}µs ({:.1} FPS)", frame_time * 1000.0, fps)
        } else {
            format!("Frame time {:05.3}ms ({:.1} FPS)", frame_time, fps)
        };
        let cpu_time_text = if cpu_time.as_millis() < 1 {
            format!("CPU time {:05.3}µs", cpu_time.as_micros() as f64)
        } else {
            format!("CPU time {:05.3}ms", cpu_time.as_millis() as f64)
        };
        let sort_time_text = if last_sort_time.as_millis() < 1 {
            format!(
                "Last Sort time {:05.3}µs",
                last_sort_time.as_micros() as f64
            )
        } else {
            format!(
                "Last Sort time {:05.3}ms",
                last_sort_time.as_millis() as f64
            )
        };
        let simulation_time_text = if last_simulation_time.as_millis() < 1 {
            format!(
                "Last Simulation time {:.2}µs",
                last_simulation_time.as_micros()
            )
        } else {
            format!(
                "Last Simulation time {:.2}ms",
                last_simulation_time.as_millis()
            )
        };
        let simulation_state_text = match simulation_state {
            SimulationState::Active => "Simulation Active",
            SimulationState::Paused => "Simulation Paused",
            SimulationState::Stable => "Simulation Stable, nothing to simulate",
        };
        let debug_text = format!(
            "{}\n{}\n{}",
            cpu_time_text, simulation_time_text, sort_time_text
        );
        let info_text = format!(
            "Font Size: {}\nLine Height Multiplier: {}\nColor Method: {}\nSimulation Rules: {}\nSimulation Tick Rate: {}ms",
            self.font_metrics.font_size, self.line_height_multiplier, self.color_method, simulation_rules, simulation_tick_rate
        );

        self.buffers.fps_buffer.set_text(
            &mut self.font_system,
            &fps_text,
            glyphon::Attrs::new().family(glyphon::Family::Monospace),
            glyphon::Shaping::Advanced,
        );
        self.buffers.info_buffer.set_text(
            &mut self.font_system,
            &info_text,
            glyphon::Attrs::new().family(glyphon::Family::Monospace),
            glyphon::Shaping::Advanced,
        );
        self.buffers.simulation_state_text_buffer.set_text(
            &mut self.font_system,
            simulation_state_text,
            glyphon::Attrs::new().family(glyphon::Family::Monospace),
            glyphon::Shaping::Advanced,
        );
        self.buffers.debug_buffer.set_text(
            &mut self.font_system,
            &debug_text,
            glyphon::Attrs::new().family(glyphon::Family::Monospace),
            glyphon::Shaping::Advanced,
        );
    }
}
