use crate::{
    constants::{
        CUBE_SIZE, DEFAULT_DOMAIN_SIZE, DEFAULT_FONT_SIZE, DEFAULT_LINE_HEIGHT_MULTIPLIER,
        DEFAULT_NOISE_AMOUNT, DEFAULT_SIMULATION_TICK_RATE, DEFAULT_SPAWN_SIZE,
        DEFAULT_TRANSPARENCY, FRAME_COUNTER_UPDATE_INTERVAL, MIN_SPACE_BETWEEN_INSTANCES,
    },
    simulation::{ColorMethod, ColorType, SimulationRules},
    utils::{CommandLineArgs, UpdateEnum, UpdateQueue, Validator},
};
use std::{sync::Arc, time::Instant};
use wgpu::Surface;
use winit::{
    dpi::PhysicalSize,
    event::{DeviceEvent, Event, KeyEvent, StartCause, WindowEvent},
    event_loop::{EventLoop, EventLoopWindowTarget},
    keyboard::{Key, NamedKey},
    window::Window,
};

pub trait App: 'static + Sized {
    const SRGB: bool = true;

    fn optional_features() -> wgpu::Features {
        wgpu::Features::empty()
    }

    fn required_features() -> wgpu::Features {
        wgpu::Features::empty()
    }

    fn required_downlevel_capabilities() -> wgpu::DownlevelCapabilities {
        wgpu::DownlevelCapabilities {
            flags: wgpu::DownlevelFlags::empty(),
            shader_model: wgpu::ShaderModel::Sm5,
            ..wgpu::DownlevelCapabilities::default()
        }
    }

    fn required_limits() -> wgpu::Limits {
        wgpu::Limits::downlevel_webgl2_defaults() // These downlevel limits will allow the code to run on all possible hardware
    }

    fn init(
        config: &wgpu::SurfaceConfiguration,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        command_line_args: &CommandLineArgs,
        scale_factor: f64,
    ) -> Self;

    fn resize(
        &mut self,
        config: &wgpu::SurfaceConfiguration,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    );

    fn update(&mut self, dt: f32, avg_fps: f32, queue: &wgpu::Queue, device: &wgpu::Device);

    fn update_window_event(&mut self, event: WindowEvent);

    fn update_device_event(&mut self, event: DeviceEvent);

    fn render(&mut self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue);

    fn trim_atlas(&mut self);
}

struct EventLoopWrapper {
    event_loop: EventLoop<()>,
    window: Arc<Window>,
}

impl EventLoopWrapper {
    pub fn new(title: &str) -> Self {
        let event_loop = EventLoop::new().unwrap();
        let mut builder = winit::window::WindowBuilder::new();
        builder = builder.with_title(title);
        let window = Arc::new(builder.build(&event_loop).unwrap());

        Self { event_loop, window }
    }
}

/// Wrapper type which manages the surface and surface configuration.
///
/// As surface usage varies per platform, wrapping this up cleans up the event loop code.
struct SurfaceWrapper {
    surface: Option<wgpu::Surface<'static>>,
    config: Option<wgpu::SurfaceConfiguration>,
}

impl SurfaceWrapper {
    /// Create a new surface wrapper with no surface or configuration.
    fn new() -> Self {
        Self {
            surface: None,
            config: None,
        }
    }

    /// Called when an event which matches [`Self::start_condition`] is received.
    ///
    /// On all native platforms, this is where we create the surface.
    ///
    /// Additionally, we configure the surface based on the (now valid) window size.
    fn resume(&mut self, context: &AppContext, window: Arc<Window>, srgb: bool) {
        // Window size is only actually valid after we enter the event loop.
        let window_size = window.inner_size();
        let width = window_size.width.max(1);
        let height = window_size.height.max(1);

        log::debug!("Surface resume {window_size:?}");

        // We didn't create the surface in pre_adapter, so we need to do so now.

        self.surface = Some(context.instance.create_surface(window).unwrap());

        // From here on, self.surface should be Some.

        let surface = self.surface.as_ref().unwrap();

        // Get the default configuration,
        let mut config = surface
            .get_default_config(&context.adapter, width, height)
            .expect("Surface isn't supported by the adapter.");
        if srgb {
            // Not all platforms (WebGPU) support sRGB swapchains, so we need to use view formats
            let view_format = config.format.add_srgb_suffix();
            config.view_formats.push(view_format);
        } else {
            // All platforms support non-sRGB swapchains, so we can just use the format directly.
            let format = config.format.remove_srgb_suffix();
            config.format = format;
            config.view_formats.push(format);
        };

        surface.configure(&context.device, &config);
        self.config = Some(config);
    }

    /// Resize the surface, making sure to not resize to zero.
    fn resize(&mut self, context: &AppContext, size: PhysicalSize<u32>) {
        log::debug!("Surface resize {size:?}");

        let config = self.config.as_mut().unwrap();
        config.width = size.width.max(1);
        config.height = size.height.max(1);
        let surface = self.surface.as_ref().unwrap();
        surface.configure(&context.device, config);
    }

    /// Acquire the next surface texture.
    fn acquire(&mut self, context: &AppContext) -> wgpu::SurfaceTexture {
        let surface = self.surface.as_ref().unwrap();

        match surface.get_current_texture() {
            Ok(frame) => frame,
            // If we timed out, just try again
            Err(wgpu::SurfaceError::Timeout) => surface
                .get_current_texture()
                .expect("Failed to acquire next surface texture!"),
            Err(
                // If the surface is outdated, or was lost, reconfigure it.
                wgpu::SurfaceError::Outdated
                | wgpu::SurfaceError::Lost
                // If OutOfMemory happens, reconfiguring may not help, but we might as well try
                | wgpu::SurfaceError::OutOfMemory,
            ) => {
                surface.configure(&context.device, self.config());
                surface
                    .get_current_texture()
                    .expect("Failed to acquire next surface texture!")
            }
        }
    }

    fn get(&self) -> Option<&Surface> {
        self.surface.as_ref()
    }

    fn config(&self) -> &wgpu::SurfaceConfiguration {
        self.config.as_ref().unwrap()
    }

    fn toggle_vsync(&mut self, context: &AppContext) {
        let config = self.config.as_mut().unwrap();
        config.present_mode = match config.present_mode {
            wgpu::PresentMode::Fifo => wgpu::PresentMode::Immediate,
            wgpu::PresentMode::Immediate => wgpu::PresentMode::Fifo,
            _ => wgpu::PresentMode::Fifo,
        };
        if config.present_mode == wgpu::PresentMode::Fifo {
            log::info!("VSync enabled");
        } else {
            log::info!("VSync disabled");
        }
        let surface = self.surface.as_ref().unwrap();
        surface.configure(&context.device, config);
    }
}

/// Context containing global wgpu resources.
struct AppContext {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}
impl AppContext {
    /// Initializes the app context.
    async fn init_async<E: App>(surface: &mut SurfaceWrapper, _window: Arc<Window>) -> Self {
        log::debug!("Initializing wgpu...");

        let backends = wgpu::util::backend_bits_from_env().unwrap_or_default();
        let dx12_shader_compiler = wgpu::util::dx12_shader_compiler_from_env().unwrap_or_default();
        let gles_minor_version = wgpu::util::gles_minor_version_from_env().unwrap_or_default();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            flags: wgpu::InstanceFlags::from_build_config().with_env(),
            dx12_shader_compiler,
            gles_minor_version,
        });
        let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, surface.get())
            .await
            .expect("No suitable GPU adapters found on the system!");

        let adapter_info = adapter.get_info();
        log::debug!("Using {} ({:?})", adapter_info.name, adapter_info.backend);

        let optional_features = E::optional_features();
        let required_features = E::required_features();
        let adapter_features = adapter.features();
        assert!(
            adapter_features.contains(required_features),
            "Adapter does not support required features for this app: {:?}",
            required_features - adapter_features
        );

        let required_downlevel_capabilities = E::required_downlevel_capabilities();
        let downlevel_capabilities = adapter.get_downlevel_capabilities();
        assert!(
            downlevel_capabilities.shader_model >= required_downlevel_capabilities.shader_model,
            "Adapter does not support the minimum shader model required to run this app: {:?}",
            required_downlevel_capabilities.shader_model
        );
        assert!(
            downlevel_capabilities
                .flags
                .contains(required_downlevel_capabilities.flags),
            "Adapter does not support the downlevel capabilities required to run this app: {:?}",
            required_downlevel_capabilities.flags - downlevel_capabilities.flags
        );

        // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the surface.
        let needed_limits = E::required_limits().using_resolution(adapter.limits());

        let trace_dir = std::env::var("WGPU_TRACE");
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: (optional_features & adapter_features) | required_features,
                    required_limits: needed_limits,
                },
                trace_dir.ok().as_ref().map(std::path::Path::new),
            )
            .await
            .expect("Unable to find a suitable GPU adapter!");

        Self {
            instance,
            adapter,
            device,
            queue,
        }
    }
}

struct FrameCounter {
    // Instant of the last time we printed the frame time.
    last_printed_instant: Instant,
    // Number of frames since the last time we printed the frame time.
    frame_count: u32,
    frame_time: f32,
    avg_fps: f32,
    report_interval: f32,
}

impl FrameCounter {
    fn new(report_interval: f32) -> Self {
        Self {
            last_printed_instant: Instant::now(),
            frame_count: 0,
            frame_time: 0.0,
            avg_fps: 0.0,
            report_interval,
        }
    }

    fn update(&mut self) {
        self.frame_count += 1;
        let new_instant = Instant::now();
        let elapsed_secs = (new_instant - self.last_printed_instant).as_secs_f32();
        if elapsed_secs > self.report_interval {
            let elapsed_ms = elapsed_secs * 1000.0;
            let frame_time = elapsed_ms / self.frame_count as f32;
            let fps = self.frame_count as f32 / elapsed_secs;
            // log::info!("Frame time {:.2}ms ({:.1} FPS)", frame_time, fps);

            self.last_printed_instant = new_instant;
            self.frame_count = 0;
            self.frame_time = frame_time;
            self.avg_fps = fps;
        }
    }

    fn get_frame_time(&self) -> f32 {
        self.frame_time
    }

    fn get_avg_fps(&self) -> f32 {
        self.avg_fps
    }
}

pub async fn start<E: App>(title: &str, command_line_args: CommandLineArgs) {
    log::info!("Initializing...");
    let window_loop = EventLoopWrapper::new(title);
    let mut surface = SurfaceWrapper::new();
    let context = AppContext::init_async::<E>(&mut surface, window_loop.window.clone()).await;
    let mut frame_counter = FrameCounter::new(FRAME_COUNTER_UPDATE_INTERVAL);

    // We wait to create the app until we have a valid surface.
    let mut app = None;

    log::debug!("Entering event loop...");
    let loop_result = EventLoop::run(
        window_loop.event_loop,
        move |event: Event<()>, target: &EventLoopWindowTarget<()>| {
            match event {
                Event::NewEvents(StartCause::Init) => {
                    surface.resume(&context, window_loop.window.clone(), E::SRGB);

                    // If we haven't created the app yet, do so now.
                    if app.is_none() {
                        app = Some(E::init(
                            surface.config(),
                            &context.adapter,
                            &context.device,
                            &context.queue,
                            &command_line_args,
                            window_loop.window.scale_factor(),
                        ));
                        log::info!("Initialization complete");
                    }
                }
                Event::Suspended => {
                    println!("Suspended");
                }
                Event::DeviceEvent {
                    event,
                    .. // We're not using device_id currently
                } => app.as_mut().unwrap().update_device_event(event),
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::Resized(size) => {
                        surface.resize(&context, size);
                        app.as_mut().unwrap().resize(
                            surface.config(),
                            &context.device,
                            &context.queue,
                        );

                        window_loop.window.request_redraw();
                    }
                    WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                logical_key: Key::Named(NamedKey::Escape),
                                ..
                            },
                        ..
                    }
                    | WindowEvent::CloseRequested => {
                        target.exit();
                    }
                    WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                logical_key: Key::Character(s),
                                state: winit::event::ElementState::Pressed,
                                ..
                            },
                        ..
                    } if s == "r" || s == "v" => {
                        if s == "r" {
                            println!("{:#?}", context.instance.generate_report());
                        } else if s == "v" {
                            surface.toggle_vsync(&context);
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        // On MacOS, currently redraw requested comes in _before_ Init does.
                        // If this happens, just drop the requested redraw on the floor.
                        //
                        // See https://github.com/rust-windowing/winit/issues/3235 for some discussion
                        if app.is_none() {
                            return;
                        }

                        frame_counter.update();
                        app.as_mut().unwrap().update(frame_counter.get_frame_time(), frame_counter.get_avg_fps(), &context.queue, &context.device);
                        let frame = surface.acquire(&context);
                        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
                            format: Some(surface.config().view_formats[0]),
                            ..wgpu::TextureViewDescriptor::default()
                        });

                        app
                            .as_mut()
                            .unwrap()
                            .render(&view, &context.device, &context.queue);

                        frame.present();
                        app.as_mut().unwrap().trim_atlas();
                        window_loop.window.request_redraw();
                    }
                    _ => app.as_mut().unwrap().update_window_event(event),
                },
                _ => {}
            }
        },
    );
    log::debug!("Event loop ended with result {:?}", loop_result);
    log::info!("Exiting...");
}

#[derive(Debug, PartialEq)]
pub enum SimulationMode {
    SingleThreaded,
    #[cfg(feature = "multithreading")]
    MultiThreaded,
    // Gpu,
}

// Allow Dead Code as multithreading uses simulation mode but when rayon is not enabled it is not used
#[allow(dead_code)]
pub struct Settings {
    pub bounding_box_active: bool,
    pub world_grid_active: bool,
    pub help_gui_active: bool,
    pub domain_size: u32,
    pub domain_max_dist_from_center: f32,
    pub cube_size: f32,
    pub space_between_instances: f32,
    pub transparency: f32,
    pub simulation_tick_rate: u16,
    pub last_simulation_tick: std::time::Instant,
    pub spawn_size: u8,
    pub noise_amount: u8,
    pub simulation_rules: SimulationRules,
    pub color_method: ColorMethod,
    pub simulation_mode: SimulationMode,
    pub debug_mode: bool,
    pub font_size: f32,
    pub line_height_multiplier: f32,
    pub initial_color_type: ColorType,
}

impl Default for Settings {
    fn default() -> Self {
        Self::new(CommandLineArgs::default())
    }
}

impl Settings {
    pub fn new(command_line_args: CommandLineArgs) -> Self {
        let simulation_rules = SimulationRules::parse_rules(command_line_args.rules.as_deref());
        let (color_method, initial_color_type) = ColorMethod::parse_method(
            command_line_args.color_method.as_deref(),
            DEFAULT_TRANSPARENCY,
        );
        let simulation_tick_rate = Validator::validate_simulation_tick_rate(
            command_line_args
                .simulation_tick_rate
                .unwrap_or(DEFAULT_SIMULATION_TICK_RATE),
        );
        let domain_size = Validator::validate_domain_size(
            command_line_args.domain_size.unwrap_or(DEFAULT_DOMAIN_SIZE),
        );
        let domain_magnitude = Self::calculate_max_distance_from_center(&domain_size);
        let spawn_size = Validator::validate_spawn_size(
            command_line_args
                .initial_spawn_size
                .unwrap_or(DEFAULT_SPAWN_SIZE),
            domain_size as u8,
        );
        // Add 1 to avoid no noise on setting noise to 1 as it will calculate random offset from 0 to 1
        // which is not visible in the grid, hence we need at least 2 to see the noise
        let noise_amount = Validator::validate_noise_amount(
            command_line_args
                .noise_level
                .unwrap_or(DEFAULT_NOISE_AMOUNT),
        );

        let bounding_box_active = false;
        let world_grid_active = !command_line_args.disable_world_grid;
        let help_gui_active = false;
        let last_simulation_tick = std::time::Instant::now();

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
            help_gui_active,
            domain_size,
            space_between_instances: MIN_SPACE_BETWEEN_INSTANCES,
            domain_max_dist_from_center: domain_magnitude,
            cube_size: CUBE_SIZE,
            transparency: DEFAULT_TRANSPARENCY,
            simulation_tick_rate,
            last_simulation_tick,
            spawn_size,
            noise_amount,
            simulation_rules,
            color_method,
            simulation_mode,
            debug_mode: command_line_args.debug,
            font_size: DEFAULT_FONT_SIZE,
            line_height_multiplier: DEFAULT_LINE_HEIGHT_MULTIPLIER,
            initial_color_type,
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
        self.domain_max_dist_from_center =
            Self::calculate_max_distance_from_center(&self.domain_size);
        log::info!("Domain size set to: {}", self.domain_size);
    }

    pub fn calculate_max_distance_from_center(domain_size: &u32) -> f32 {
        // calculate the distance from 0,0,0 to domain_size, domain_size, domain_size, used for ColorMethod::DistToCenter
        ((domain_size.pow(2) * 3) as f32).sqrt()
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
