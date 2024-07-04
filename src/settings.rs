use winit::keyboard::NamedKey;

use crate::{
    constants::{
        CUBE_SIZE, DEFAULT_DOMAIN_SIZE, DEFAULT_FONT_SIZE, DEFAULT_LINE_HEIGHT_MULTIPLIER,
        DEFAULT_NOISE_AMOUNT, DEFAULT_SIMULATION_TICK_RATE, DEFAULT_SPAWN_SIZE,
        DEFAULT_TRANSPARENCY, MIN_SPACE_BETWEEN_INSTANCES,
    },
    framework::SimulationMode,
    simulation::{ColorMethod, ColorType, SimulationRules},
    utils::{CommandLineArgs, UpdateEnum, UpdateQueue, Validator},
};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputKey {
    MoveForward,
    MoveLeft,
    MoveBackward,
    MoveRight,
    MoveUp,
    MoveDown,
    ToggleVsync,
    ToggleBoundingBox,
    ToggleSimulationState,
    ToggleSimulationMode,
    ToggleWorldGrid,
    IncreaseTransparency,
    DecreaseTransparency,
    IncreaseSpaceBetweenInstances,
    DecreaseSpaceBetweenInstances,
    IncreaseDomainSize,
    DecreaseDomainSize,
    IncreaseFontSize,
    DecreaseFontSize,
    IncreaseLineHeightMultiplier,
    DecreaseLineHeightMultiplier,
    Unknown,
}

impl InputKey {
    pub fn is_movement_key(&self) -> bool {
        match self {
            InputKey::MoveForward
            | InputKey::MoveLeft
            | InputKey::MoveBackward
            | InputKey::MoveRight
            | InputKey::MoveUp
            | InputKey::MoveDown => true,
            _ => false,
        }
    }

    pub fn requires_paused_simulation(&self) -> bool {
        match self {
            | InputKey::IncreaseTransparency
            | InputKey::DecreaseTransparency
            | InputKey::IncreaseSpaceBetweenInstances
            | InputKey::DecreaseSpaceBetweenInstances
            | InputKey::IncreaseDomainSize
            | InputKey::DecreaseDomainSize => true,
            _ => false,
        }
    }
}

impl From<winit::keyboard::Key> for InputKey {
    fn from(key: winit::keyboard::Key) -> Self {
        use winit::keyboard::Key;
        match key {
            Key::Character(s) => match s.as_str() {
                "w" => InputKey::MoveForward,
                "a" => InputKey::MoveLeft,
                "s" => InputKey::MoveBackward,
                "d" => InputKey::MoveRight,
                "v" => InputKey::ToggleVsync,
                "b" => InputKey::ToggleBoundingBox,
                "p" => InputKey::ToggleSimulationState,
                "m" => InputKey::ToggleSimulationMode,
                "z" => InputKey::ToggleWorldGrid,
                "t" => InputKey::IncreaseTransparency,
                "g" => InputKey::DecreaseTransparency,
                "i" => InputKey::IncreaseSpaceBetweenInstances,
                "k" => InputKey::DecreaseSpaceBetweenInstances,
                "o" => InputKey::IncreaseDomainSize,
                "l" => InputKey::DecreaseDomainSize,
                "u" => InputKey::IncreaseFontSize,
                "j" => InputKey::DecreaseFontSize,
                "y" => InputKey::IncreaseLineHeightMultiplier,
                "h" => InputKey::DecreaseLineHeightMultiplier,
                _ => {
                    log::debug!("Unknown key: {}", s);
                    InputKey::Unknown
                }
            },
            Key::Named(named_key) => match named_key {
                NamedKey::ArrowUp => InputKey::MoveForward,
                NamedKey::ArrowLeft => InputKey::MoveLeft,
                NamedKey::ArrowDown => InputKey::MoveBackward,
                NamedKey::ArrowRight => InputKey::MoveRight,
                NamedKey::Space => InputKey::MoveUp,
                NamedKey::Shift => InputKey::MoveDown,
                _ => {
                    log::debug!("Unknown key: {:?}", named_key);
                    InputKey::Unknown
                }
            },
            _ => {
                log::debug!("Unknown key: {:?}", key);
                InputKey::Unknown
            }
        }
    }
}
