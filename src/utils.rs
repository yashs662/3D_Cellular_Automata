use cgmath::Vector4;
use clap::Parser;

/// Initialize logging in platform dependant ways.
pub fn init_logger(debug_mode: bool) {
    let filter_level = if debug_mode {
        log::LevelFilter::Debug
    } else {
        log::LevelFilter::Info
    };

    env_logger::builder()
        .filter_level(filter_level)
        // We keep wgpu at Error level, as it's very noisy.
        .filter_module("wgpu_core", log::LevelFilter::Info)
        .filter_module("wgpu_hal", log::LevelFilter::Error)
        .filter_module("naga", log::LevelFilter::Error)
        .parse_default_env()
        .init();
}

#[derive(Parser, Debug, Clone)]
#[command(
    version,
    about,
    long_about = "A 3D cellular automata simulation using wgpu. To use any of the aliases make sure to use them with the long flag. Example: --dwg and not -dwg."
)]
#[derive(Default)]
pub struct CommandLineArgs {
    /// The rules for the simulation in the format S/B/N/M"
    /// where S is the survival rules, B is the birth rules, N is the number of states, and M is the neighbor method.
    /// survival and birth can be in the format 0-2,4,6-11,13-17,21-26/9-10,16,23-24. where - is a range between <x>-<y> and , is a list of numbers.
    /// N is a number between 1 and 255 and the last is either M or V for Von Neumann or Moore neighborhood.
    /// Example: 4/4/5/M
    #[arg(short, long)]
    pub rules: Option<String>,
    /// Enable Debug mode
    /// This will enable debug mode which will print out more information to the console.
    #[arg(short, long)]
    pub debug: bool,
    /// Set the simulation tick rate in milliseconds
    #[arg(long, visible_alias = "str")]
    pub simulation_tick_rate: Option<u16>,
    /// Noise level
    /// This will set the noise level for the initial spawn of the simulation.
    #[arg(short, long, visible_alias = "nl")]
    pub noise_level: Option<u8>,
    /// Simulation Domain Size
    /// This will set the size of the simulation domain, any instances that are outside the initial spawn size will be spawned as dead.
    #[arg(long, visible_alias = "ds")]
    pub domain_size: Option<u32>,
    /// Initial Spawn Size
    /// This will set the initial spawn size for the simulation which is less than or equal to the size of the simulation.
    #[arg(short, long, visible_alias = "iss")]
    pub initial_spawn_size: Option<u8>,
    /// Disable world Grid
    #[arg(short = 'w', long, visible_alias = "dwg")]
    pub disable_world_grid: bool,
    /// Color Method to use
    /// This will determine how the colors are calculated for the simulation.
    /// accepts hex values or color range from 0-1 or 0-255 in three channels red green adn blue (no Alpha). Append with H for hex, 1 for 0-1, and 255 for 0-255.
    /// The options are S (Single), SL (StateLerp), DTC (DistToCenter), N (Neighbor).
    /// Examples:
    /// S/H/#FF0000, S/1/1.0,0.0,0.0, S/255/255,0,0
    /// SL/H/#FF0000/#00FF00, SL/1/1.0,0.0,0.0/0.0,1.0,0.0, SL/255/255,0,0/0,255,0
    /// DTC/H/#FF0000/#00FF00, DTC/1/1.0,0.0,0.0/0.0,1.0,0.0, DTC/255/255,0,0/0,255,0
    /// N/H/#FF0000/#00FF00, N/1/1.0,0.0,0.0/0.0,1.0,0.0, N/255/255,0,0/0,255,0
    #[arg(short, long, visible_alias = "cm")]
    pub color_method: Option<String>,
}

pub struct Validator;

impl Validator {
    pub fn validate_simulation_tick_rate(simulation_tick_rate: u16) -> u16 {
        match simulation_tick_rate {
            x if x < 1 => {
                log::warn!("Simulation tick rate cannot be less than 1");
                1
            }
            x if x > 1000 => {
                log::warn!("Simulation tick rate cannot be more than 1000 for a smooth experience");
                1000
            }
            _ => simulation_tick_rate,
        }
    }

    pub fn validate_domain_size(domain_size: u32) -> u32 {
        match domain_size {
            x if x < 3 => {
                log::warn!("Domain size cannot be less than 3");
                3
            }
            x if x > 100 => {
                log::warn!("Domain size cannot be more than 100");
                100
            }
            _ => domain_size,
        }
    }

    pub fn validate_spawn_size(spawn_size: u8, domain_size: u8) -> u8 {
        match spawn_size {
            x if x < 2 => {
                log::warn!("Spawn size cannot be less than 2");
                2
            }
            x if x > domain_size => {
                log::warn!("Spawn size cannot be more than the Domain size",);
                domain_size
            }
            x if x % 2 != 0 => {
                log::warn!("Spawn size must be an even number",);
                spawn_size - 1
            }
            _ => spawn_size,
        }
    }

    pub fn validate_noise_amount(noise_amount: u8) -> u8 {
        if noise_amount > 10 {
            log::warn!("Noise level cannot be more than 10");
            10
        } else {
            noise_amount
        }
    }

    pub fn validate_space_between_instances(space_between_instances: f32) -> f32 {
        if space_between_instances < 0.1 {
            log::warn!(
                "Space between instances cannot be less than 0.1 to avoid overlapping instances"
            );
            0.1
        } else if space_between_instances > 10.0 {
            log::warn!("Space between instances cannot be more than 10.0");
            10.0
        } else {
            (space_between_instances * 100.0).round() / 100.0
        }
    }

    pub fn validate_transparency(transparency: f32) -> f32 {
        if transparency < 0.0 {
            log::warn!("Transparency cannot be less than 0.0");
            0.0
        } else if transparency > 1.0 {
            log::warn!("Transparency cannot be more than 1.0");
            1.0
        } else {
            (transparency * 100.0).round() / 100.0
        }
    }
}

pub fn lerp_color(color_1: Vector4<f32>, color_2: Vector4<f32>, dt: f32) -> Vector4<f32> {
    let dt = dt.max(0.0).min(1.0);
    let inv = 1.0 - dt;
    let lerped = [
        color_1.x * dt + color_2.x * inv,
        color_1.y * dt + color_2.y * inv,
        color_1.z * dt + color_2.z * inv,
        color_1.w * dt + color_2.w * inv,
    ];
    Vector4::new(lerped[0], lerped[1], lerped[2], lerped[3])
}
