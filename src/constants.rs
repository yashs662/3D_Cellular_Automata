#![cfg_attr(rustfmt, rustfmt_skip)]
use crate::utils::Color;
use std::f32::consts::FRAC_PI_2;

pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.5,
    0.0, 0.0, 0.0, 1.0,
);
pub const ANGLE_THRESHOLD_FOR_SORTING: f32 = 0.1;
pub const CUBE_SIZE: f32 = 1.0;
pub const DEFAULT_CAMERA_SENSITIVITY: f32 = 0.7;
pub const DEFAULT_CAMERA_SPEED: f32 = 15.0;
pub const DEFAULT_COLORS: [Color; 2] = [Color::Red, Color::Orange];
pub const DEFAULT_FONT_SIZE: f32 = 15.0;
pub const DEFAULT_LINE_HEIGHT_MULTIPLIER: f32 = 1.2;
pub const DEFAULT_TRANSPARENCY: f32 = 0.5;
pub const DEFAULT_SIMULATION_TICK_RATE: u16 = 25;
pub const DEFAULT_DOMAIN_SIZE: u32 = 40;
pub const DEFAULT_SPAWN_SIZE: u8 = 12;
pub const DEFAULT_NOISE_AMOUNT: u8 = 1; // 1 is the same as 0, i.e. essentially no noise
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
pub const DOMAIN_SIZE_STEP_SIZE: u32 = 2;
pub const FONT_SIZE_STEP_SIZE: f32 = 1.0;
pub const FRAME_COUNTER_UPDATE_INTERVAL: f32 = 0.5;
pub const LINE_HEIGHT_MULTIPLIER_STEP_SIZE: f32 = 0.1;
pub const MAX_DOMAIN_SIZE: u32 = 100;
pub const MAX_FONT_SIZE: f32 = 100.0;
pub const MAX_LINE_HEIGHT_MULTIPLIER: f32 = 2.0;
pub const MAX_NOISE_AMOUNT: u8 = 10;
pub const MAX_SIMULATION_TICK_RATE: u16 = 1000;
pub const MAX_SPACE_BETWEEN_INSTANCES: f32 = 10.0;
pub const MAX_TRANSPARENCY: f32 = 1.0;
pub const MIN_DOMAIN_SIZE: u32 = 2;
pub const MIN_FONT_SIZE: f32 = 10.0;
pub const MIN_LINE_HEIGHT_MULTIPLIER: f32 = 0.1;
pub const MIN_SIMULATION_TICK_RATE: u16 = 1;
pub const MIN_SPACE_BETWEEN_INSTANCES: f32 = 0.1;
pub const MIN_SPAWN_SIZE: u8 = 2;
pub const MIN_TRANSPARENCY: f32 = 0.05;
pub const SAFE_FRAC_PI_2: f32 = FRAC_PI_2 - 0.0001;
pub const SIMULATION_ACTIVE_TEXT: &str = "Simulation Active";
pub const SIMULATION_PAUSED_TEXT: &str = "Simulation Paused";
pub const SIMULATION_STABLE_TEXT: &str = "Simulation Stable, nothing to simulate";
pub const SPACE_BETWEEN_INSTANCES_STEP_SIZE: f32 = 0.05;
pub const TRANSLATION_THRESHOLD_FOR_SORTING: f32 = 10.0;
pub const TRANSPARENCY_STEP_SIZE: f32 = 0.05;
pub const WORLD_GRID_SIZE_MULTIPLIER: f32 = 1.5;
