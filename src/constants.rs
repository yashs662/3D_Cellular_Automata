#![cfg_attr(rustfmt, rustfmt_skip)]
use std::f32::consts::FRAC_PI_2;

pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.5,
    0.0, 0.0, 0.0, 1.0,
);
pub const SAFE_FRAC_PI_2: f32 = FRAC_PI_2 - 0.0001;
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
pub const DEFAULT_COLORS: [[f32; 4]; 2] = [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]];

// TODO: Allow for predefined colors to be used from the CLI
pub const ALICE_BLUE   : (f32, f32, f32) = (0.94, 0.97, 1.0 );
pub const ANTIQUE_WHITE: (f32, f32, f32) = (0.98, 0.92, 0.84);
pub const AQUAMARINE   : (f32, f32, f32) = (0.49, 1.0 , 0.83);
pub const AZURE        : (f32, f32, f32) = (0.94, 1.0 , 1.0 );
pub const BEIGE        : (f32, f32, f32) = (0.96, 0.96, 0.86);
pub const BISQUE       : (f32, f32, f32) = (1.0 , 0.89, 0.77);
pub const BLACK        : (f32, f32, f32) = (0.0 , 0.0 , 0.0 );
pub const BLUE         : (f32, f32, f32) = (0.0 , 0.0 , 1.0 );
pub const CRIMSON      : (f32, f32, f32) = (0.86, 0.08, 0.24);
pub const CYAN         : (f32, f32, f32) = (0.0 , 1.0 , 1.0 );
pub const DARK_GRAY    : (f32, f32, f32) = (0.25, 0.25, 0.25);
pub const DARK_GREEN   : (f32, f32, f32) = (0.0 , 0.5 , 0.0 );
pub const FUCHSIA      : (f32, f32, f32) = (1.0 , 0.0 , 1.0 );
pub const GOLD         : (f32, f32, f32) = (1.0 , 0.84, 0.0 );
pub const GRAY         : (f32, f32, f32) = (0.5 , 0.5 , 0.5 );
pub const GREEN        : (f32, f32, f32) = (0.0 , 1.0 , 0.0 );
pub const INDIGO       : (f32, f32, f32) = (0.29, 0.0 , 0.51);
pub const LIME_GREEN   : (f32, f32, f32) = (0.2 , 0.8 , 0.2 );
pub const MAROON       : (f32, f32, f32) = (0.5 , 0.0 , 0.0 );
pub const MIDNIGHT_BLUE: (f32, f32, f32) = (0.1 , 0.1 , 0.44);
pub const NAVY         : (f32, f32, f32) = (0.0 , 0.0 , 0.5 );
pub const OLIVE        : (f32, f32, f32) = (0.5 , 0.5 , 0.0 );
pub const ORANGE       : (f32, f32, f32) = (1.0 , 0.65, 0.0 );
pub const ORANGE_RED   : (f32, f32, f32) = (1.0 , 0.27, 0.0 );
pub const PINK         : (f32, f32, f32) = (1.0 , 0.08, 0.58);
pub const PURPLE       : (f32, f32, f32) = (0.5 , 0.0 , 0.5 );
pub const RED          : (f32, f32, f32) = (1.0 , 0.0 , 0.0 );
pub const SALMON       : (f32, f32, f32) = (0.98, 0.5 , 0.45);
pub const SEA_GREEN    : (f32, f32, f32) = (0.18, 0.55, 0.34);
pub const SILVER       : (f32, f32, f32) = (0.75, 0.75, 0.75);
pub const TEAL         : (f32, f32, f32) = (0.0 , 0.5 , 0.5 );
pub const TOMATO       : (f32, f32, f32) = (1.0 , 0.39, 0.28);
pub const TURQUOISE    : (f32, f32, f32) = (0.25, 0.88, 0.82);
pub const VIOLET       : (f32, f32, f32) = (0.93, 0.51, 0.93);
pub const WHITE        : (f32, f32, f32) = (1.0 , 1.0 , 1.0 );
pub const YELLOW       : (f32, f32, f32) = (1.0 , 1.0 , 0.0 );
pub const YELLOW_GREEN : (f32, f32, f32) = (0.6 , 0.8 , 0.2 );
