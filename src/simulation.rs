use crate::{constants::DEFAULT_COLORS, utils::Color};
use cgmath::Vector4;
use std::str::FromStr;

#[derive(Clone, Debug, PartialEq)]
pub enum SimulationState {
    Active,
    Paused,
    Stable,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum NeighborMethod {
    /// 26 neighbors
    Moore,
    /// 6 neighbors
    VonNeumann,
}

impl NeighborMethod {
    pub fn get_neighbor_iter(&self) -> &'static [(i32, i32, i32)] {
        match self {
            NeighborMethod::VonNeumann => &VON_NEUMANN_NEIGHBORS[..],
            NeighborMethod::Moore => &MOORE_NEIGHBORS[..],
        }
    }

    pub fn total_num_neighbors(&self) -> usize {
        match self {
            NeighborMethod::VonNeumann => 6,
            NeighborMethod::Moore => 26,
        }
    }
}

pub static VON_NEUMANN_NEIGHBORS: [(i32, i32, i32); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, -1),
    (0, 0, 1),
];

pub static MOORE_NEIGHBORS: [(i32, i32, i32); 26] = [
    (-1, -1, -1),
    (0, -1, -1),
    (1, -1, -1),
    (-1, 0, -1),
    (0, 0, -1),
    (1, 0, -1),
    (-1, 1, -1),
    (0, 1, -1),
    (1, 1, -1),
    (-1, -1, 0),
    (0, -1, 0),
    (1, -1, 0),
    (-1, 0, 0),
    (1, 0, 0),
    (-1, 1, 0),
    (0, 1, 0),
    (1, 1, 0),
    (-1, -1, 1),
    (0, -1, 1),
    (1, -1, 1),
    (-1, 0, 1),
    (0, 0, 1),
    (1, 0, 1),
    (-1, 1, 1),
    (0, 1, 1),
    (1, 1, 1),
];

#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub enum CellStateEnum {
    #[default]
    Alive,
    Fading,
    Dead,
}

impl CellStateEnum {
    pub fn to_int(self) -> u8 {
        match self {
            CellStateEnum::Alive => 1,
            CellStateEnum::Fading => 2,
            CellStateEnum::Dead => 0,
        }
    }
}

#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct CellState {
    pub state: CellStateEnum,
    pub fade_level: u8,
}

impl CellState {
    pub fn dead() -> Self {
        CellState {
            state: CellStateEnum::Dead,
            fade_level: 0,
        }
    }
}

/// Color Method to use
/// This will determine how the colors are calculated for the simulation.
/// accepts hex values or color range from 0-1 or 0-255 in three channels red green adn blue (no Alpha).
/// Append with H for hex, 1 for 0-1, 255 for 0-255, and D for predefined colors. you can use predefined colors with
/// any option to mix and match but can only use predefined colors when using the D option.
/// The options are S (Single), SL (StateLerp), DTC (DistToCenter), N (Neighbor).
///
/// Examples:
/// S/H/#FF0000,
/// S/1/1.0,0.0,0.0,
/// S/255/255,0,0,
/// SL/H/#FF0000/#00FF00,
/// SL/1/1.0,0.0,0.0/0.0,1.0,0.0,
/// SL/255/255,0,0/0,255,0,
/// DTC/H/#FF0000/#00FF00,
/// DTC/1/1.0,0.0,0.0/0.0,1.0,0.0,
/// DTC/255/255,0,0/0,255,0,
/// N/H/#FF0000/#00FF00,
/// N/1/1.0,0.0,0.0/0.0,1.0,0.0,
/// N/255/255,0,0/0,255,0,
/// SL/D/Red/Green,
/// SL/H/#FF0000/Green
#[derive(Debug)]
enum ColorChannelEnum {
    Red,
    Green,
    Blue,
}

#[derive(Debug, PartialEq)]
pub enum ColorMethod {
    Single(Vector4<f32>),
    StateLerp(Vector4<f32>, Vector4<f32>),
    DistToCenter(Vector4<f32>, Vector4<f32>),
    Neighbor(Vector4<f32>, Vector4<f32>),
}

impl ColorMethod {
    fn new(default_transparency: f32) -> Self {
        ColorMethod::Single(Vector4::new(
            DEFAULT_COLORS[0][0],
            DEFAULT_COLORS[0][1],
            DEFAULT_COLORS[0][2],
            default_transparency,
        ))
    }
}

impl ColorMethod {
    pub fn to_int(&self) -> u32 {
        match self {
            ColorMethod::Single(_) => 0,
            ColorMethod::StateLerp(_, _) => 1,
            ColorMethod::DistToCenter(_, _) => 2,
            ColorMethod::Neighbor(_, _) => 3,
        }
    }

    pub fn parse_method(method: Option<&str>, default_transparency: f32) -> ColorMethod {
        if method.is_none() {
            log::warn!("No color method provided, using default method");
            return ColorMethod::new(default_transparency);
        }
        let method = method.unwrap();
        let parsed_method = {
            let parts: Vec<&str> = method.split('/').collect();
            if parts.len() < 3 {
                log::error!("Invalid color method format");
                log::warn!(
                    "Color method format must be 'method/type/color1/color2' where color2 is optional"
                );
                log::warn!("Using default method");
                return ColorMethod::new(default_transparency);
            }
            let method_type = parts[0].replace(' ', "");
            let method_type = method_type.as_str();
            let color_type = parts[1].replace(' ', "");
            let color_type = color_type.as_str();
            let color1 = parts[2].replace(' ', "");
            let color1 = color1.as_str();
            let color2 = parts.get(3).copied();

            match method_type {
                "S" => ColorMethod::Single(Self::parse_color(
                    color_type,
                    color1,
                    true,
                    default_transparency,
                )),
                "SL" | "DTC" | "N" => color2.map_or_else(
                    || {
                        log::error!(
                            "Color method '{}' requires two colors, only one provided",
                            method_type
                        );
                        log::warn!("Using default method");
                        return ColorMethod::new(default_transparency);
                    },
                    |color2| {
                        let color2 = color2.replace(' ', "");
                        let color2 = color2.as_str();

                        match method_type {
                            "SL" => ColorMethod::StateLerp(
                                Self::parse_color(color_type, color1, true, default_transparency),
                                Self::parse_color(color_type, color2, false, default_transparency),
                            ),
                            "DTC" => ColorMethod::DistToCenter(
                                Self::parse_color(color_type, color1, true, default_transparency),
                                Self::parse_color(color_type, color2, false, default_transparency),
                            ),
                            "N" => ColorMethod::Neighbor(
                                Self::parse_color(color_type, color1, true, default_transparency),
                                Self::parse_color(color_type, color2, false, default_transparency),
                            ),
                            _ => unreachable!(),
                        }
                    },
                ),
                _ => {
                    log::error!("Invalid color method, must be 'S', 'SL', 'DTC', or 'N'");
                    log::warn!("Using default method");
                    return ColorMethod::new(default_transparency);
                }
            }
        };
        log::debug!("Parsed color method: {:?}", parsed_method);
        parsed_method
    }

    fn parse_color(
        color_type: &str,
        color: &str,
        is_color_1: bool,
        default_transparency: f32,
    ) -> Vector4<f32> {
        match color_type {
            "H" => Self::parse_hex_color(color, is_color_1, default_transparency),
            "1" => Self::parse_0_1_color(color, is_color_1, default_transparency),
            "255" => Self::parse_0_255_color(color, is_color_1, default_transparency),
            "D" => Self::parse_predefined_color(color, is_color_1, default_transparency),
            _ => {
                log::error!("Invalid color type, must be 'H', '1', or '255'");
                log::warn!("Using default color");
                Self::get_default_color(is_color_1)
            }
        }
    }

    fn parse_predefined_color(
        color: &str,
        is_color_1: bool,
        default_transparency: f32,
    ) -> Vector4<f32> {
        let predefined_color = Color::from_str(color);
        if predefined_color.is_ok() {
            let color = predefined_color.unwrap().value();
            Vector4::new(color[0], color[1], color[2], default_transparency)
        } else {
            log::error!("{} No such color exists in Default Colors", color);
            log::warn!("Using default color");
            Self::get_default_color(is_color_1)
        }
    }

    fn parse_hex_color(color: &str, is_color_1: bool, default_transparency: f32) -> Vector4<f32> {
        if color.len() != 7 || !color.starts_with('#') {
            let predefined_color = Color::from_str(color);
            if predefined_color.is_ok() {
                let color = predefined_color.unwrap().value();
                return Vector4::new(color[0], color[1], color[2], default_transparency);
            } else {
                log::error!("Invalid color format");
                log::warn!("Color format must be '#RRGGBB' or a predefined color");
                log::warn!("Using default color");
                return Self::get_default_color(is_color_1);
            }
        }

        let color = color.trim_start_matches('#');
        let r = Self::parse_color_channel_for_hex(&color[0..2], is_color_1, ColorChannelEnum::Red);
        let g =
            Self::parse_color_channel_for_hex(&color[2..4], is_color_1, ColorChannelEnum::Green);
        let b = Self::parse_color_channel_for_hex(&color[4..6], is_color_1, ColorChannelEnum::Blue);

        Vector4::new(r, g, b, default_transparency)
    }

    fn parse_color_channel_for_hex(
        channel: &str,
        is_color_1: bool,
        channel_type: ColorChannelEnum,
    ) -> f32 {
        let parsed = u8::from_str_radix(channel, 16);
        if parsed.is_err() {
            log::error!("Invalid color format");
            log::warn!("Hex code '{}' is not valid", channel);
            log::warn!(
                "Using default color for channel {:?} of color{}",
                channel_type,
                if is_color_1 { "1" } else { "2" }
            );
            match channel_type {
                ColorChannelEnum::Red => return Self::get_default_color(is_color_1)[0],
                ColorChannelEnum::Green => return Self::get_default_color(is_color_1)[1],
                ColorChannelEnum::Blue => return Self::get_default_color(is_color_1)[2],
            }
        }
        parsed.unwrap() as f32 / 255.0
    }

    fn parse_color_channel_for_0_1(
        channel: &str,
        is_color_1: bool,
        channel_type: ColorChannelEnum,
    ) -> f32 {
        let parsed = channel.parse::<f32>();
        if parsed.is_err() {
            log::error!("Invalid color format");
            log::warn!("Color must be between '0.0-1.0'");
            log::warn!("Using default color");
            match channel_type {
                ColorChannelEnum::Red => return Self::get_default_color(is_color_1)[0],
                ColorChannelEnum::Green => return Self::get_default_color(is_color_1)[1],
                ColorChannelEnum::Blue => return Self::get_default_color(is_color_1)[2],
            }
        }
        let parsed = parsed.unwrap();
        if !(0.0..=1.0).contains(&parsed) {
            log::error!("Invalid color format");
            log::warn!("Color must be between '0.0-1.0'");
            log::warn!("Using default color");
            match channel_type {
                ColorChannelEnum::Red => Self::get_default_color(is_color_1)[0],
                ColorChannelEnum::Green => Self::get_default_color(is_color_1)[1],
                ColorChannelEnum::Blue => Self::get_default_color(is_color_1)[2],
            }
        } else {
            parsed
        }
    }

    fn parse_color_channel_for_0_255(
        channel: &str,
        is_color_1: bool,
        channel_type: ColorChannelEnum,
    ) -> f32 {
        let parsed = channel.parse::<u8>();
        if parsed.is_err() {
            log::error!("Invalid color format");
            log::warn!("Color must be between '0-255'");
            log::warn!("Using default color");
            match channel_type {
                ColorChannelEnum::Red => return Self::get_default_color(is_color_1)[0],
                ColorChannelEnum::Green => return Self::get_default_color(is_color_1)[1],
                ColorChannelEnum::Blue => return Self::get_default_color(is_color_1)[2],
            }
        }
        parsed.unwrap() as f32 / 255.0
    }

    fn get_default_color(is_color_1: bool) -> Vector4<f32> {
        if is_color_1 {
            Vector4::new(
                DEFAULT_COLORS[0][0],
                DEFAULT_COLORS[0][1],
                DEFAULT_COLORS[0][2],
                DEFAULT_COLORS[0][3],
            )
        } else {
            Vector4::new(
                DEFAULT_COLORS[1][0],
                DEFAULT_COLORS[1][1],
                DEFAULT_COLORS[1][2],
                DEFAULT_COLORS[1][3],
            )
        }
    }

    fn parse_0_1_color(color: &str, is_color_1: bool, default_transparency: f32) -> Vector4<f32> {
        let parts: Vec<&str> = color.split(',').collect();
        if parts.len() != 3 {
            let predefined_color = Color::from_str(color);
            if predefined_color.is_ok() {
                log::debug!("predefined color {:?}", predefined_color);
                let color = predefined_color.unwrap().value();
                return Vector4::new(color[0], color[1], color[2], default_transparency);
            } else {
                log::error!("Invalid color format");
                log::warn!(
                    "Color format must be 'R,G,B' in the range 0.0 - 1.0 <Float> or a predefined color"
                );
                log::warn!("Using default color");
                return Self::get_default_color(is_color_1);
            }
        }

        let r = Self::parse_color_channel_for_0_1(parts[0], is_color_1, ColorChannelEnum::Red);
        let g = Self::parse_color_channel_for_0_1(parts[1], is_color_1, ColorChannelEnum::Green);
        let b = Self::parse_color_channel_for_0_1(parts[2], is_color_1, ColorChannelEnum::Blue);

        Vector4::new(r, g, b, default_transparency)
    }

    fn parse_0_255_color(color: &str, is_color_1: bool, default_transparency: f32) -> Vector4<f32> {
        let parts: Vec<&str> = color.split(',').collect();
        if parts.len() != 3 {
            let predefined_color = Color::from_str(color);
            if predefined_color.is_ok() {
                log::debug!("predefined color {:?}", predefined_color);
                let color = predefined_color.unwrap().value();
                return Vector4::new(color[0], color[1], color[2], default_transparency);
            } else {
                log::error!("Invalid color format");
                log::warn!(
                    "Color format must be 'R,G,B' in the range 0 - 255 <Int> or a predefined color"
                );
                log::warn!("Using default color");
                return Self::get_default_color(is_color_1);
            }
        }

        let r = Self::parse_color_channel_for_0_255(parts[0], is_color_1, ColorChannelEnum::Red);
        let g = Self::parse_color_channel_for_0_255(parts[1], is_color_1, ColorChannelEnum::Green);
        let b = Self::parse_color_channel_for_0_255(parts[2], is_color_1, ColorChannelEnum::Blue);

        Vector4::new(r, g, b, default_transparency)
    }
}

#[derive(Debug)]
enum SimulationRuleParseEnum {
    Survival,
    Birth,
}

// Using the format from Softology's Blog (https://softologyblog.wordpress.com/2019/12/28/3d-cellular-automata-3/)
/// The rules for the simulation in the format S/B/N/M"
/// where S is the survival rules, B is the birth rules, N is the number of states, and M is the neighbor method.
/// survival and birth can be in the format 0-2,4,6-11,13-17,21-26/9-10,16,23-24. where - is a range between <x>-<y> and , is a list of numbers.
/// N is a number between 1 and 255 and the last is either M or V for Von Neumann or Moore neighborhood.
///
/// Example:
/// 4/4/5/M
/// 2,6,9/4,6,8-10/10/M
#[derive(Clone, Debug, PartialEq)]
pub struct SimulationRules {
    pub survival: Vec<u8>,
    pub birth: Vec<u8>,
    pub num_states: u8,
    pub neighbor_method: NeighborMethod,
}

impl Default for SimulationRules {
    fn default() -> Self {
        SimulationRules {
            survival: vec![4],
            birth: vec![4],
            num_states: 5,
            neighbor_method: NeighborMethod::Moore,
        }
    }
}

impl SimulationRules {
    pub fn new(
        survival: Vec<u8>,
        birth: Vec<u8>,
        num_states: u8,
        neighbor_method: NeighborMethod,
    ) -> Self {
        SimulationRules {
            survival,
            birth,
            num_states,
            neighbor_method,
        }
    }

    pub fn parse_rules(rules: Option<&str>) -> SimulationRules {
        if rules.is_none() {
            log::warn!("No rules provided, using default rules");
            return SimulationRules::default();
        }
        let rules = rules.unwrap();
        let parts: Vec<String> = rules.split('/').map(|part| part.replace(' ', "")).collect();
        if parts.len() != 4 {
            log::error!("Invalid rule format");
            log::warn!("Rule format must be 'survival/birth/num_states/neighbor_method'");
            log::warn!("Using default rules");
            return SimulationRules::default();
        }

        let survival = Self::parse_rule_parts(&parts[0], SimulationRuleParseEnum::Survival);
        if survival.is_err() {
            log::error!("Invalid survival rules");
            log::warn!("Using default rules");
            return SimulationRules::default();
        }
        let survival = survival.unwrap();
        let birth = Self::parse_rule_parts(&parts[1], SimulationRuleParseEnum::Birth);
        if birth.is_err() {
            log::error!("Invalid birth rules");
            log::warn!("Using default rules");
            return SimulationRules::default();
        }
        let birth = birth.unwrap();
        let num_states = parts[2].parse::<u8>().unwrap();
        let neighbor_method = match parts[3].as_str() {
            "M" => NeighborMethod::Moore,
            "V" => NeighborMethod::VonNeumann,
            _ => {
                log::error!(
                    "Invalid neighbor method, must be 'M' for Moore or 'V' for Von Neumann"
                );
                log::warn!("Using default rules");
                return SimulationRules::default();
            }
        };

        let parsed_rules = SimulationRules::new(survival, birth, num_states, neighbor_method);
        log::debug!("Parsed rules: {:?}", parsed_rules);
        parsed_rules
    }

    fn parse_rule_parts(
        rule_parts: &str,
        rule_to_parse: SimulationRuleParseEnum,
    ) -> Result<Vec<u8>, ()> {
        let parts: Vec<&str> = rule_parts.split(',').collect();
        if parts.len() == 1 && parts[0].is_empty() {
            log::debug!("{:?} parts {:?}", rule_to_parse, parts);
            return Ok(Vec::new());
        }

        let mut rules = Vec::new();
        for part in parts {
            if part.contains('-') {
                let range: Vec<&str> = part.split('-').collect();
                let start = range[0].parse::<u8>();
                let end = range[1].parse::<u8>();
                if start.is_err() || end.is_err() {
                    log::error!(
                        "Invalid rule format for part in {:?} rules: {}",
                        rule_to_parse,
                        part
                    );
                    log::warn!("Using default rules");
                    return Err(());
                }
                for i in start.unwrap()..=end.unwrap() {
                    rules.push(i);
                }
            } else {
                let rule = part.parse::<u8>();
                if rule.is_err() {
                    log::error!(
                        "Invalid rule format for part in {:?} rules: {}",
                        rule_to_parse,
                        part
                    );
                    log::warn!("Using default rules");
                    return Err(());
                }
                rules.push(rule.unwrap());
            }
        }

        if rules.len() != rules.iter().collect::<std::collections::HashSet<_>>().len() {
            log::error!("{:?} rules contain repeating values", rule_to_parse);
            log::warn!("Using default rules");
            return Err(());
        }

        Ok(rules)
    }
}
