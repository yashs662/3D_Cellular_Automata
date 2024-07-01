use crate::{constants::DEFAULT_COLORS, utils::Color};
use cgmath::Vector4;
use std::{fmt::Display, str::FromStr};

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

impl FromStr for NeighborMethod {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "M" => Ok(NeighborMethod::Moore),
            "V" => Ok(NeighborMethod::VonNeumann),
            _ => Err(()),
        }
    }
}

impl Display for NeighborMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NeighborMethod::Moore => write!(f, "M"),
            NeighborMethod::VonNeumann => write!(f, "V"),
        }
    }
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

    fn user_friendly_string(&self) -> &str {
        match self {
            NeighborMethod::VonNeumann => "Von Neumann",
            NeighborMethod::Moore => "Moore",
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
/// Append with H for hex, 1 for 0-1, 255 for 0-255, and PD for predefined colors. you can use predefined colors with
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
/// SL/PD/Red/Green,
/// SL/H/#FF0000/Green
#[derive(Default, Copy, Clone, Debug, PartialEq)]
pub enum ColorType {
    Hex,
    ZeroTo1,
    ZeroTo255,
    #[default]
    PreDefined,
}

impl FromStr for ColorType {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "H" => Ok(ColorType::Hex),
            "1" => Ok(ColorType::ZeroTo1),
            "255" => Ok(ColorType::ZeroTo255),
            "PD" => Ok(ColorType::PreDefined),
            _ => Err(()),
        }
    }
}

impl Display for ColorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ColorType::Hex => write!(f, "H"),
            ColorType::ZeroTo1 => write!(f, "1"),
            ColorType::ZeroTo255 => write!(f, "255"),
            ColorType::PreDefined => write!(f, "PD"),
        }
    }
}

#[derive(Default, PartialEq)]
pub enum ColorMethodType {
    #[default]
    Single,
    StateLerp,
    DistToCenter,
    Neighbor,
}

impl FromStr for ColorMethodType {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "S" => Ok(ColorMethodType::Single),
            "SL" => Ok(ColorMethodType::StateLerp),
            "DTC" => Ok(ColorMethodType::DistToCenter),
            "N" => Ok(ColorMethodType::Neighbor),
            _ => Err(()),
        }
    }
}

impl Display for ColorMethodType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ColorMethodType::Single => write!(f, "S"),
            ColorMethodType::StateLerp => write!(f, "SL"),
            ColorMethodType::DistToCenter => write!(f, "DTC"),
            ColorMethodType::Neighbor => write!(f, "N"),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum ColorMethod {
    Single(Vector4<f32>),
    StateLerp(Vector4<f32>, Vector4<f32>),
    DistToCenter(Vector4<f32>, Vector4<f32>),
    Neighbor(Vector4<f32>, Vector4<f32>),
}

impl ColorMethod {
    pub fn new(default_transparency: f32) -> Self {
        ColorMethod::StateLerp(
            DEFAULT_COLORS[0].as_vec4(default_transparency),
            DEFAULT_COLORS[1].as_vec4(default_transparency),
        )
    }

    pub fn to_int(&self) -> u32 {
        match self {
            ColorMethod::Single(_) => 0,
            ColorMethod::StateLerp(_, _) => 1,
            ColorMethod::DistToCenter(_, _) => 2,
            ColorMethod::Neighbor(_, _) => 3,
        }
    }

    pub fn parse_method(
        method: Option<&str>,
        default_transparency: f32,
    ) -> (ColorMethod, ColorType) {
        if method.is_none() {
            log::warn!("No color method provided, using default method");
            return (ColorMethod::new(default_transparency), ColorType::default());
        }
        let method = method.unwrap();
        let parsed_method = {
            let parts: Vec<&str> = method.split('/').collect();

            let method_type = parts[0].replace(' ', "");
            let method_type = if let Ok(method_type) = ColorMethodType::from_str(&method_type) {
                method_type
            } else {
                log::error!("Invalid color method, must be 'S', 'SL', 'DTC', or 'N'");
                log::warn!("Using default method");
                ColorMethodType::default()
            };

            if method_type == ColorMethodType::Single && parts.len() != 3 {
                log::error!("Color method '{}' requires only one color", method_type);
                log::warn!("Using default method");
                return (ColorMethod::new(default_transparency), ColorType::default());
            } else if method_type != ColorMethodType::Single && parts.len() != 4 {
                log::error!("Color method '{}' requires only two colors", method_type);
                log::warn!("Using default method");
                return (ColorMethod::new(default_transparency), ColorType::default());
            }

            let color_type = parts[1].replace(' ', "");
            let color_type = if let Ok(color_type) = ColorType::from_str(&color_type) {
                color_type
            } else {
                log::error!("Invalid color type, must be 'H', '1', '255', or 'D'");
                log::warn!("Using default method");
                ColorType::default()
            };

            let color1 = parts[2].replace(' ', "");
            let color1 = color1.as_str();
            let color2 = parts.get(3).copied();

            if method_type == ColorMethodType::Single {
                (
                    ColorMethod::Single(Self::parse_color(
                        color_type,
                        color1,
                        true,
                        default_transparency,
                    )),
                    color_type,
                )
            } else {
                color2.map_or_else(
                    || {
                        log::error!(
                            "Color method '{}' requires two colors, only one provided",
                            method_type
                        );
                        log::warn!("Using default method");
                        (ColorMethod::new(default_transparency), color_type)
                    },
                    |color2| {
                        let color2 = color2.replace(' ', "");
                        let color2 = color2.as_str();

                        match method_type {
                            ColorMethodType::StateLerp => (
                                ColorMethod::StateLerp(
                                    Self::parse_color(
                                        color_type,
                                        color1,
                                        true,
                                        default_transparency,
                                    ),
                                    Self::parse_color(
                                        color_type,
                                        color2,
                                        false,
                                        default_transparency,
                                    ),
                                ),
                                color_type,
                            ),
                            ColorMethodType::DistToCenter => (
                                ColorMethod::DistToCenter(
                                    Self::parse_color(
                                        color_type,
                                        color1,
                                        true,
                                        default_transparency,
                                    ),
                                    Self::parse_color(
                                        color_type,
                                        color2,
                                        false,
                                        default_transparency,
                                    ),
                                ),
                                color_type,
                            ),
                            ColorMethodType::Neighbor => (
                                ColorMethod::Neighbor(
                                    Self::parse_color(
                                        color_type,
                                        color1,
                                        true,
                                        default_transparency,
                                    ),
                                    Self::parse_color(
                                        color_type,
                                        color2,
                                        false,
                                        default_transparency,
                                    ),
                                ),
                                color_type,
                            ),
                            _ => unreachable!(),
                        }
                    },
                )
            }
        };
        log::debug!("Parsed color method: {:?}", parsed_method);
        parsed_method
    }

    fn parse_color(
        color_type: ColorType,
        color: &str,
        is_color_1: bool,
        default_transparency: f32,
    ) -> Vector4<f32> {
        match color_type {
            ColorType::Hex => Self::parse_hex_color(color, is_color_1, default_transparency),
            ColorType::ZeroTo1 => Self::parse_0_1_color(color, is_color_1, default_transparency),
            ColorType::ZeroTo255 => {
                Self::parse_0_255_color(color, is_color_1, default_transparency)
            }
            ColorType::PreDefined => {
                Self::parse_predefined_color(color, is_color_1, default_transparency)
            }
        }
    }

    fn parse_predefined_color(
        color: &str,
        is_color_1: bool,
        default_transparency: f32,
    ) -> Vector4<f32> {
        let predefined_color = Color::from_str(color);
        if let Ok(predefined_color) = predefined_color {
            predefined_color.as_vec4(default_transparency)
        } else {
            log::error!("{} No such color exists in Default Colors", color);
            log::warn!("Using default color");
            Self::get_default_color(is_color_1, default_transparency)
        }
    }

    fn parse_hex_color(color: &str, is_color_1: bool, default_transparency: f32) -> Vector4<f32> {
        if let Some(parsed_color) = Color::from_hex(color) {
            parsed_color.as_vec4(default_transparency)
        } else {
            let predefined_color = Color::from_str(color);
            if let Ok(predefined_color) = predefined_color {
                predefined_color.as_vec4(default_transparency)
            } else {
                log::error!("Invalid color format, Hex code '{}' is not valid", color);
                log::warn!("Color format must be '#RRGGBB' or a predefined color");
                log::warn!("Using default color");
                Self::get_default_color(is_color_1, default_transparency)
            }
        }
    }

    fn get_default_color(is_color_1: bool, default_transparency: f32) -> Vector4<f32> {
        if is_color_1 {
            DEFAULT_COLORS[0].as_vec4(default_transparency)
        } else {
            DEFAULT_COLORS[1].as_vec4(default_transparency)
        }
    }

    fn log_error_for_0_1_color() {
        log::error!("Invalid color format");
        log::warn!(
            "Color format must be 'R,G,B' in the range 0.0 - 1.0 <Float> or a predefined color"
        );
        log::warn!("Using default color");
    }

    fn log_error_for_0_255_color() {
        log::error!("Invalid color format");
        log::warn!("Color format must be 'R,G,B' in the range 0 - 255 <Int> or a predefined color");
        log::warn!("Using default color");
    }

    fn parse_0_1_color(color: &str, is_color_1: bool, default_transparency: f32) -> Vector4<f32> {
        let parts: Vec<&str> = color.split(',').collect();
        if parts.len() != 3 || parts.iter().any(|part| part.parse::<f32>().is_err()) {
            let predefined_color = Color::from_str(color);
            if let Ok(predefined_color) = predefined_color {
                return predefined_color.as_vec4(default_transparency);
            } else {
                Self::log_error_for_0_1_color();
                return Self::get_default_color(is_color_1, default_transparency);
            }
        }

        let rgb = parts
            .iter()
            .map(|part| part.parse::<f32>().unwrap())
            .collect::<Vec<f32>>();

        if let Some(color) = Color::from_value([rgb[0], rgb[1], rgb[2]]) {
            color.as_vec4(default_transparency)
        } else {
            Self::log_error_for_0_1_color();
            Self::get_default_color(is_color_1, default_transparency)
        }
    }

    fn parse_0_255_color(color: &str, is_color_1: bool, default_transparency: f32) -> Vector4<f32> {
        let parts: Vec<&str> = color.split(',').collect();
        if parts.len() != 3 {
            let predefined_color = Color::from_str(color);
            if let Ok(predefined_color) = predefined_color {
                return predefined_color.as_vec4(default_transparency);
            } else {
                Self::log_error_for_0_255_color();
                return Self::get_default_color(is_color_1, default_transparency);
            }
        }

        let rgb = parts
            .iter()
            .map(|part| part.parse::<f32>().unwrap() / 255.0)
            .collect::<Vec<f32>>();

        if let Some(color) = Color::from_value([rgb[0], rgb[1], rgb[2]]) {
            color.as_vec4(default_transparency)
        } else {
            Self::log_error_for_0_255_color();
            Self::get_default_color(is_color_1, default_transparency)
        }
    }

    fn format_color_option(
        color_option: Option<Color>,
        default_color: cgmath::Vector4<f32>,
    ) -> String {
        match color_option {
            Some(color) => color.to_string(),
            None => format!(
                "{:.1},{:.1},{:.1}",
                default_color.x, default_color.y, default_color.z
            ),
        }
    }

    fn format_method(
        prefix: &str,
        color1: Option<Color>,
        color2: Option<Color>,
        default1: cgmath::Vector4<f32>,
        default2: cgmath::Vector4<f32>,
        initial_color_method: &ColorType,
    ) -> String {
        let color1_str = Self::format_color_option(color1, default1);
        let color2_str = Self::format_color_option(color2, default2);
        format!(
            "{}/{}/{}/{}",
            prefix, initial_color_method, color1_str, color2_str
        )
    }

    pub fn to_formatted_string(&self, initial_color_type: &ColorType) -> String {
        match self {
            ColorMethod::Single(color) => {
                let parsed_color = Color::from_value([color.x, color.y, color.z]);
                let color_str = Self::format_color_option(parsed_color, *color);
                format!("S/{}/{}", initial_color_type, color_str)
            }
            ColorMethod::StateLerp(color1, color2) => {
                let parsed_color1 = Color::from_value([color1.x, color1.y, color1.z]);
                let parsed_color2 = Color::from_value([color2.x, color2.y, color2.z]);
                Self::format_method(
                    "SL",
                    parsed_color1,
                    parsed_color2,
                    *color1,
                    *color2,
                    initial_color_type,
                )
            }
            ColorMethod::DistToCenter(color1, color2) => {
                let parsed_color1 = Color::from_value([color1.x, color1.y, color1.z]);
                let parsed_color2 = Color::from_value([color2.x, color2.y, color2.z]);
                Self::format_method(
                    "DTC",
                    parsed_color1,
                    parsed_color2,
                    *color1,
                    *color2,
                    initial_color_type,
                )
            }
            ColorMethod::Neighbor(color1, color2) => {
                let parsed_color1 = Color::from_value([color1.x, color1.y, color1.z]);
                let parsed_color2 = Color::from_value([color2.x, color2.y, color2.z]);
                Self::format_method(
                    "N",
                    parsed_color1,
                    parsed_color2,
                    *color1,
                    *color2,
                    initial_color_type,
                )
            }
        }
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
    pub user_friendly_string: String,
}

impl Display for SimulationRules {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.user_friendly_string)
    }
}

impl Default for SimulationRules {
    fn default() -> Self {
        Self::new(
            vec![2, 6, 9],
            vec![4, 6, 8, 9, 10],
            10,
            NeighborMethod::Moore,
        )
    }
}

impl SimulationRules {
    pub fn new(
        survival: Vec<u8>,
        birth: Vec<u8>,
        num_states: u8,
        neighbor_method: NeighborMethod,
    ) -> Self {
        let user_friendly_string =
            Self::prepare_user_friendly_string(&survival, &birth, &num_states, &neighbor_method);

        SimulationRules {
            survival,
            birth,
            num_states,
            neighbor_method,
            user_friendly_string,
        }
    }

    pub fn prepare_user_friendly_string(
        survival: &[u8],
        birth: &[u8],
        num_states: &u8,
        neighbor_method: &NeighborMethod,
    ) -> String {
        // compress the continuous numbers
        let survival = Self::compress_continuous_numbers(survival);
        let birth = Self::compress_continuous_numbers(birth);

        format!(
            "\nSurvival: {}\nBirth: {}\nNum States: {}\nNeighbor Method: {}",
            survival, birth, num_states, neighbor_method.user_friendly_string()
        )
    }

    pub fn compress_continuous_numbers(numbers: &[u8]) -> String {
        let mut compressed_numbers = String::new();
        if numbers.is_empty() {
            return compressed_numbers;
        }
        let mut start = numbers[0];
        let mut end = numbers[0];
        for &number in &numbers[1..] {
            if number == end + 1 {
                end = number;
            } else {
                if start == end {
                    compressed_numbers.push_str(&format!("{},", start));
                } else {
                    compressed_numbers.push_str(&format!("{}-{},", start, end));
                }
                start = number;
                end = number;
            }
        }
        if start == end {
            compressed_numbers.push_str(&format!("{}", start));
        } else {
            compressed_numbers.push_str(&format!("{}-{}", start, end));
        }
        compressed_numbers.trim_end_matches(',').to_string()
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
        let neighbour_method = NeighborMethod::from_str(parts[3].as_str());
        if neighbour_method.is_err() {
            log::error!("Invalid neighbor method, must be 'M' for Moore or 'V' for Von Neumann");
            log::warn!("Using default rules");
            return SimulationRules::default();
        };
        let neighbor_method = neighbour_method.unwrap();

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
            let default = SimulationRules::default();
            return match rule_to_parse {
                SimulationRuleParseEnum::Survival => Ok(default.survival),
                SimulationRuleParseEnum::Birth => Ok(default.birth),
            };
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
