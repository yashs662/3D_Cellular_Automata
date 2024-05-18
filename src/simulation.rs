use std::fmt::{Display, Formatter};

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

impl Display for CellStateEnum {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            CellStateEnum::Alive => write!(f, "Alive"),
            CellStateEnum::Fading => write!(f, "Fading"),
            CellStateEnum::Dead => write!(f, "Dead"),
        }
    }
}

#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct CellState {
    pub state: CellStateEnum,
    pub fade_level: u8,
}

// Using the format from Softology's Blog (https://softologyblog.wordpress.com/2019/12/28/3d-cellular-automata-3/)
// Example: Rule 445 is the first rule in the video and shown as 4/4/5/M. This is fairly standard survival/birth CA syntax.
// The first 4 indicates that a state 1 cell survives if it has 4 neighbor cells.
// The second 4 indicates that a cell is born in an empty location if it has 4 neighbors.
// The 5 means each cell has 5 total states it can be in (state 4 for newly born which then fades to state 1 and then state 0 for no cell)
// M means a Moore neighborhood.

#[derive(Clone, Debug)]
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
        // check if rules is empty
        if rules.is_none() {
            log::warn!("No rules provided, using default rules");
            return SimulationRules::default();
        }
        let rules = rules.unwrap();
        let mut survival = Vec::new();
        let mut birth = Vec::new();
        // replace any whitespace in the string
        let parts: Vec<String> = rules.split('/').map(|part| part.replace(' ', "")).collect();
        if parts.len() != 4 {
            log::error!("Invalid rule format");
            log::warn!("Rule format must be 'survival/birth/num_states/neighbor_method'");
            log::warn!("Using default rules");
            return SimulationRules::default();
        }
        let survival_parts: Vec<&str> = parts[0].split(',').collect();
        // check if survival parts has only one element which is a empty string, i.e. empty rule
        if !(survival_parts.len() == 1 && survival_parts[0].is_empty()) {
            for part in survival_parts {
                if part.contains('-') {
                    let range: Vec<&str> = part.split('-').collect();
                    let start = range[0].parse::<u8>();
                    if start.is_err() {
                        log::error!("Invalid rule format for part in survival rules: {}", part);
                        log::warn!("Using default rules");
                        return SimulationRules::default();
                    }
                    let start = start.unwrap();
                    let end = range[1].parse::<u8>();
                    if end.is_err() {
                        log::error!("Invalid rule format for part in survival rules: {}", part);
                        log::warn!("Using default rules");
                        return SimulationRules::default();
                    }
                    let end = end.unwrap();
                    for i in start..=end {
                        survival.push(i);
                    }
                } else {
                    let rule = part.parse::<u8>();
                    if rule.is_err() {
                        log::error!("Invalid rule format for part in survival rules: {}", part);
                        log::warn!("Using default rules");
                        return SimulationRules::default();
                    }
                    survival.push(rule.unwrap());
                }
            }
        } else {
            log::debug!("Survival parts {:?}", survival_parts);
        }
        // check if survival has any repeating values
        if survival.len()
            != survival
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len()
        {
            log::error!("Survival rules contain repeating values");
            log::warn!("Using default rules");
            return SimulationRules::default();
        }
        let birth_parts: Vec<&str> = parts[1].split(',').collect();
        // check if birth parts has only one element which is a empty string, i.e. empty rule
        if !(birth_parts.len() == 1 && birth_parts[0].is_empty()) {
            for part in birth_parts {
                if part.contains('-') {
                    let range: Vec<&str> = part.split('-').collect();
                    let start = range[0].parse::<u8>();
                    if start.is_err() {
                        log::error!("Invalid rule format for part in birth rules: {}", part);
                        log::warn!("Using default rules");
                        return SimulationRules::default();
                    }
                    let start = start.unwrap();
                    let end = range[1].parse::<u8>();
                    if end.is_err() {
                        log::error!("Invalid rule format for part in birth rules: {}", part);
                        log::warn!("Using default rules");
                        return SimulationRules::default();
                    }
                    let end = end.unwrap();
                    for i in start..=end {
                        birth.push(i);
                    }
                } else {
                    let rule = part.parse::<u8>();
                    if rule.is_err() {
                        log::error!("Invalid rule format for part in birth rules: {}", part);
                        log::warn!("Using default rules");
                        return SimulationRules::default();
                    }
                    birth.push(rule.unwrap());
                }
            }
        }
        // check if birth has any repeating values
        if birth.len() != birth.iter().collect::<std::collections::HashSet<_>>().len() {
            log::error!("Birth rules contain repeating values");
            log::warn!("Using default rules");
            return SimulationRules::default();
        }
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_rules() {
        let rules = SimulationRules::parse_rules(Some("4/4/5/M"));
        assert_eq!(rules.survival, vec![4]);
        assert_eq!(rules.birth, vec![4]);
        assert_eq!(rules.num_states, 5);
        assert_eq!(rules.neighbor_method, NeighborMethod::Moore);
    }

    #[test]
    fn test_parse_rules_with_ranges() {
        let rules = SimulationRules::parse_rules(Some("4-5/3-4/5/V"));
        assert_eq!(rules.survival, vec![4, 5]);
        assert_eq!(rules.birth, vec![3, 4]);
        assert_eq!(rules.num_states, 5);
        assert_eq!(rules.neighbor_method, NeighborMethod::VonNeumann);
    }

    #[test]
    fn test_parse_rules_with_weird_spacing() {
        let rules = SimulationRules::parse_rules(Some("4,5, 10 -15 /3,4/5/V"));
        assert_eq!(rules.survival, vec![4, 5, 10, 11, 12, 13, 14, 15]);
        assert_eq!(rules.birth, vec![3, 4]);
        assert_eq!(rules.num_states, 5);
        assert_eq!(rules.neighbor_method, NeighborMethod::VonNeumann);
    }

    #[test]
    fn test_parse_rule_with_empty_rules() {
        let rules = SimulationRules::parse_rules(Some("//5/M"));
        assert_eq!(rules.survival, Vec::new());
        assert_eq!(rules.birth, Vec::new());
        assert_eq!(rules.num_states, 5);
        assert_eq!(rules.neighbor_method, NeighborMethod::Moore);
    }

    #[test]
    fn test_parse_rules_invalid_format() {
        // should return default rules
        let rules = SimulationRules::parse_rules(Some("4,4/4/5"));
        let default = SimulationRules::default();
        assert_eq!(rules.survival, default.survival);
        assert_eq!(rules.birth, default.birth);
        assert_eq!(rules.num_states, default.num_states);
        assert_eq!(rules.neighbor_method, default.neighbor_method);
    }
}
