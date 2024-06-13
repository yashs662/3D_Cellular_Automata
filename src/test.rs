#[cfg(test)]
mod rule_parse_tests {

    use crate::simulation::{NeighborMethod, SimulationRules};

    #[test]
    fn parse_rules() {
        let rules = SimulationRules::parse_rules(Some("10/4/5/M"));
        let correct = SimulationRules {
            survival: vec![10],
            birth: vec![4],
            num_states: 5,
            neighbor_method: NeighborMethod::Moore,
        };
        assert_eq!(rules, correct);
    }

    #[test]
    fn ranges() {
        let rules = SimulationRules::parse_rules(Some("4-5/3-4/5/V"));
        let correct = SimulationRules {
            survival: vec![4, 5],
            birth: vec![3, 4],
            num_states: 5,
            neighbor_method: NeighborMethod::VonNeumann,
        };
        assert_eq!(rules, correct);
    }

    #[test]
    fn weird_spacing() {
        let rules = SimulationRules::parse_rules(Some("4,5, 10 -15 /3,4/5/V"));
        let correct = SimulationRules {
            survival: vec![4, 5, 10, 11, 12, 13, 14, 15],
            birth: vec![3, 4],
            num_states: 5,
            neighbor_method: NeighborMethod::VonNeumann,
        };
        assert_eq!(rules, correct);
    }

    #[test]
    fn empty_rules() {
        let rules = SimulationRules::parse_rules(Some("//5/M"));
        let correct = SimulationRules {
            survival: Vec::new(),
            birth: Vec::new(),
            num_states: 5,
            neighbor_method: NeighborMethod::Moore,
        };
        assert_eq!(rules, correct);
    }

    #[test]
    fn invalid_format() {
        let rules = SimulationRules::parse_rules(Some("4,4/4/5"));
        let default = SimulationRules::default();
        assert_eq!(rules, default);
    }
}

#[cfg(test)]
mod color_method_parse_tests {
    use cgmath::Vector4;

    use crate::{constants::DEFAULT_COLORS, simulation::ColorMethod};

    #[test]
    fn parse_color() {
        let color_method = ColorMethod::parse_method(Some("S/H/#FF00FF"), 1.0);
        assert_eq!(
            color_method,
            ColorMethod::Single(Vector4::new(1.0, 0.0, 1.0, 1.0))
        );
    }

    #[test]
    fn state_lerp() {
        let color_method = ColorMethod::parse_method(Some("SL/H/#FFFF00/#00FFFF"), 1.0);
        assert_eq!(
            color_method,
            ColorMethod::StateLerp(
                Vector4::new(1.0, 1.0, 0.0, 1.0),
                Vector4::new(0.0, 1.0, 1.0, 1.0)
            )
        );
    }

    #[test]
    fn dist_to_center() {
        let color_method = ColorMethod::parse_method(Some("DTC/H/#FFFF00/#00FFFF"), 1.0);
        assert_eq!(
            color_method,
            ColorMethod::DistToCenter(
                Vector4::new(1.0, 1.0, 0.0, 1.0),
                Vector4::new(0.0, 1.0, 1.0, 1.0)
            )
        );
    }

    #[test]
    fn neighbor() {
        let color_method = ColorMethod::parse_method(Some("N/H/#FFFF00/#00FFFF"), 1.0);
        assert_eq!(
            color_method,
            ColorMethod::Neighbor(
                Vector4::new(1.0, 1.0, 0.0, 1.0),
                Vector4::new(0.0, 1.0, 1.0, 1.0)
            )
        );
    }

    #[test]
    fn invalid_format() {
        let color_method = ColorMethod::parse_method(Some("S/H/#FF0000/#00FF00"), 1.0);
        assert_eq!(
            color_method,
            ColorMethod::Single(Vector4::new(
                DEFAULT_COLORS[0][0],
                DEFAULT_COLORS[0][1],
                DEFAULT_COLORS[0][2],
                DEFAULT_COLORS[0][3]
            ))
        );
    }

    #[test]
    fn no_method() {
        let color_method = ColorMethod::parse_method(None, 1.0);
        assert_eq!(
            color_method,
            ColorMethod::Single(Vector4::new(
                DEFAULT_COLORS[0][0],
                DEFAULT_COLORS[0][1],
                DEFAULT_COLORS[0][2],
                DEFAULT_COLORS[0][3]
            ))
        );
    }

    #[test]
    fn invalid_color_type() {
        let color_method = ColorMethod::parse_method(Some("S/X/#FFFFFF"), 1.0);
        assert_eq!(
            color_method,
            ColorMethod::Single(Vector4::new(
                DEFAULT_COLORS[0][0],
                DEFAULT_COLORS[0][1],
                DEFAULT_COLORS[0][2],
                DEFAULT_COLORS[0][3]
            ))
        );
    }

    #[test]
    fn invalid_color_format() {
        let color_method = ColorMethod::parse_method(Some("S/H/#FF00"), 1.0);
        assert_eq!(
            color_method,
            ColorMethod::Single(Vector4::new(
                DEFAULT_COLORS[0][0],
                DEFAULT_COLORS[0][1],
                DEFAULT_COLORS[0][2],
                DEFAULT_COLORS[0][3]
            ))
        );
    }

    #[test]
    fn invalid_color_format_0_1() {
        let color_method = ColorMethod::parse_method(Some("S/1/2.0,5.0,0.0"), 1.0);
        assert_eq!(
            color_method,
            ColorMethod::Single(Vector4::new(
                DEFAULT_COLORS[0][0],
                DEFAULT_COLORS[0][1],
                DEFAULT_COLORS[0][2],
                DEFAULT_COLORS[0][3]
            ))
        );
    }

    #[test]
    fn invalid_color_format_0_255() {
        let color_method = ColorMethod::parse_method(Some("S/255/256,310,0"), 1.0);
        assert_eq!(
            color_method,
            ColorMethod::Single(Vector4::new(
                DEFAULT_COLORS[0][0],
                DEFAULT_COLORS[0][1],
                DEFAULT_COLORS[0][2],
                DEFAULT_COLORS[0][3]
            ))
        );
    }

    #[test]
    fn wrong_number_of_colors() {
        let color_method = ColorMethod::parse_method(Some("SL/H/#000000"), 1.0);
        assert_eq!(
            color_method,
            ColorMethod::Single(Vector4::new(
                DEFAULT_COLORS[0][0],
                DEFAULT_COLORS[0][1],
                DEFAULT_COLORS[0][2],
                DEFAULT_COLORS[0][3]
            ))
        );
    }

    #[test]
    fn weird_spacing() {
        let color_method = ColorMethod::parse_method(Some(" SL / H/  #00FFFF  / #0000FF"), 1.0);
        assert_eq!(
            color_method,
            ColorMethod::StateLerp(
                Vector4::new(0.0, 1.0, 1.0, 1.0),
                Vector4::new(0.0, 0.0, 1.0, 1.0)
            )
        );
    }

    #[test]
    fn wrong_separator() {
        let color_method = ColorMethod::parse_method(Some("SL/H/#00FFFF-#0000FF"), 1.0);
        assert_eq!(
            color_method,
            ColorMethod::Single(Vector4::new(
                DEFAULT_COLORS[0][0],
                DEFAULT_COLORS[0][1],
                DEFAULT_COLORS[0][2],
                DEFAULT_COLORS[0][3]
            ))
        );
    }

    #[test]
    fn wrong_separator2() {
        let color_method = ColorMethod::parse_method(Some("SL/H/#00FFFF,#0000FF"), 1.0);
        assert_eq!(
            color_method,
            ColorMethod::Single(Vector4::new(
                DEFAULT_COLORS[0][0],
                DEFAULT_COLORS[0][1],
                DEFAULT_COLORS[0][2],
                DEFAULT_COLORS[0][3]
            ))
        );
    }

    #[test]
    fn no_color() {
        let color_method = ColorMethod::parse_method(Some("SL/H/"), 1.0);
        assert_eq!(
            color_method,
            ColorMethod::Single(Vector4::new(
                DEFAULT_COLORS[0][0],
                DEFAULT_COLORS[0][1],
                DEFAULT_COLORS[0][2],
                DEFAULT_COLORS[0][3]
            ))
        );
    }
}
