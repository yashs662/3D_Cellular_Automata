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
    use crate::{
        constants::DEFAULT_TRANSPARENCY,
        simulation::{ColorMethod, ColorMethodType, ColorType},
        utils::Color,
    };

    #[test]
    fn single() {
        let test_color_method_type = ColorMethodType::Single;
        let test_color_type = ColorType::PreDefined;
        let test_color = Color::Orange;

        let test_string = format!(
            "{}/{}/{}",
            test_color_method_type, test_color_type, test_color
        );

        assert_eq!(
            ColorMethod::parse_method(Some(&test_string), DEFAULT_TRANSPARENCY),
            (
                ColorMethod::Single(test_color.as_vec4(DEFAULT_TRANSPARENCY)),
                test_color_type
            )
        );
    }

    #[test]
    fn state_lerp() {
        let test_color_method_type = ColorMethodType::StateLerp;
        let test_color_type = ColorType::PreDefined;
        let test_color1 = Color::Orange;
        let test_color2 = Color::Cyan;

        let test_string = format!(
            "{}/{}/{}/{}",
            test_color_method_type, test_color_type, test_color1, test_color2
        );

        assert_eq!(
            ColorMethod::parse_method(Some(&test_string), DEFAULT_TRANSPARENCY),
            (
                ColorMethod::StateLerp(
                    test_color1.as_vec4(DEFAULT_TRANSPARENCY),
                    test_color2.as_vec4(DEFAULT_TRANSPARENCY)
                ),
                test_color_type
            )
        );
    }

    #[test]
    fn dist_to_center() {
        let test_color_method_type = ColorMethodType::DistToCenter;
        let test_color_type = ColorType::PreDefined;
        let test_color1 = Color::Orange;
        let test_color2 = Color::Cyan;

        let test_string = format!(
            "{}/{}/{}/{}",
            test_color_method_type, test_color_type, test_color1, test_color2
        );

        assert_eq!(
            ColorMethod::parse_method(Some(&test_string), DEFAULT_TRANSPARENCY),
            (
                ColorMethod::DistToCenter(
                    test_color1.as_vec4(DEFAULT_TRANSPARENCY),
                    test_color2.as_vec4(DEFAULT_TRANSPARENCY)
                ),
                test_color_type
            )
        );
    }

    #[test]
    fn neighbor() {
        let test_color_method_type = ColorMethodType::Neighbor;
        let test_color_type = ColorType::PreDefined;
        let test_color1 = Color::Orange;
        let test_color2 = Color::Cyan;

        let test_string = format!(
            "{}/{}/{}/{}",
            test_color_method_type, test_color_type, test_color1, test_color2
        );

        assert_eq!(
            ColorMethod::parse_method(Some(&test_string), DEFAULT_TRANSPARENCY),
            (
                ColorMethod::Neighbor(
                    test_color1.as_vec4(DEFAULT_TRANSPARENCY),
                    test_color2.as_vec4(DEFAULT_TRANSPARENCY)
                ),
                test_color_type
            )
        );
    }

    #[test]
    fn no_method() {
        assert_eq!(
            ColorMethod::parse_method(None, DEFAULT_TRANSPARENCY),
            (ColorMethod::new(DEFAULT_TRANSPARENCY), ColorType::default())
        );
    }

    #[test]
    fn invalid_color_type() {
        let test_color_method_type = ColorMethodType::Single;
        let test_color_type = "This is not a color type";
        let test_color1 = Color::Orange;
        let test_color2 = Color::Cyan;

        let test_string = format!(
            "{}/{}/{}/{}",
            test_color_method_type, test_color_type, test_color1, test_color2
        );

        assert_eq!(
            ColorMethod::parse_method(Some(&test_string), DEFAULT_TRANSPARENCY),
            (ColorMethod::new(DEFAULT_TRANSPARENCY), ColorType::default())
        );
    }

    #[test]
    fn invalid_color_format_hex() {
        let test_color_method_type = ColorMethodType::Single;
        let test_color_type = ColorType::Hex;
        let test_color = "#FF0";
        let test_string = format!(
            "{}/{}/{}",
            test_color_method_type, test_color_type, test_color
        );

        assert_eq!(
            ColorMethod::parse_method(Some(&test_string), DEFAULT_TRANSPARENCY),
            (ColorMethod::new(DEFAULT_TRANSPARENCY), test_color_type)
        );
    }

    #[test]
    fn invalid_color_format_0_1() {
        let test_color_method_type = ColorMethodType::Single;
        let test_color_type = ColorType::ZeroTo1;
        let test_color = "1.0,2.0,3.0";
        let test_string = format!(
            "{}/{}/{}",
            test_color_method_type, test_color_type, test_color
        );

        assert_eq!(
            ColorMethod::parse_method(Some(&test_string), DEFAULT_TRANSPARENCY),
            (ColorMethod::new(DEFAULT_TRANSPARENCY), test_color_type)
        );
    }

    #[test]
    fn invalid_color_format_0_255() {
        let test_color_method_type = ColorMethodType::Single;
        let test_color_type = ColorType::ZeroTo255;
        let test_color = "255,256,257";
        let test_string = format!(
            "{}/{}/{}",
            test_color_method_type, test_color_type, test_color
        );

        assert_eq!(
            ColorMethod::parse_method(Some(&test_string), DEFAULT_TRANSPARENCY),
            (ColorMethod::new(DEFAULT_TRANSPARENCY), test_color_type)
        );
    }

    #[test]
    fn wrong_number_of_colors() {
        let test_color_method_type = ColorMethodType::Single;
        let test_color_type = ColorType::PreDefined;
        let test_color1 = Color::Orange;
        let test_color2 = Color::Cyan;

        let test_string = format!(
            "{}/{}/{}/{}",
            test_color_method_type, test_color_type, test_color1, test_color2
        );

        assert_eq!(
            ColorMethod::parse_method(Some(&test_string), DEFAULT_TRANSPARENCY),
            (ColorMethod::new(DEFAULT_TRANSPARENCY), test_color_type)
        );
    }

    #[test]
    fn weird_spacing() {
        let test_color_method_type = ColorMethodType::Single;
        let test_color_type = ColorType::PreDefined;
        let test_color = Color::Orange;
        let test_string = format!(
            "   {} /{} /   {}    ",
            test_color_method_type, test_color_type, test_color
        );

        assert_eq!(
            ColorMethod::parse_method(Some(&test_string), DEFAULT_TRANSPARENCY),
            (
                ColorMethod::Single(test_color.as_vec4(DEFAULT_TRANSPARENCY)),
                test_color_type
            )
        );
    }

    #[test]
    fn wrong_separator() {
        let test_color_method_type = ColorMethodType::Single;
        let test_color_type = ColorType::PreDefined;
        let test_color = Color::Orange;
        let test_string = format!(
            "{},{},{}",
            test_color_method_type, test_color_type, test_color
        );

        assert_eq!(
            ColorMethod::parse_method(Some(&test_string), DEFAULT_TRANSPARENCY),
            (ColorMethod::new(DEFAULT_TRANSPARENCY), test_color_type)
        );
    }
    #[test]
    fn no_color() {
        let test_color_method_type = ColorMethodType::Single;
        let test_color_type = ColorType::Hex;
        let test_string = format!("{}/{}", test_color_method_type, test_color_type);

        assert_eq!(
            ColorMethod::parse_method(Some(&test_string), DEFAULT_TRANSPARENCY),
            (ColorMethod::new(DEFAULT_TRANSPARENCY), ColorType::default())
        );
    }
}
