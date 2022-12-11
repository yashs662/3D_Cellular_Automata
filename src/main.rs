#![windows_subsystem = "windows"]

use bevy::{
    prelude::*,
    render::view::NoFrustumCulling,
    render::{
        render_resource::WgpuFeatures,
        settings::WgpuSettings
    },
    diagnostic::{
        Diagnostics,
        FrameTimeDiagnosticsPlugin
    },
    window::PresentMode,
};
use cell_event::CellStatesChangedEvent;
pub mod cell_event;
mod cell_renderer;
mod cells_multithreaded;
mod neighbours;
mod rotating_camera;
mod rule;
mod utils;
use cell_renderer::*;
use cells_multithreaded::*;
use neighbours::NeighbourMethod;
use rotating_camera::{RotatingCamera, RotatingCameraPlugin};
use rule::*;

#[derive(Debug)]
pub struct CellState {
    value: u8,
    neighbours: u8,
    dist_to_center: f32,
}

impl CellState {
    pub fn new(value: u8, neighbours: u8, dist_to_center: f32) -> Self {
        CellState {
            value,
            neighbours,
            dist_to_center,
        }
    }
}

#[derive(Component)]
struct FpsText;

fn main() {
    let rule = Rule {
        bounding_size: 40,

        // builder
        survival_rule: Value::Singles(vec![2, 6, 9]),
        birth_rule: Value::Singles(vec![4, 6, 8, 9, 10]),
        states: 10,
        color_method: ColorMethod::StateLerp(Color::RED, Color::YELLOW),
        neighbour_method: NeighbourMethod::Moore,
    
        // glider
        // survival_rule: Value::Singles(vec![6,7]),
        // birth_rule: Value::Singles(vec![4,6,9,10,11]),
        // states: 6,
        // color_method: ColorMethod::StateLerp(Color::ORANGE_RED, Color::BLACK),
        // neighbour_method: NeighbourMethod::Moore,

        // lines
        // survival_rule: Value::Singles(vec![6]),
        // birth_rule: Value::Singles(vec![4, 6, 9, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23, 24]),
        // states: 35,
        // color_method: ColorMethod::StateLerp(Color::CYAN, Color::BLACK),
        // neighbour_method: NeighbourMethod::Moore,
    };
    let mut task_pool_settings = DefaultTaskPoolOptions::default();
    task_pool_settings.async_compute.percent = 1.0f32;
    task_pool_settings.compute.percent = 1.0f32;
    task_pool_settings.io.percent = 10.0f32;
    App::new()
        .insert_resource(Msaa { samples: 4 })
        .insert_resource(WindowDescriptor {
            title: "Cellular Automata".to_string(),
            present_mode: PresentMode::Fifo,
            ..Default::default()
        })
        .insert_resource(task_pool_settings)
        .add_plugin(FrameTimeDiagnosticsPlugin::default())
        .insert_resource(WgpuSettings {
            features: WgpuFeatures::POLYGON_MODE_LINE,
            ..Default::default()
        })
        .add_system(text_update_system)
        .add_plugins(DefaultPlugins)
        .insert_resource(ClearColor(Color::rgba(0.0, 0.0, 0.0, 0.0)))
        .add_event::<CellStatesChangedEvent>()
        .add_plugin(RotatingCameraPlugin)
        .add_plugin(CellMaterialPlugin)
        .insert_resource(rule)
        .add_plugin(CellsMultithreadedPlugin)
        .add_startup_system(setup)
        .run();
}

fn setup(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, asset_server: Res<AssetServer>) {

    // for fps counter
    commands.spawn_bundle(UiCameraBundle::default());

    commands.spawn().insert_bundle((
        meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
        Transform::from_xyz(0.0, 0.0, 0.0),
        GlobalTransform::default(),
        InstanceMaterialData(
            (1..=10)
                .flat_map(|x| (1..=100).map(move |y| (x as f32 / 10.0, y as f32 / 10.0)))
                .map(|(x, y)| InstanceData {
                    position: Vec3::new(x * 10.0 - 5.0, y * 10.0 - 5.0, 0.0),
                    scale: 1.0,
                    color: Color::hsla(x * 360., y, 0.5, 1.0).as_rgba_f32(),
                })
                .collect(),
        ),
        Visibility::default(),
        ComputedVisibility::default(),
        NoFrustumCulling,
    ));

    // camera
    commands
        .spawn_bundle(PerspectiveCameraBundle {
            transform: Transform::from_xyz(0.0, 0.0, 20.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..Default::default()
        })
        .insert(RotatingCamera::default());

    // Rich text with multiple sections
    commands
        .spawn_bundle(TextBundle {
            style: Style {
                align_self: AlignSelf::FlexEnd,
                ..Default::default()
            },
            // Use `Text` directly
            text: Text {
                // Construct a `Vec` of `TextSection`s
                sections: vec![
                    TextSection {
                        value: "FPS: ".to_string(),
                        style: TextStyle {
                            font: asset_server.load("fonts\\FiraSans-Bold.ttf"),
                            font_size: 60.0,
                            color: Color::WHITE,
                        },
                    },
                    TextSection {
                        value: "".to_string(),
                        style: TextStyle {
                            font: asset_server.load("fonts\\FiraMono-Medium.ttf"),
                            font_size: 60.0,
                            color: Color::GOLD,
                        },
                    },
                ],
                ..Default::default()
            },
            ..Default::default()
        })
        .insert(FpsText);
}

fn text_update_system(diagnostics: Res<Diagnostics>, mut query: Query<&mut Text, With<FpsText>>) {
    for mut text in query.iter_mut() {
        if let Some(fps) = diagnostics.get(FrameTimeDiagnosticsPlugin::FPS) {
            if let Some(average) = fps.average() {
                // Update the value of the second section
                text.sections[1].value = format!("{:.2}", average);
            }
        }
    }
}