#[derive(Clone)]
pub enum NeighborMethod {
    Moore,
    VonNeuman,
}

impl NeighborMethod {
    pub fn get_neighbor_iter(&self) -> &'static [(i32, i32, i32)] {
        match self {
            NeighborMethod::Moore => &MOOSE_NEIGHBORS[..],
            NeighborMethod::VonNeuman => &VONNEUMAN_NEIGHBORS[..],
        }
    }
}

pub static VONNEUMAN_NEIGHBORS: [(i32, i32, i32); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, -1),
    (0, 0, 1),
];

pub static MOOSE_NEIGHBORS: [(i32, i32, i32); 26] = [
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
    (1, 1, 1)
];
