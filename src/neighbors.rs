#[derive(Clone)]
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
    (1, 1, 1)
];
