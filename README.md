# 3d_celluar_automata
A multithreaded 3D cellular automata implementation with Rust and Bevy

## How it works
A few rules are defined for the cells which decides whether
<li> Will the cell be alive next simulation step
<li> Will the cell die in the next simulation step
<li> Will the cell allow other cells around itself to be born
The simulation is kept alive automatically with a spawn chance of 0.01% every frame by spawning a specific arrangement of cells in the center of the grid

## Controls
press "i" to clear all cells
press "1-6" to add a uniform cube of cells in the center of the grid
press "p" to add random noise in the center of the grid
press "Esc" to exit

## To do
Sometimes pressing "i" to clear cells the program crashes
