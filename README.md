# 3d_celluar_automata
A multithreaded 3D cellular automata implementation with Rust and Bevy

## How it works
A few rules are defined for the cells which decides whether
<li> Will the cell be alive next simulation step
<li> Will the cell die in the next simulation step
<li> Will the cell allow other cells around itself to be born

The simulation is kept alive automatically with a spawn chance of 0.01% every frame by spawning a specific arrangement of cells in the center of the grid

## Controls
<li> press "i" to clear all cells
<li> press "1-6" to add a uniform cube of cells in the center of the grid
<li> press "p" to add random noise in the center of the grid
<li> press "Esc" to exit

## To do
Sometimes pressing "i" to clear cells the program crashes

## Screenshots
![cellular_automata](https://user-images.githubusercontent.com/66156000/206888650-435b93a0-1981-41dd-9e6c-6a5fd30096a9.png)
![cellular_automata_1](https://user-images.githubusercontent.com/66156000/206888768-745b520a-8492-43de-a657-219a4cb7a9d5.png)
