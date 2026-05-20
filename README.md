# 3D Heat Conduction Solver (Mixed Boundary Conditions)

## Aim
To solve 3D steady-state heat conduction in a cuboid with uniform heat generation and a combination of Dirichlet, Convective, Constant Heat Flux, and Adiabatic boundary conditions on its faces.

## Theory
The governing equation is the 3D Poisson equation:

$$\nabla^2 T + \frac{\dot{g}}{k} = 0 \implies \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} + \frac{\partial^2 T}{\partial z^2} + \frac{\dot{g}}{k} = 0$$

This solver supports highly complex boundary setups across the six faces:
- **Bottom Face ($z=0$)**: Dirichlet boundary (Constant Temperature, $T = T_{wall}$).
- **Top Face ($z=L_z$)**: Constant Heat Flux ($q_{top}$).
- **Left Face ($x=0$)**: Constant Heat Flux ($q_{left}$).
- **Right Face ($x=L_x$)**: Convective boundary ($h, T_\infty$).
- **Front Face ($y=0$)**: Convective boundary ($h, T_\infty$).
- **Back Face ($y=L_y$)**: Adiabatic boundary (Insulated, $\frac{\partial T}{\partial y} = 0$).

Ghost nodes are introduced at all convective, heat flux, and adiabatic boundaries to maintain second-order accuracy.

## File Structure
- `3d conduction_Diff BC_no_eq.py` - Primary script containing the sparse matrix generation, boundary formulations, solver execution, and 3D scatter plotting.
- `v0.py` - Initial prototype code.
- `output.txt` - Complete output data detailing 3D nodal temperatures by layer.
- `3D Grid.png` - Visual plot containing the 3D grid scatter map showing thermal distribution.

## How to Run
Ensure you have the required dependencies:
```bash
pip install numpy matplotlib
python "3d conduction_Diff BC_no_eq.py"
```
