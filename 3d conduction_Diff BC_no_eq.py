import numpy as np
import matplotlib.pyplot as plt

def solve_3d_conduction():
    print("\n3D Heat Conduction Solver (Mixed Boundary Conditions)")
    print("Supports Dirichlet, Convective, Heat Flux, and Adiabatic Boundaries\n")
    
    # Geometry
    try:
        Lx = float(input("Enter length in x (Lx): "))
        Ly = float(input("Enter length in y (Ly): "))
        Lz = float(input("Enter length in z (Lz): "))
        nx_div = int(input("Enter divisions in x (nx_div): "))
        ny_div = int(input("Enter divisions in y (ny_div): "))
        nz_div = int(input("Enter divisions in z (nz_div): "))
        k = float(input("Enter conductivity k (W/mK): "))
        h = float(input("Enter convection coefficient h (W/m^2K): "))
        T_inf = float(input("Enter ambient temperature (K): "))
        q_flux_left = float(input("Enter heat flux at x=0 (Left Face) (W/m^2): "))
        q_flux_top = float(input("Enter heat flux at z=Lz (Top Face) (W/m^2): "))
        g = float(input("Enter volumetric heat generation g (W/m^3): "))
        T_wall = float(input("Enter z=0 (Bottom Face) temperature (K): "))
    except ValueError:
        print("Invalid input.")
        return

    nx = nx_div + 1
    ny = ny_div + 1
    nz = nz_div + 1
    total = nx * ny * nz
    
    dx = Lx / nx_div
    dy = Ly / ny_div
    dz = Lz / nz_div
    
    bx = 1.0 / dx**2
    by = 1.0 / dy**2
    bz = 1.0 / dz**2
    
    center = -2 * (bx + by + bz)
    
    A = np.zeros((total, total))
    B = np.zeros(total)
    
    def idx(i, j, k_idx): 
        return k_idx * (nx * ny) + i * nx + j

    for k1 in range(nz):
        for i in range(ny):
            for j in range(nx):
                p = idx(i, j, k1)
                
                # z=0 is Constant Temperature (Dirichlet)
                if k1 == 0:
                    A[p, p] = 1
                    B[p] = T_wall
                    continue
                
                # We start with the center coefficient and the heat generation source term
                coeff_p = center
                rhs_val = -g / k
                
                # Z-Direction
                # Note: k1 == 0 is handled above.
                if k1 == nz - 1: # z=Lz (Heat Flux)
                    A[p, idx(i, j, k1-1)] += 2 * bz
                    rhs_val -= 2 * q_flux_top / (k * dz)
                else:
                    A[p, idx(i, j, k1-1)] += bz
                    A[p, idx(i, j, k1+1)] += bz
                
                # X-Direction
                if j == 0: # x=0 (Heat Flux)
                    A[p, idx(i, j+1, k1)] += 2 * bx
                    rhs_val -= 2 * q_flux_left / (k * dx)
                elif j == nx - 1: # x=Lx (Convective)
                    beta_x = h * dx / k
                    coeff_p -= 2 * bx * beta_x
                    A[p, idx(i, j-1, k1)] += 2 * bx
                    rhs_val -= 2 * bx * beta_x * T_inf
                else:
                    A[p, idx(i, j-1, k1)] += bx
                    A[p, idx(i, j+1, k1)] += bx
                
                # Y-Direction
                if i == 0: # y=0 (Convective)
                    beta_y = h * dy / k
                    coeff_p -= 2 * by * beta_y
                    A[p, idx(i+1, j, k1)] += 2 * by
                    rhs_val -= 2 * by * beta_y * T_inf
                elif i == ny - 1: # y=Ly (Adiabatic)
                    A[p, idx(i-1, j, k1)] += 2 * by
                else:
                    A[p, idx(i-1, j, k1)] += by
                    A[p, idx(i+1, j, k1)] += by
                
                A[p, p] += coeff_p
                B[p] = rhs_val

    T_flat = np.linalg.solve(A, B)
    T = T_flat.reshape((nz, ny, nx))

    # Calculate coordinates for visualization
    x_vals = np.linspace(0, Lx, nx)
    y_vals = np.linspace(0, Ly, ny)
    z_vals = np.linspace(0, Lz, nz)
    
    print("\nFinal Results (All Nodes by Z-Slice):")
    for k1 in range(nz):
        print(f"Z layer = {k1} (z = {z_vals[k1]:.3f} m)")
        for i in range(ny):
            for j in range(nx):
                print(f"Node ({j},{i},{k1}): {T[k1, i, j]:.2f} K")
        print()

    # Plotting 1: Interactive 3D Volume (Scatter)
    fig1 = plt.figure("Interactive 3D Grid", figsize=(10, 8))
    ax = fig1.add_subplot(111, projection='3d')
    
    X_list, Y_list, Z_list, T_list = [], [], [], []
    for k1 in range(nz):
        for i in range(ny):
            for j in range(nx):
                x, y, z = x_vals[j], y_vals[i], z_vals[k1]
                t_val = T[k1, i, j]
                
                X_list.append(x)
                Y_list.append(y)
                Z_list.append(z)
                T_list.append(t_val)
                
                # Add text label for each 3D node
                ax.text(x, y, z, f"({j},{i},{k1})\n{t_val:.0f} K", 
                        size=8, color='black', ha='center', va='bottom', zorder=10)

    scatter = ax.scatter(X_list, Y_list, Z_list, c=T_list, cmap='turbo', s=100, alpha=0.9, edgecolors='black', zorder=1)
    fig1.colorbar(scatter, ax=ax, label='Temperature (K)', pad=0.1)
    ax.set_title('3D Temperature Distribution (Mixed BCs)', weight='bold', pad=15, fontname='Times New Roman')
    ax.set_xlabel('Position x (m)', weight='bold', fontname='Times New Roman')
    ax.set_ylabel('Position y (m)', weight='bold', fontname='Times New Roman')
    ax.set_zlabel('Position z (m)', weight='bold', fontname='Times New Roman')
    ax.view_init(elev=20, azim=45)
    plt.show()

if __name__ == "__main__":
    solve_3d_conduction()
