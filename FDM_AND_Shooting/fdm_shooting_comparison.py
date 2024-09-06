import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def my_initial_values(x, y):
    # Initialize at 0
    return 0*x*y

def my_initial_velocity(x, y):
    # Initialize at 0
    return 0*x*y

def my_source(t,x,y,freq):
    # Define where the source "is". A quarter circle in this case
    mask = np.where((x**2+y**2<=0.5**2),True, False)
    # Return an oscillating value everywhere where mask is true
    return mask*np.sin(2*np.pi*freq*t)

def my_is_within_boundary(x,y):
    # Defines the boundaries of the problem
    Lx = 10
    Ly = 10
    lx = 1
    ly = 1

    # Is inside the outer walls?
    inside_outer_boundary = np.abs((x-Lx/2)/Lx + (y-Ly/2)/Ly) + \
        np.abs((x-Lx/2)/Lx - (y-Ly/2)/Ly) < 1

    # Is outside the inner column/walls
    outside_inner_boundary = np.abs((x-Lx/2)/lx + (y-Ly/2)/ly) + \
        np.abs((x-Lx/2)/lx - (y-Ly/2)/ly) > 1

    return inside_outer_boundary & outside_inner_boundary

def fdm_wave_equation(
    set_initial_values,
    set_initial_velocity_values,
    is_within_boundary,
    set_source, freq,
    Xv, Yv, Tv,
    wave_speed):
    """Returns the solution for the wave equation given user parameters.

    Parameters
    ----------
    set_initial_values: function
        A function where we can call to set the initial
        values of the solution at time t=0
    set_initial_velocity_values: function
        A function where we can call to set the initial
        "velocity" values of the solution at time t=0
    is_within_boundary: function
        Is a function that defines the domain values for
        which the solution is defined. In other words, its
        the region for which the solution is not fixed.
        It returns an array of True and False values.
    set_source: function
        Is a function that gives the values of the source
        in the relevant domain for a certain time. It returns
        an array of the size of the solution for a certain step.
    freq: Float
        Represents the frequency of the source function.
    Xv: 1-D Array of Floats
    Yv: 1-D Array of Floats
    Tv: 1-D Array of Floats
    wave_speed: Float
        The speed of the wave. For light it should be c.

    Returns
    -------
    U: Array of size (Tv.size, Xv.size, Yv.size)
        Returns an array with the solution approximated
        by the finite difference method for the inhomogeneous
        wave equation.

    """

    dx = Xv[1]-Xv[0]
    dy = Yv[1]-Yv[0]
    dt = Tv[1]-Tv[0]
    v = wave_speed

    Cx = v*dt/dx
    Cy = v*dt/dy
    print("Courant numbers:")
    print(f'Cx: {Cx}')
    print(f'Cy: {Cy}')
    print()

    # Time index
    It = range(0, Tv.size)

    # Initialize the final solution
    U = np.zeros((Tv.size, Xv.size, Yv.size))

    # Set the initial conditions
    U[0] = set_initial_values(Xv[np.newaxis,:], Yv[:,np.newaxis])
    V_initial = set_initial_velocity_values(Xv[np.newaxis,:], Yv[:,np.newaxis])

    # Define the static boundaries of the simulation
    Within_Boundary = is_within_boundary(Xv[np.newaxis,:], Yv[:,np.newaxis])

    # Create the source
    S = np.zeros((Tv.size, Xv.size, Yv.size))

    # Get the values for the first step separately
    # since it uses a slightly different equation
    U[1,1:-1,1:-1] = \
    (Cx**2)*(U[0,0:-2,1:-1]-2*U[0,1:-1,1:-1]+U[0,2:,1:-1])+\
        (Cy**2)*(U[0,1:-1,0:-2]-2*U[0,1:-1,1:-1]+U[0,1:-1,2:])+\
        U[0,1:-1,1:-1] + dt*V_initial[1:-1,1:-1] + (dt**2)*S[0,1:-1,1:-1]

    # Replace values at the boundaries as 0
    U[1,~Within_Boundary] = 0

    # Advance the solution for each time step
    for k in It[:-1]:
        # Update the source array
        S[k] = set_source(Tv[k], Xv[np.newaxis,:], Yv[:,np.newaxis], freq)

        # Get the next step through finite difference method (FDM)
        U[k+1,1:-1,1:-1] = \
        (Cx**2)*(U[k,0:-2,1:-1]-2*U[k,1:-1,1:-1]+U[k,2:,1:-1])+\
            (Cy**2)*(U[k,1:-1,0:-2]-2*U[k,1:-1,1:-1]+U[k,1:-1,2:])+\
            2*U[k,1:-1,1:-1] - U[k-1,1:-1,1:-1] + (dt**2)*S[k,1:-1,1:-1]

        # Override values at boundaries
        U[k+1,~Within_Boundary] = 0

    return U

# Simulation paramters
Nx = 2**7
Ny = 2**7
Nt = 2**13

v = sp.constants.speed_of_light
factor = 1e-2 # Time resolution
# We use a fraction of the desired frequency
# due to the fact that the frequency is
# an order of magnitude greater than velocity
# and the waves bunch together in the simulation
# We could use the real 2.4e9 frequency but it would
# require we increase the spatial resolution to outside
# our available resources. With more memory and allowed time
# we could use 2.4e9 directly.
freq = 0.1*2.4e9


dt = factor/v

Xv = np.linspace(0, 10, num=Nx)
Yv = np.linspace(0, 10, num=Ny)
Tv = np.linspace(0, dt*Nt, num=Nt)

print("Calculating...\n")
my_U = fdm_wave_equation(my_initial_values,
                         my_initial_velocity,
                         my_is_within_boundary,
                         my_source, freq, Xv, Yv, Tv, v)

Intensity = np.absolute(my_U)

# Plotting
print("Plotting...\n")

# Set field min and max color values
#U_vmin = np.min(my_U)
U_vmax = 4*np.std(my_U)
U_vmin = -U_vmax

# Set intensity min and max color values
I_vmin = np.min(Intensity)
I_vmax = 5*np.std(Intensity)

fig, ax = plt.subplots()

U1 = my_U
vmin1 = U_vmin
vmax1 = U_vmax

U2 = Intensity
vmin2 = I_vmin
vmax2 = I_vmax

# Animation duration and resolution
ani_duration = 30 #[s]
my_fps = 60
my_interval = int(1000/my_fps)
plot_prop = int(Nt/(ani_duration*my_fps))

# Other parameters
ax.axis('equal')
time_template = 'Raw field (t=%.2f [ns])'

im = ax.pcolormesh(Xv, Yv, U1[0], vmin=vmin1, vmax=vmax1, shading='gouraud')  # show an initial one first
ax.set_title('')
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
fig.colorbar(im, ax=ax)

def animate(i):
    im.set_array(U1[plot_prop*i].ravel())
    ax.set_title(time_template % (plot_prop*i*dt*1e9))
    return im, ax

ani = animation.FuncAnimation(fig,
                              animate,
                              int(Nt/plot_prop),
                              interval=my_interval,
                              blit=False,
                              repeat_delay=0)


plt.show()

# Print time average of intensity
fig2, ax2 = plt.subplots()
im2 = ax2.pcolormesh(Xv, Yv, np.mean(U2, axis=0),vmin=vmin2, vmax=vmax2)
ax2.axis('equal')
ax2.set_title(f'Time-averaged Field Intensity after {Nt*dt*1e9:.2f} ns')
ax2.set_xlabel('$x$ [m]')
ax2.set_ylabel('$y$ [m]')
fig2.colorbar(im2, ax=ax2)
