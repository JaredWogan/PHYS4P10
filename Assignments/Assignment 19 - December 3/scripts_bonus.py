import numpy as np
import numba as nb
import matplotlib.pyplot as plt
plt.style.use("seaborn")

# Initialize spins array
@nb.jit(nopython=True)
def initialize_spins(N):
    return 2*np.pi*np.random.rand(N, N)

# Magnetization of the system
@nb.jit(nopython=True)
def magnetization(s):
    return np.sum(np.cos(s)), np.sum(np.sin(s))

# Energy of the nearest neighbor interaction
@nb.jit(nopython=True)
def energy(s, J):
    N = s.shape[0]
    E = 0
    for i in range(N):
        for j in range(N):
            if i == 0:
                if j == N-1:
                    e = np.cos(s[i, j] - s[i+1, j])
                else:
                    e = np.cos(s[i, j] - s[i+1, j]) + np.cos(s[i, j] - s[i, j+1])
            if i == N-1:
                if j == N-1:
                    e = 0
                else:
                    e = np.cos(s[i, j] - s[i, j+1])
            else:
                if j == N-1:
                    e = np.cos(s[i, j] - s[i+1, j])
                else:
                    e = np.cos(s[i, j] - s[i+1, j]) + np.cos(s[i, j] - s[i, j+1])

            E += -J * e
    return E

# Step of the simulation
@nb.jit(nopython=True)
def do_step(spins, J, temp, steps=100, print_steps=False):

    N = spins.shape[0]

    # Initialize the energy and magnetization
    E = energy(spins, J)
    Mx, My = magnetization(spins)

    # Loop over the number of steps
    accepted = 0
    for _ in range(steps):
        # Choose a random spin
        i = np.random.randint(N)
        j = np.random.randint(N)
        # Flip the spin
        dtheta = 2*np.pi*np.random.rand()
        spins[i, j] += dtheta
        E_last = E
        E = energy(spins, J)
        dE = E - E_last

        if dE > 0:
            if np.random.random() > np.exp(-dE/temp):
                spins[i, j] -= dtheta
                E = E_last
            else:
                accepted += 1
        else:
            accepted += 1

    if print_steps:
        print("\nEnergy =", E, "Magnetization =[", Mx, ",", My, "]")
        print("Acceptance rate:", 100.0*accepted/steps, "%")
    
    return spins

# Running the MC simulation
@nb.jit(nopython=True)
def simulate(spins, J, temp, iters=50, steps=100, print_steps=False):
    E = energy(spins, J)
    M = magnetization(spins)
    E_list = np.empty(iters + 1)
    Mx_list = np.empty(iters + 1)
    My_list = np.empty(iters + 1)
    E_list[0] = E
    Mx_list[0], My_list[0] = M
    
    for i in range(iters):
        spins = do_step(spins, J, temp, steps=steps, print_steps=print_steps)
        E_list[i+1] = energy(spins, J)
        Mx_list[i+1], My_list[i+1] = magnetization(spins)
    
    return np.mean(E_list), np.mean(Mx_list), np.mean(My_list), np.std(E_list), np.std(Mx_list), np.std(My_list), spins


def simulation(N=10, J=1, T=0.001, iters=50, steps=10_000, print_steps=False):
    s = initialize_spins(N)

    *junk, fs = simulate(np.copy(s), J, T, iters=iters, steps=steps, print_steps=print_steps)

    fig, ax = plt.subplots(figsize=(20, 10), nrows=1, ncols=2)
    fig.suptitle(f"N={N}, J={J}, T={T}, iters={iters}, steps={steps}", fontsize=32)
    ax[0].set_title("Initial state", fontsize=20)
    ax[1].set_title("Final state", fontsize=20)
    ax[0].set_xticks(np.arange(N))
    ax[0].set_yticks(np.arange(N))
    ax[1].set_xticks(np.arange(N))
    ax[1].set_yticks(np.arange(N))
    for i in range(N):
        for j in range(N):
            ax[0].arrow(
                i, j, 
                np.cos(s[i, j])/4, np.sin(s[i, j])/4, 
                head_width=0.15, 
                head_length=0.15, 
                fc='k', 
                ec='k'
            )
            ax[1].arrow(
                i, j, 
                np.cos(fs[i, j])/4, np.sin(fs[i, j])/4, 
                head_width=0.15, 
                head_length=0.15, 
                fc='k', 
                ec='k'
            )
    plt.show()
    plt.close()