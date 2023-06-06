import numpy as np
import numba as nb
import concurrent.futures as cf

# Initialize spins array
@nb.jit(nopython=True)
def initialize_spins(N):
    return np.random.choice(np.array([-1, 1]), size=(N, N))

# Magnetization of the system
@nb.jit(nopython=True)
def magnetization(s):
    return np.sum(s)

# Energy of the nearest neighbor interaction
@nb.jit(nopython=True)
def energy(s, J):
    N = s.shape[0]
    E = 0
    for i in range(N):
        for j in range(N):
            if i == 0:
                if j == N-1:
                    e = s[i, j] * s[i+1, j]
                else:
                    e = s[i, j] * (s[i+1, j] + s[i, j+1])
            if i == N-1:
                if j == N-1:
                    e = 0
                else:
                    e = s[i, j] * s[i, j+1]
            else:
                if j == N-1:
                    e = s[i, j] * s[i+1, j]
                else:
                    e = s[i, j] * (s[i+1, j] + s[i, j+1])

            E += -J * e
    return E

# Step of the simulation
@nb.jit(nopython=True)
def do_step(spins, J, temp, steps=100, print_steps=False) -> None:

    N = spins.shape[0]

    # Initialize the energy and magnetization
    E = energy(spins, J)
    M = magnetization(spins)

    # Loop over the number of steps
    accepted = 0
    for _ in range(steps):
        # Choose a random spin
        i = np.random.randint(N)
        j = np.random.randint(N)
        # Flip the spin
        spins[i, j] *= -1
        E_last = E
        E = energy(spins, J)
        dE = E - E_last

        if dE > 0:
            if np.random.random() > np.exp(-dE/temp):
                spins[i, j] *= -1
                E = E_last
            else:
                accepted += 1

    if print_steps:
        print("\nEnergy =", E, "Magnetization =", M)
        print("Acceptance rate:", 100.0*accepted/steps, "%")
    
    return None

# Running the MC simulation
@nb.jit(nopython=True)
def simulate(spins, J, temp, iters=50, steps=100, print_steps=False):
    E = energy(spins, J)
    M = magnetization(spins)
    E_list = np.empty(iters + 1)
    M_list = np.empty(iters + 1)
    E_list[0] = E
    M_list[0] = M
    
    for i in range(iters):
        do_step(spins, J, temp, steps=steps, print_steps=print_steps)
        E_list[i+1] = energy(spins, J)
        M_list[i+1] = magnetization(spins)
    
    return np.mean(E_list), np.mean(M_list), np.std(E_list), np.std(M_list)


def simulation(
    T0, 
    T1, 
    N_T, 
    N_S = 10, 
    J = 1, 
    iters = 50, 
    steps = 100, 
    print_steps = False,
    print_status = False
):
    T = np.linspace(T0, T1, N_T)

    spins = np.array([
        initialize_spins(N_S)
        for _ in range(N_T)
    ])
    E_avgs = np.empty(N_T)
    M_avgs = np.empty(N_T)
    E_stds = np.empty(N_T)
    M_stds = np.empty(N_T)
    
    def simulate_T(i):
        do_step(spins[i], J, T[0], steps=100000)
        E_avgs[i], M_avgs[i], E_stds[i], M_stds[i] = simulate(spins[i], J, T[i], iters=iters, steps=steps, print_steps=print_steps)
        return "T = " + str(round(T[i], 3)) + " Finished"
    
    with cf.ThreadPoolExecutor() as executor:
        results = [
            executor.submit(simulate_T, i) 
            for i in range(N_T)
        ]
        for r in cf.as_completed(results):
            if print_status:
                print(r.result())

    return T, E_avgs, M_avgs, E_stds, M_stds