# Import all required modules
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pyscf
from scipy.linalg import expm

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.circuit.library import HartreeFock

from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import OptimizerResult
from qiskit_algorithms import VQE
from qiskit_algorithms import PhaseEstimation, IterativePhaseEstimation

from qiskit.primitives import StatevectorSampler, Sampler
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import UnitaryGate
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.visualization import plot_histogram

from qiskit_ibm_runtime import SamplerV2

# Declare global variables and define plotting functions

GRAPHS_DIR = 'graphs'

bond_lengths = [
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    6.0
]

def makePlotFile(filename):
    output_file = os.path.join(GRAPHS_DIR, filename)  # Full file path

    # Create the directory if it doesn't exist
    if not os.path.exists(GRAPHS_DIR):
        os.makedirs(GRAPHS_DIR)

    return output_file

def plot_energy(points):
    print("Plotting Energy")

    # Unpack the x and y coordinates
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]

    # Create the plot
    plt.scatter(x_coords, y_coords, color='blue', label='Points')  # Plot the points
    plt.plot(x_coords, y_coords, color='red', linestyle='--', label='Line')  # Connect the points with a line

    # Add labels and title
    plt.xlabel('R (a.u.)')
    plt.ylabel('E - 1/R (a.u.)')
    plt.title('Plot of Energy')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.savefig("graphs/energies.png")

    print("Plotting Alt. Energy")
    plt.clf()
    # Unpack the x and y coordinates
    x_coords = [point[0] for point in points]
    y_coords = [(point[1] + 1/point[0]) for point in points]

    # Create the plot
    plt.scatter(x_coords, y_coords, color='blue', label='Points')  # Plot the points
    plt.plot(x_coords, y_coords, color='red', linestyle='--', label='Line')  # Connect the points with a line

    # Add labels and title
    plt.xlabel('R (a.u.)')
    plt.ylabel('E (a.u.)')
    plt.title('Plot of Energy*')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.savefig("graphs/alt-energies.png")

    plt.close()


def saveResults(resolution, results):
    print("Saving Results")

    energies = []

    for i, result in enumerate(results):
        counts = result.get_counts()
        fig = plot_histogram(counts)

        result_file = makePlotFile(f'qpe_hist_bl={bond_lengths[i]}.png')
        fig.savefig(result_file)
        fig.clf()

        # get the most frequent result
        max_key, max_value = max(counts.items(), key=lambda x: x[1])
        # Reverse the string with this circuit because the 
        # first is the least significative bit, etc.
        phi = int(max_key[::-1], 2) / ((2**resolution)-1)

        energy = 2*np.pi * (-phi) - 1
        energies.append((bond_lengths[i], energy))

        print(f'phi = {phi} => best approximation with 3 bits = {energy:.2f} Ha')

    plot_energy(energies)


# Define Quantum Circuit construction functions

def getState(bond_length):
    driver = PySCFDriver(atom=f'H .0 .0 .0; H .0 .0 {bond_length}', basis='sto3g')
    molecule = driver.run()
    hamiltonian = molecule.hamiltonian.second_q_op()
    mapper = ParityMapper(num_particles=molecule.num_particles)
    tapered_mapper = molecule.get_tapered_mapper(mapper)
    qubit_op = tapered_mapper.map(hamiltonian)

    state_in = HartreeFock(molecule.num_spatial_orbitals, molecule.num_particles, tapered_mapper)
    U = UnitaryGate(expm(1j*qubit_op.to_matrix()))

    return U, state_in

def opCircuit(pm, circuit):
    print("Optimising Circuit: " + circuit.name)
    isa_circuit = pm.run(circuit)
    isa_circuit.count_ops()

    return isa_circuit

def targetCircuit(iterations, bond_length, backend):
    print(f"Creating Initial Circuit: bl={bond_length}")

    sampler = SamplerV2(mode=backend)

    iqpe = IterativePhaseEstimation(iterations, sampler)

    U, state_in = getState(bond_length)

    eReg = QuantumRegister(1, 'eigenstate')
    pReg = QuantumRegister(1, 'phase')
    cReg = ClassicalRegister(iterations, 'res')

    circuit = QuantumCircuit(pReg, eReg, cReg, name=f'bl-{bond_length}')

    for k in range(iterations):
        circuit.compose(
            iqpe.construct_circuit(U, state_in, k), 
            inplace=True
        )
        circuit.measure(eReg, cReg[k])

    output_file = makePlotFile(f'qpe_circ_bl={bond_length}.png')
    makePlotFile(f'qpe_circ_bl={bond_length}.png')
    circuit.decompose().draw(output='mpl', filename=output_file)

    return circuit


# Define functions to run quantum circuits on hardware.

def getBackend():
    print("Getting Backend")
    service = QiskitRuntimeService(channel="ibm_quantum")
    backend = service.least_busy(operational=True, simulator=False)
    print(backend.name)
    target = backend.target

    return backend, generate_preset_pass_manager(target=target, optimization_level=3)

def runCircuits(backend, circuits):
    print("Running circuits")

    sampler = SamplerV2(mode=backend)

    job = sampler.run(circuits)
    id = job.job_id()

    result = job.result()

    service = QiskitRuntimeService()
    job = service.job(id)

    return job.result()

# Run the Experiment

def main():

    run = runType()
    # Validate the user's input
    if run == "1":
        print("Cant run Simulator; Switch to AER")
        #simRun(bond_lengths)
    elif run == "2":
        print("Running Circuits on Hardware")


        
        # Run the circuits on Target Hardware
        results = runCircuits(backend, circuits)

        saveResults(results)
    else:
        print("Invalid choice. Please enter 1 or 2.")
        return


if __name__ == "__main__":
    main()