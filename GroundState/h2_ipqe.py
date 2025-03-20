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
from qiskit_algorithms import PhaseEstimation

from qiskit.primitives import StatevectorSampler, Sampler
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import UnitaryGate
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.visualization import plot_histogram

from qiskit_ibm_runtime import SamplerV2
from qiskit_ibm_runtime import QiskitRuntimeService

GRAPHS_DIR = 'graphs'

bond_lengths = [
    0.05,
    0.4,
    0.735,
    1.5,
    3.0,
    6.0
]

def makePlotFile(filename):
    output_file = os.path.join(GRAPHS_DIR, filename)  # Full file path

    # Create the directory if it doesn't exist
    if not os.path.exists(GRAPHS_DIR):
        os.makedirs(GRAPHS_DIR)

    return output_file

def plot_energy(points):
    """
    Plots any number of points on a 2D graph and connects them with lines.

    Parameters:
    points (list of tuples): A list of points, each as a tuple (x, y).
    """
    # Unpack the x and y coordinates
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]

    # Create the plot
    plt.scatter(x_coords, y_coords, color='blue', label='Points')  # Plot the points
    plt.plot(x_coords, y_coords, color='red', linestyle='--', label='Line')  # Connect the points with a line

    # Add labels and title
    plt.xlabel('R (a.u.)')
    plt.ylabel('E - 1/R (a.u.)')
    plt.title('Plot of Points')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()
    plt.savefig("graphs/energies.png")

def runType():
    # Prompt the user to choose between simulator and real-time run
    print("Choose an option:")
    print("1. Run on Simulator")
    print("2. Run on Real Hardware")
    choice = input("Enter your choice (1 or 2): ").strip()
    return choice

def getBackend():
    print("Getting Backend")
    service = QiskitRuntimeService(channel="ibm_quantum")
    backend = service.least_busy(operational=True, simulator=False)
    print(backend.name)
    target = backend.target

    return backend, generate_preset_pass_manager(target=target, optimization_level=3)

def simRun(bond_lengths):
    
    eVals = []
    for bond_length in bond_lengths:
        driver = PySCFDriver(atom=f'H .0 .0 .0; H .0 .0 {bond_length}', basis='sto3g')
        molecule = driver.run()

        hamiltonian = molecule.hamiltonian.second_q_op()
        mapper = ParityMapper(num_particles=molecule.num_particles)
        tapered_mapper = molecule.get_tapered_mapper(mapper)
        qubit_op = tapered_mapper.map(hamiltonian)

        qpe = PhaseEstimation(9, Sampler())

        state_in = HartreeFock(molecule.num_spatial_orbitals, molecule.num_particles, tapered_mapper)
        U = UnitaryGate(expm(1j*qubit_op.to_matrix()))

        result = qpe.estimate(U, state_in)

        print(
            f'Ground state energy from QPE, \
            Bond Length = {bond_length} : \
            {(result.phase - 1) * 2*np.pi} Ha'
        )

        exact_eigensolver = NumPyMinimumEigensolver()
        exact_eigensolver.filter_criterion = molecule.get_default_filter_criterion()
        gse = GroundStateEigensolver(mapper, exact_eigensolver)
        result = gse.solve(molecule)
        print(f'The exact ground state energy is: {result.eigenvalues[0]} Ha')
        eVals.append(result.eigenvalues[0])


    plot_energy(list(zip(bond_lengths, eVals)))



def QPE_circuit(bond_length):
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

def targetCircuit(num_result_bits, bond_length):
    print(f"Creating Initial Circuit: bl={bond_length}")

    qpe = PhaseEstimation(num_result_bits, StatevectorSampler())

    U, state_in = QPE_circuit(bond_length)

    qr1 = QuantumRegister(num_result_bits, 'control')
    qr2 = QuantumRegister(1, 'target')
    cr = ClassicalRegister(num_result_bits, 'res')

    circuit = QuantumCircuit(qr1, qr2, cr, name=f'bl-{bond_length}')
    circuit.compose(qpe.construct_circuit(U, state_in), inplace=True)
    circuit.measure(qr1, cr)

    #output_file = makePlotFile(f'qpe_circ_bl={bond_length}.png')
    #makePlotFile(f'qpe_circ_bl={bond_length}.png')
    circuit.decompose().draw(output='mpl', filename=output_file)

    return circuit

def runCircuits(backend, circuits):
    print("Running circuits")

    sampler = SamplerV2(mode=backend)

    job = sampler.run(circuits)
    id = job.job_id()

    result = job.result()

    service = QiskitRuntimeService()
    job = service.job(id)

    return job.result()

def saveResults(results):
    print("Saving Results")

    energies = []

    for i, result in enumerate(results):
        counts = result.data.res.get_counts()
        fig = plot_histogram(counts)

        result_file = makePlotFile(f'qpe_hist_bl={bond_lengths[i]}.png')
        fig.savefig(result_file)

        # get the most frequent result
        max_key, max_value = max(counts.items(), key=lambda x: x[1])
        # Reverse the string with this circuit because the 
        # first is the least significative bit, etc.
        phi = int(max_key[::-1], 2) / 8

        energy = 2*np.pi * (phi - 1)
        energies.append((bond_lengths[i], energy))

        print(f'phi = {phi} => best approximation with 3 bits = {energy:.2f} Ha')

    plot_energy(energies)

def main():

    run = runType()
    # Validate the user's input
    if run == "1":
        print("Running Circuits on Simulator")
        simRun(bond_lengths)
    elif run == "2":
        print("Running Circuits on Hardware")

        backend, pm = getBackend()

        circuits = []

        for bond_length in bond_lengths:
            circuit = targetCircuit(3, bond_length)

            isa_circuit = opCircuit(pm, circuit)

            circuits.append(isa_circuit)
        
        # Run the circuits on Target Hardware
        results = runCircuits(backend, circuits)

        saveResults(results)
    else:
        print("Invalid choice. Please enter 1 or 2.")
        return


if __name__ == "__main__":
    main()