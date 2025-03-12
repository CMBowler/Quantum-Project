import time
import pylab
import numpy as np
from scipy.linalg import expm
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Z2Symmetries
from qiskit.circuit.library import UnitaryGate
from qiskit.primitives import Sampler
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_algorithms import IterativePhaseEstimation
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(channel="ibm_quantum")
backend = service.least_busy(operational=True, simulator=False)
print(backend.name)
target = backend.target
pm = generate_preset_pass_manager(target=target, optimization_level=3)

def compute_energy(i, distance, algorithm):
    driver = PySCFDriver(
        atom=f'H .0 .0 .0; H .0 .0 {distance}',
        unit=DistanceUnit.ANGSTROM,
        charge=0,
        spin=0,
        basis='sto3g'
    )

    molecule = driver.run()
    mapper = ParityMapper(num_particles=molecule.num_particles)
    fer_op = molecule.hamiltonian.second_q_op()
    tapered_mapper = molecule.get_tapered_mapper(mapper)
    qubit_op = tapered_mapper.map(fer_op)

    if algorithm == 'NumPyMinimumEigensolver':
        algo = NumPyMinimumEigensolver()
        algo.filter_criterion = molecule.get_default_filter_criterion()
        solver = GroundStateEigensolver(mapper, algo)
        result = solver.solve(molecule)
        gs_energy = result.total_energies[0]
        
    elif algorithm == 'IQPE':
        num_iterations = 12  # Number of iterations for IQPE
        state_in = HartreeFock(molecule.num_spatial_orbitals, molecule.num_particles, tapered_mapper)
        sampler = Sampler(backend=backend)  # Use the quantum hardware backend
        iqpe = IterativePhaseEstimation(num_iterations, sampler)
        U = UnitaryGate(expm(1j * qubit_op.to_matrix()))
        result = 2 * np.pi * (iqpe.estimate(U, state_in).phase - 1)
        gs_energy = result + molecule.nuclear_repulsion_energy
    else:
        raise ValueError(f'Unrecognized algorithm: {algorithm}')
    return i, distance, gs_energy

# Parameters
algorithms = ['IQPE', 'NumPyMinimumEigensolver']
start = 0.5  # Start distance
by = 0.5  # Increment in distance
steps = 3  # Number of steps (reduce for hardware)
energies = np.empty([len(algorithms), steps + 1])
distances = np.empty(steps + 1)

# Run computations
start_time = time.time()
for j in range(len(algorithms)):
    algorithm = algorithms[j]
    for i in range(steps + 1):
        d = start + i * by / steps
        i, d, energy = compute_energy(i, d, algorithm)
        energies[j][i] = energy
        distances[i] = d
        print(f"Computed {algorithm} at distance {d:.2f}: Energy = {energy:.4f}")

print("--- %s seconds ---" % (time.time() - start_time))

# Plot results
for j in range(len(algorithms)):
    pylab.plot(distances, energies[j], label=algorithms[j])
pylab.xlabel('Interatomic distance (Å)')
pylab.ylabel('Energy (Hartree)')
pylab.title('H2 Ground State Energy on Quantum Hardware')
pylab.legend(loc='upper right')
pylab.show()