# %% [markdown]
# ## _*H2 ground state energy computation using Quantum Phase Estimation*_
# 
# This notebook demonstrates using Qiskit Chemistry to compute ground state energy of the Hydrogen (H2) molecule using QPE (Quantum Phase Estimation) algorithm. Let's look at how to carry out such computation programmatically.
# 
# This notebook has been written to use the PYSCF chemistry driver.

# %% [markdown]
# ### Step 1: Map problem to quantum circuits and operators

# %%
import numpy as np
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

# %% [markdown]
# We first set up the H2 molecule, create the fermionic and in turn the qubit operator using PySCF.

# %%
driver = PySCFDriver(atom='H .0 .0 .0; H .0 .0 0.735', basis='sto3g')
molecule = driver.run()
hamiltonian = molecule.hamiltonian.second_q_op()
mapper = ParityMapper(num_particles=molecule.num_particles)
tapered_mapper = molecule.get_tapered_mapper(mapper)
qubit_op = tapered_mapper.map(hamiltonian)

# %% [markdown]
# Using a classical exact eigenvalue solver, we can establish the reference groundtruth value of the ground state energy:

# %%
exact_eigensolver = NumPyMinimumEigensolver()
exact_eigensolver.filter_criterion = molecule.get_default_filter_criterion()
gse = GroundStateEigensolver(mapper, exact_eigensolver)
result = gse.solve(molecule)
print(f'The exact ground state energy is: {result.eigenvalues[0]} Ha')

# %% [markdown]
# Next we set up the QPE algorithm instance using the HartreeFock initial state:

# %%
qpe = PhaseEstimation(9, Sampler())

state_in = HartreeFock(molecule.num_spatial_orbitals, molecule.num_particles, tapered_mapper)
U = UnitaryGate(expm(1j*qubit_op.to_matrix()))

result = qpe.estimate(U, state_in)

print(f'Ground state energy from QPE: {(result.phase - 1) * 2*np.pi} Ha')

# %% [markdown]
# As can be easily seen, the QPE computed energy is quite close to the groundtruth value we computed earlier.

# %% [markdown]
# ### Step 2: Optimize for target hardware

# %%
num_result_bits = 3
qpe = PhaseEstimation(num_result_bits, StatevectorSampler())

qr1 = QuantumRegister(num_result_bits, 'control')
qr2 = QuantumRegister(1, 'target')
cr = ClassicalRegister(num_result_bits, 'res')
circuit = QuantumCircuit(qr1, qr2, cr)
circuit.compose(qpe.construct_circuit(U, state_in), inplace=True)
circuit.measure(qr1, cr)
circuit.decompose().draw(output='mpl')

# %%
service = QiskitRuntimeService(channel="ibm_quantum")
backend = service.least_busy(operational=True, simulator=False)
print(backend.name)
target = backend.target
pm = generate_preset_pass_manager(target=target, optimization_level=3)

isa_circuit = pm.run(circuit)
isa_circuit.count_ops()

# %% [markdown]
# ### Step 3: Execute on target hardware

# %%
sampler = SamplerV2(mode=backend)

job = sampler.run([isa_circuit])

result = job.result()
id = job.job_id()

print(id)

# %%
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()
job = service.job(id)
job_result = job.result()

# %% [markdown]
# ### Step 4: Post-process results

# %%
""" best possible result using 3 bits:
phi = 0.c1c2c3 can be: 0 _ 0.125 _ 0.25 _ 0.375 _ ... _ 0.875
=> 2pi*(phi - 1) can be: ~ -6.28 _ -5.50 _ -4.71 _ ... _ -2.36 _ -1.57 _ -0.79 _ ... _ 0
So the closest result is -1.57, which corresponds to phi = 0.75 (= 0.110 in binary)
"""

# %%
counts = job_result[0].data.res.get_counts()
plot_histogram(counts)

# %%
# get the most frequent result
max_key, max_value = max(counts.items(), key=lambda x: x[1])

phi = int(max_key[::-1], 2) / 8 # I have to reverse the string with this circuit because the first is the least significative bit, etc.
                                # in fact, if you look at the plot of the circuit used here: https://learning.quantum.ibm.com/course/fundamentals-of-quantum-algorithms/phase-estimation-and-factoring#general-procedure-and-analysis
                                # you can see that the inverse QFT has the number in reverse order as the circuit here.
print(f'phi = {phi} => best approximation with 3 bits = {2*np.pi * (phi - 1):.2f} Ha')

# %%
! pip freeze | grep qiskit


