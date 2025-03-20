import sys
from qiskit_ibm_runtime import QiskitRuntimeService
 
token = sys.argv[1]

print("Adding Token:", token)

QiskitRuntimeService.save_account(
	token=token, # `token` is the API token for the IBM Quantum Experience account.
	channel="ibm_quantum" # `channel` distinguishes between different account types
)
exit()
