import cirq
import sympy
import logging
from config.settings import NUM_QUBITS
from data.preprocessor import qubits

logger = logging.getLogger(__name__)


def create_pqc():
    """Create Parameterized Quantum Circuit."""
    symbols = sympy.symbols(f'theta0:{NUM_QUBITS}')
    circuit = cirq.Circuit()
    for i, q in enumerate(qubits):
        circuit.append(cirq.ry(symbols[i])(q))
    for i in range(NUM_QUBITS - 1):
        circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))
    readout_operators = [cirq.Z(q) for q in qubits]
    return circuit, symbols, readout_operators

def create_reuploading_pqc(num_layers=3):
    data_symbols = sympy.symbols(f'x0:{NUM_QUBITS}')
    theta_symbols = sympy.symbols(f'theta0:{num_layers * NUM_QUBITS}')
    circuit = cirq.Circuit()
    theta_idx = 0
    for layer in range(num_layers):
        for i, q in enumerate(qubits):
            circuit.append(cirq.rx(data_symbols[i])(q))
        for i, q in enumerate(qubits):
            circuit.append(cirq.ry(theta_symbols[theta_idx])(q))
            theta_idx += 1
        for i in range(NUM_QUBITS - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
    readout_operators = [cirq.Z(q) for q in qubits]
    return circuit, data_symbols, theta_symbols, readout_operators

def create_reuploading_pqc_with_multiqubit_correlators(num_layers=3):
    data_symbols = sympy.symbols(f'x0:{NUM_QUBITS}')
    theta_symbols = sympy.symbols(f'theta0:{num_layers * NUM_QUBITS}')
    circuit = cirq.Circuit()
    theta_idx = 0
    for layer in range(num_layers):
        for i, q in enumerate(qubits):
            circuit.append(cirq.rx(data_symbols[i])(q))
        for i, q in enumerate(qubits):
            circuit.append(cirq.ry(theta_symbols[theta_idx])(q))
            theta_idx += 1
        for i in range(NUM_QUBITS - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
    readout_operators = [
        cirq.Z(qubits[0]),
        cirq.Z(qubits[0]) * cirq.Z(qubits[1]),
        cirq.Z(qubits[1]) * cirq.Z(qubits[2]),
        cirq.Z(qubits[2]) * cirq.Z(qubits[3]),
        cirq.Z(qubits[3]) * cirq.Z(qubits[4]),
    ]
    return circuit, data_symbols, theta_symbols, readout_operators

def create_reuploading_pqc_with_alltoallentanglement_multiqubit_correlators(num_layers=3):
    data_symbols = sympy.symbols(f'x0:{NUM_QUBITS}')
    theta_symbols = sympy.symbols(f'theta0:{num_layers * NUM_QUBITS}')
    circuit = cirq.Circuit()
    theta_idx = 0
    for layer in range(num_layers):
        for i, q in enumerate(qubits):
            circuit.append(cirq.rx(data_symbols[i])(q))
        for i, q in enumerate(qubits):
            circuit.append(cirq.ry(theta_symbols[theta_idx])(q))
            theta_idx += 1
        for i in range(NUM_QUBITS):
            for j in range(i + 1, NUM_QUBITS):
                circuit.append(cirq.CNOT(qubits[i], qubits[j]))
    readout_operators = [
        cirq.Z(qubits[0]),
        cirq.Z(qubits[0]) * cirq.Z(qubits[1]),
        cirq.Z(qubits[1]) * cirq.Z(qubits[2]),
        cirq.Z(qubits[2]) * cirq.Z(qubits[3]),
        cirq.Z(qubits[3]) * cirq.Z(qubits[4]),
    ]

    return circuit, data_symbols, theta_symbols, readout_operators
