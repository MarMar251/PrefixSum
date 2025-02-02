# Compiler and flags
MPICC = mpicc
MPIRUN = mpirun
MPI_HOST_FILE = mpi_host
NUM_PROCESSES = 3
PYTHON = python
SCRIPT = PrefixSum.py

# Default target: run the MPI program
all: run_mpi

# Target to run the MPI program with mpirun
run_mpi:
	$(MPIRUN) -n $(NUM_PROCESSES) -f $(MPI_HOST_FILE) $(PYTHON) $(SCRIPT)

# Target to clean up (useful if you generate any compiled or temporary files)
clean:
	@echo "No files to clean."

