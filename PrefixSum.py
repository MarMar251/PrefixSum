import numpy as np
from mpi4py import MPI

# Function to compute parallel prefix sum
def parallel_prefix_sum(local_data, communicator):
    rank = communicator.Get_rank()  
    size = communicator.Get_size()  # Get the total number of processes

    # Step 1: Compute the local prefix sum of the segment (in parallel across processes)
    for idx in range(1, len(local_data)):
        local_data[idx] += local_data[idx - 1]

    # Show the local prefix sum computed by each process
    print(f"[Process #{rank}] Local prefix sum of segment: {local_data}")

    # Step 2: Collect the last element of each process' local prefix sum
    local_end = np.array([local_data[-1]], dtype=int)
    end_elements = None

    if rank == 0:
        # Only the main process (rank 0) will initialize the array to collect last elements
        end_elements = np.zeros(size, dtype=int)

    # Gather the last elements across all processes into the root process
    communicator.Gather(local_end, end_elements, root=0)

    # Step 3: Compute the offsets for each block in the root process
    if rank == 0:
        offsets = np.zeros(size, dtype=int)
        for i in range(1, size):
            offsets[i] = offsets[i - 1] + end_elements[i - 1]
    else:
        offsets = None

    # Scatter the calculated offsets back to all processes
    local_offset = np.zeros(1, dtype=int)
    communicator.Scatter(offsets, local_offset, root=0)

    # Step 4: Apply the computed offset to the local prefix sum in parallel across processes
    local_data += local_offset[0]

    return local_data

def main_execution():
    communicator = MPI.COMM_WORLD  
    rank = communicator.Get_rank() 
    num_processes = communicator.Get_size()

    total_elements = 20 
    input_array = None  
    # Initialize the full array in the main process (rank 0)
    if rank == 0:
        input_array = np.random.randint(0, 11, total_elements, dtype=int)
        print("[Main Process] Generated input array:", input_array)

        # Add padding if the total number of elements is not divisible by the number of processes
        padded_length = (total_elements + num_processes - 1) // num_processes * num_processes
        if total_elements != padded_length:
            input_array = np.pad(input_array, (0, padded_length - total_elements), constant_values=0)
        print("[Main Process] Padded input array (if necessary):", input_array)
    else:
        input_array = None

    local_segment = np.zeros((total_elements + num_processes - 1) // num_processes, dtype=int)

    # Scatter the array to all processes. Each process will receive a segment of the array.
    communicator.Scatter(input_array, local_segment, root=0)

    # Start timing the parallel prefix sum computation
    start_time = MPI.Wtime()

    # Perform the parallel prefix sum computation on the local segment
    local_result = parallel_prefix_sum(local_segment, communicator)

    # Gather the results from all processes into the main process
    if rank == 0:
        final_result = np.zeros_like(input_array, dtype=int)
    else:
        final_result = None

    communicator.Gather(local_result, final_result, root=0)

    # Stop the timer
    end_time = MPI.Wtime()

    if rank == 0:
        final_result = final_result[:total_elements]
        print(f"[Main Process] Final computed prefix sum result:", final_result)

        # Compute the sequential prefix sum for comparison
        sequential_result = np.zeros_like(final_result, dtype=int)
        sequential_result[0] = input_array[0]
        for i in range(1, total_elements):
            sequential_result[i] = sequential_result[i - 1] + input_array[i]
        print(f"[Main Process] Sequential prefix sum result (for verification):", sequential_result)

        print(f"[Main Process] Parallel execution time: {end_time - start_time:.6f} seconds")

        if np.array_equal(final_result, sequential_result):
            print("[Main Process] Verification successful: Results match!")
        else:
            print("[Main Process] Verification failed: Results do not match!")

if __name__ == "__main__":
    main_execution()
