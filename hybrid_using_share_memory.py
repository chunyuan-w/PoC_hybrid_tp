import torch
import torch.multiprocessing as mp

M = 1024
K = 7168
dtype = torch.bfloat16

def gpu_worker(shared_input, shared_output, ready_event, done_event):
    # === Step 1. GPU writes tensor into shared pinned input ===
    gpu_tensor = torch.arange(shared_input.numel(), device="cuda", dtype=shared_input.dtype).reshape(shared_input.size())
    
    # TODO: collect the time of this operation: GPU to CPU
    shared_input.copy_(gpu_tensor.cpu(), non_blocking=True)
    
    print("[GPU] Wrote tensor into shared pinned input")

    # Signal CPU that input is ready
    ready_event.set()

    # Wait for CPU to finish reduction
    done_event.wait()

    # === Step 3. GPU reads reduced result from shared pinned output ===

    # TODO: collect the time of this operation: CPU to GPU
    result_gpu = shared_output.to("cuda", non_blocking=True)

    print("[GPU] Read reduced result back to GPU:", result_gpu)


def cpu_worker(shared_input, shared_output, ready_event, done_event):
    # TODO: bind core and memory here?
    
    # Wait for GPU to signal input is ready
    ready_event.wait()

    # === Step 2. CPU reduces the tensor ===
    print("[CPU] Reading shared input...")
    s = shared_input.relu_()
    shared_output.copy_(s, non_blocking=True)  # copy result into shared memory
    print("[CPU] Wrote reduced result into shared output:", s)

    # Signal GPU that reduction is done
    done_event.set()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # safer for CUDA + multiprocessing

    # Shared pinned input tensor (large data buffer)
    shared_input = torch.empty(M, K, dtype=dtype, pin_memory=True)
    # Make shared_input accessible by multiple processes
    shared_input.share_memory_()

    # Shared pinned output tensor (just 1 value for reduction result)
    shared_output = torch.empty(M, K, dtype=dtype, pin_memory=True)
    # Make shared_output accessible by multiple processes
    shared_output.share_memory_()

    # Events for synchronization
    ready_event = mp.Event()  # GPU -> CPU
    done_event = mp.Event()   # CPU -> GPU

    # Launch processes
    p1 = mp.Process(target=gpu_worker, args=(shared_input, shared_output, ready_event, done_event))
    p2 = mp.Process(target=cpu_worker, args=(shared_input, shared_output, ready_event, done_event))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
