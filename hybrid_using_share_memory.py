import torch
import torch.multiprocessing as mp

def gpu_worker(shared_input, shared_output, ready_event, done_event):
    # === Step 1. GPU writes tensor into shared pinned input ===
    gpu_tensor = torch.arange(shared_input.numel(), device="cuda", dtype=shared_input.dtype)
    shared_input.copy_(gpu_tensor.cpu(), non_blocking=True)
    print("[GPU] Wrote tensor into shared pinned input")

    # Signal CPU that input is ready
    ready_event.set()

    # Wait for CPU to finish reduction
    done_event.wait()

    # === Step 3. GPU reads reduced result from shared pinned output ===
    result_gpu = shared_output.to("cuda", non_blocking=True)
    print("[GPU] Read reduced result back to GPU:", result_gpu.item())


def cpu_worker(shared_input, shared_output, ready_event, done_event):
    # Wait for GPU to signal input is ready
    ready_event.wait()

    # === Step 2. CPU reduces the tensor ===
    print("[CPU] Reading shared input...")
    s = shared_input.sum()
    shared_output[0] = s  # write reduced result into shared pinned output
    print("[CPU] Wrote reduced result into shared output:", s.item())

    # Signal GPU that reduction is done
    done_event.set()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # safer for CUDA + multiprocessing

    # Shared pinned input tensor (large data buffer)
    shared_input = torch.empty(1024, dtype=torch.float32, pin_memory=True)
    # Make shared_input accessible by multiple processes
    shared_input.share_memory_()

    # Shared pinned output tensor (just 1 value for reduction result)
    shared_output = torch.empty(1, dtype=torch.float32, pin_memory=True)
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
