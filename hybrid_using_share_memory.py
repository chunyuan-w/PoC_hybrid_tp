import torch
import torch.multiprocessing as mp

M = 1024
K = 7168
dtype = torch.bfloat16

def gpu_worker(shared_input, shared_output, ready_event, done_event):
    # === Step 1. GPU writes tensor into shared pinned input ===
    gpu_tensor = torch.randn(shared_input.numel(), device="cuda", dtype=shared_input.dtype).reshape(shared_input.size())
    
    # Create CUDA events for timing
    start_event_cpu = torch.cuda.Event(enable_timing=True)
    end_event_cpu = torch.cuda.Event(enable_timing=True)
    start_event_copy = torch.cuda.Event(enable_timing=True)
    end_event_copy = torch.cuda.Event(enable_timing=True)

    # TODO: directly write the result of .cpu() into shared_input?
    
    # Record start of .cpu()
    start_event_cpu.record()
    cpu_tensor = gpu_tensor.cpu()  # device transfer
    end_event_cpu.record()
    torch.cuda.synchronize()
    print(f"[GPU] GPU -> CPU (.cpu()) took {start_event_cpu.elapsed_time(end_event_cpu):.3f} ms")

    # Record start of copy_ into shared pinned memory
    start_event_copy.record()
    shared_input.copy_(cpu_tensor, non_blocking=True)
    end_event_copy.record()
    torch.cuda.synchronize()
    print(f"[GPU] Copy into shared pinned memory took {start_event_copy.elapsed_time(end_event_copy):.3f} ms")

    print("[GPU] Wrote tensor into shared pinned input")

    # Signal CPU that input is ready
    ready_event.set()

    # Wait for CPU to finish reduction
    done_event.wait()

    # === Step 3. GPU reads reduced result from shared pinned output ===
    start_event_copy.record()
    result_gpu = shared_output.to("cuda", non_blocking=True)
    end_event_copy.record()
    torch.cuda.synchronize()
    print(f"[GPU] CPU -> GPU copy took {start_event_copy.elapsed_time(end_event_copy):.3f} ms")

    print("[GPU] Read reduced result back to GPU:", result_gpu)


def cpu_worker(shared_input, shared_output, ready_event, done_event):
    # TODO: bind core and memory here?
    
    # Wait for GPU to signal input is ready
    ready_event.wait()

    # === Step 2. CPU reduces the tensor ===
    print("[CPU] Reading shared input...")
    
    # We need a kernel which directly writes to shared_output
    torch.exp(shared_input, out=shared_output)
    
    print("[CPU] Wrote reduced result into shared output:", shared_output)

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
