import time
import torch
import torch.multiprocessing as mp

M = 1024
# M = 8
K = 7168
dtype = torch.bfloat16
NUM_REPEATS = 1000  # number of times to repeat for averaging

# directly_write_to_shared_output = True
directly_write_to_shared_output = False
flush = True
# flush = False

a = torch.ones(256 * 1024 * 1024 // 4, dtype=torch.float)
b = torch.ones(256 * 1024 * 1024 // 4, dtype=torch.float)
def flush():
    global a, b
    a += b

def gpu_worker(shared_input, shared_output, ready_event, done_event):
    # === Step 1. GPU writes tensor into shared pinned input ===
    gpu_tensor = torch.randn(shared_input.numel(), device="cuda", dtype=shared_input.dtype).reshape(shared_input.size())
    
    # Create CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    cpu_times = []
    copy_times = []

    for i in range(NUM_REPEATS):
        if flush:
            flush()
        
        # Measure .cpu() time
        start_event.record()
        cpu_tensor = gpu_tensor.cpu()  # device transfer
        end_event.record()
        torch.cuda.synchronize()
        cpu_times.append(start_event.elapsed_time(end_event))

        # Measure copy_ into shared pinned memory
        if flush:
            flush()
        start_event.record()
        shared_input.copy_(cpu_tensor, non_blocking=True)
        end_event.record()
        torch.cuda.synchronize()
        copy_times.append(start_event.elapsed_time(end_event))

    print(f"[GPU] Average GPU -> CPU (.cpu()) time over {NUM_REPEATS} runs: {sum(cpu_times)/NUM_REPEATS:.3f} ms")
    print(f"[GPU] Average copy into shared pinned memory time over {NUM_REPEATS} runs: {sum(copy_times)/NUM_REPEATS:.3f} ms")

    print("[GPU] Wrote tensor into shared pinned input")

    # Signal CPU that input is ready
    ready_event.set()

    # Wait for CPU to finish reduction
    done_event.wait()

    # === Step 3. GPU reads reduced result from shared pinned output ===
    result_times = []
    for i in range(NUM_REPEATS):
        if flush:
            flush()
        start_event.record()
        result_gpu = shared_output.to("cuda", non_blocking=True)
        end_event.record()
        torch.cuda.synchronize()
        result_times.append(start_event.elapsed_time(end_event))

    print(f"[GPU] Average CPU -> GPU copy time over {NUM_REPEATS} runs: {sum(result_times)/NUM_REPEATS:.3f} ms")
    print("[GPU] Read reduced result back to GPU:", result_gpu)


def cpu_worker(shared_input, shared_output, ready_event, done_event):
    # Wait for GPU to signal input is ready
    ready_event.wait()

    # === Step 2. CPU reduces the tensor ===
    print("[CPU] Reading shared input...")

    if directly_write_to_shared_output:
        torch.exp(shared_input, out=shared_output)
    else:
        # Measure the time for shared_output.copy_
        copy_times = []
        for _ in range(NUM_REPEATS):
            s = shared_input.relu()
            if flush:
                flush()
            start = time.perf_counter()
            shared_output.copy_(s)
            end = time.perf_counter()
            copy_times.append((end - start) * 1000)  # convert to milliseconds

        avg_time = sum(copy_times) / NUM_REPEATS
        print(f"[CPU] Average shared_output.copy_(s) time over {NUM_REPEATS} runs: {avg_time:.3f} ms")

    print("[CPU] Wrote reduced result into shared output:", shared_output)

    # Signal GPU that reduction is done
    done_event.set()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # safer for CUDA + multiprocessing

    # Shared pinned input tensor (large data buffer)
    shared_input = torch.empty(M, K, dtype=dtype, pin_memory=True)
    shared_input.share_memory_()

    # Shared pinned output tensor
    shared_output = torch.empty(M, K, dtype=dtype, pin_memory=True)
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
