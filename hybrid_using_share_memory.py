import time
import torch
import torch.multiprocessing as mp

M = 4096
M = 4

K = 7168
topk = 8
dtype = torch.bfloat16
NUM_REPEATS = 1000

flush_cache = True

a = torch.ones(256 * 1024 * 1024 // 4, dtype=torch.float)
b = torch.ones(256 * 1024 * 1024 // 4, dtype=torch.float)
def flush():
    global a, b
    a += b

def gpu_worker(shared_hidden_states, shared_output,
               shared_topk_weights, shared_topk_ids,
               ready_event, done_event, result_queue):
    gpu_hidden_states = torch.randn(shared_hidden_states.shape, device="cuda", dtype=shared_hidden_states.dtype)
    gpu_topk_weights = torch.randn(shared_topk_weights.shape, device="cuda", dtype=shared_topk_weights.dtype)
    gpu_topk_ids = torch.randint(0, K, shared_topk_ids.shape, device="cuda", dtype=shared_topk_ids.dtype)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    results = []

    # ---------------- GPU -> CPU (2 stages) ----------------
    def measure_gpu_to_cpu(name, gpu_tensor, shared_tensor):
        times_stage1, times_stage2 = [], []
        for _ in range(NUM_REPEATS):
            if flush_cache: flush()
            # stage 1: GPU → CPU
            start_event.record()
            cpu_tensor = gpu_tensor.cpu()
            end_event.record()
            torch.cuda.synchronize()
            times_stage1.append(start_event.elapsed_time(end_event))

            if flush_cache: flush()
            # stage 2: CPU → shared
            start = time.perf_counter()
            shared_tensor.copy_(cpu_tensor, non_blocking=True)
            end = time.perf_counter()
            times_stage2.append((end - start) * 1000)

        results.append((f"{name}_gpu_to_cpu", sum(times_stage1)/NUM_REPEATS))
        results.append((f"{name}_cpu_to_shared", sum(times_stage2)/NUM_REPEATS))

    measure_gpu_to_cpu("hidden_states", gpu_hidden_states, shared_hidden_states)
    measure_gpu_to_cpu("topk_weights", gpu_topk_weights, shared_topk_weights)
    measure_gpu_to_cpu("topk_ids", gpu_topk_ids, shared_topk_ids)

    ready_event.set()
    done_event.wait()

    # ---------------- CPU → GPU (2 stages, hidden_states only) ----------------
    # Stage 1 timing comes from cpu_worker via result_queue
    cpu_stage1_time = result_queue.get()

    times_stage2 = []
    for _ in range(NUM_REPEATS):
        if flush_cache: flush()
        start_event.record()
        _ = shared_output.to("cuda", non_blocking=True)
        end_event.record()
        torch.cuda.synchronize()
        times_stage2.append(start_event.elapsed_time(end_event))

    results.append(("hidden_states_cpu_to_shared", cpu_stage1_time))
    results.append(("hidden_states_shared_to_gpu", sum(times_stage2)/NUM_REPEATS))

    # Print CSV
    print("type,time_ms")
    for t, ms in results:
        print(f"{t},{ms:.3f}")


def cpu_worker(shared_hidden_states, shared_output,
               shared_topk_weights, shared_topk_ids,
               ready_event, done_event, result_queue):
    ready_event.wait()

    times_stage1 = []
    for _ in range(NUM_REPEATS):
        if flush_cache: flush()

        cpu_tensor = shared_hidden_states.relu()

        start = time.perf_counter()
        shared_output.copy_(cpu_tensor)
        end = time.perf_counter()
        times_stage1.append((end - start) * 1000)

    avg_time_stage1 = sum(times_stage1)/NUM_REPEATS
    result_queue.put(avg_time_stage1)

    done_event.set()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    shared_hidden_states = torch.empty(M, K, dtype=dtype, pin_memory=True).share_memory_()
    shared_output = torch.empty(M, K, dtype=dtype, pin_memory=True).share_memory_()
    shared_topk_weights = torch.empty(M, topk, dtype=torch.float32, pin_memory=True).share_memory_()
    shared_topk_ids = torch.empty(M, topk, dtype=torch.int32, pin_memory=True).share_memory_()

    ready_event = mp.Event()
    done_event = mp.Event()
    result_queue = mp.Queue()

    p1 = mp.Process(target=gpu_worker,
                    args=(shared_hidden_states, shared_output,
                          shared_topk_weights, shared_topk_ids,
                          ready_event, done_event, result_queue))
    p2 = mp.Process(target=cpu_worker,
                    args=(shared_hidden_states, shared_output,
                          shared_topk_weights, shared_topk_ids,
                          ready_event, done_event, result_queue))

    p1.start()
    p2.start()
    p1.join()
    p2.join()
