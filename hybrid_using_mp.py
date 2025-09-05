import multiprocessing
import torch
import torch.distributed as dist

# ---------------------- Config classes (must be top-level) ----------------------
class ServerArgs:
    tp_size_gpu = 1
    tp_size_cpu = 4
    H = 1024
    B = 8

class PortArgs:
    pass

class BenchArgs:
    pass

# ---------------------- Helper classes ----------------------
class TensorParallelShard:
    def __init__(self, full_dim, tp_rank, tp_size, device):
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.device = device
        shard_size = full_dim // tp_size
        torch.manual_seed(42 + tp_rank)
        self.weight_shard = torch.randn(full_dim, shard_size, device=device)
    
    def forward_shard(self, x):
        return x @ self.weight_shard

def tp_forward(x, shard_obj, tp_group):
    y_local = shard_obj.forward_shard(x)
    world_size = dist.get_world_size(tp_group)
    gather_list = [torch.empty_like(y_local) for _ in range(world_size)]
    dist.all_gather(gather_list, y_local, group=tp_group)
    y_full = torch.cat(gather_list, dim=1)
    return y_full

# ---------------------- CPU TP worker ----------------------
def cpu_tp_worker(server_args, port_args, bench_args, tp_rank, q_in, q_out):
    H = server_args.H
    TP_cpu = server_args.tp_size_cpu
    dist.init_process_group("gloo", init_method=port_args,
                            rank=tp_rank, world_size=TP_cpu)
    device = torch.device("cpu")
    shard = TensorParallelShard(H, tp_rank, TP_cpu, device)
    print(f"[CPU rank {tp_rank}] started")

    while True:
        item = q_in.get()
        if item is None:
            break
        req_id, x_chunk = item
        y_full = tp_forward(x_chunk, shard, dist.group.WORLD)
        q_out.put((req_id, y_full))
        print(f"[CPU rank {tp_rank}] processed chunk {req_id}")

    print(f"[CPU rank {tp_rank}] finished")
    dist.destroy_process_group()

# ---------------------- GPU TP worker ----------------------
def gpu_worker(server_args, port_args, bench_args, tp_rank):
    TP_gpu = server_args.tp_size_gpu
    TP_cpu = server_args.tp_size_cpu
    H = server_args.H
    B = server_args.B

    device = torch.device(f"cuda:{tp_rank}")
    dist.init_process_group(
        backend="nccl",
        init_method=port_args,
        world_size=TP_gpu,
        rank=tp_rank
    )
    torch.cuda.set_device(device)
    shard = TensorParallelShard(H, tp_rank, TP_gpu, device)
    print(f"[GPU rank {tp_rank}] started")

    # Backbone forward
    x = torch.randn(B, H, device=device)
    y_gpu = tp_forward(x, shard, dist.group.WORLD)
    print(f"[GPU rank {tp_rank}] completed backbone forward")

    # Offload MoE to CPU
    if tp_rank == 0:
        ctx = multiprocessing.get_context("spawn")
        q_in = ctx.Queue()
        q_out = ctx.Queue()

        # Spawn CPU TP ranks
        cpu_port = "tcp://127.0.0.1:29501"
        cpu_procs = []
        for r in range(TP_cpu):
            p = ctx.Process(target=cpu_tp_worker,
                            args=(server_args, cpu_port, bench_args, r, q_in, q_out))
            p.start()
            cpu_procs.append(p)

        # Send chunks to CPU
        chunks = y_gpu.cpu().chunk(TP_cpu, dim=0)
        req_id = 0
        for ch in chunks:
            q_in.put((req_id, ch))

        # Collect results
        results = []
        for _ in range(TP_cpu):
            rid, y_chunk = q_out.get()
            results.append(y_chunk)
        y_moe = torch.cat(results, dim=0).to(device)

        # Shutdown CPU workers
        for _ in range(TP_cpu):
            q_in.put(None)
        for p in cpu_procs:
            p.join()
        print(f"[GPU rank {tp_rank}] completed CPU MoE offload")
    else:
        # Other GPU ranks send/receive via rank 0
        dist.send(y_gpu.cpu(), dst=0)
        y_moe = torch.empty_like(y_gpu).to(device)
        dist.recv(y_moe.cpu(), src=0)

    # Continue GPU computation
    z = y_moe + 1.0
    print(f"[GPU rank {tp_rank}] final output mean: {z.mean().item():.4f}")

    dist.barrier()
    dist.destroy_process_group()
    print(f"[GPU rank {tp_rank}] finished")

# ---------------------- Launcher ----------------------
def launch_mixed(server_args, port_args, bench_args):
    if server_args.tp_size_gpu == 1:
        gpu_worker(server_args, port_args, bench_args, 0)
    else:
        workers = []
        for tp_rank in range(server_args.tp_size_gpu):
            proc = multiprocessing.Process(
                target=gpu_worker,
                args=(server_args, port_args, bench_args, tp_rank)
            )
            proc.start()
            workers.append(proc)

        for proc in workers:
            proc.join()
            proc.terminate()

# ---------------------- Main ----------------------
if __name__ == "__main__":
    server_args = ServerArgs()
    port_args = "tcp://127.0.0.1:29500"
    bench_args = BenchArgs()

    launch_mixed(server_args, port_args, bench_args)
