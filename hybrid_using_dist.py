import torch
import torch.distributed as dist
import os

def hybrid_all_reduce(rank, world_size, gpu_ranks, cpu_ranks):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # GPU ranks
    if rank in gpu_ranks:
        torch.cuda.set_device(rank)  # simple: 1 GPU per rank
        tensor = torch.ones(3, device="cuda") * rank

        # Step 1: GPU-only all_reduce
        gpu_group = dist.new_group(ranks=gpu_ranks, backend="nccl")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=gpu_group)
        print(f"[GPU {rank}] after GPU all_reduce: {tensor}")

        # Step 2: only rank 0 GPU sends to CPUs
        if rank == gpu_ranks[0]:
            cpu_tensor = tensor.cpu()
            for dst in cpu_ranks:
                dist.send(cpu_tensor, dst=dst)

        # Step 4: receive CPU result back
        recv_buf = torch.empty_like(tensor, device="cpu")
        dist.recv(recv_buf, src=cpu_ranks[0])
        tensor = recv_buf.cuda()
        print(f"[GPU {rank}] final result: {tensor}")

    # CPU ranks
    elif rank in cpu_ranks:
        tensor = torch.ones(3) * rank

        # Step 2: receive reduced GPU tensor from rank 0 GPU
        recv_buf = torch.empty(3)
        dist.recv(recv_buf, src=gpu_ranks[0])
        tensor += recv_buf
        print(f"[CPU {rank}] received GPU tensor: {recv_buf}")

        # Step 3: CPU-only all_reduce
        cpu_group = dist.new_group(ranks=cpu_ranks, backend="gloo")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=cpu_group)
        print(f"[CPU {rank}] after CPU all_reduce: {tensor}")

        # Step 4: send CPU result back to GPUs
        if rank == cpu_ranks[0]:
            for dst in gpu_ranks:
                dist.send(tensor, dst=dst)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    gpu_ranks = [0, 1]
    cpu_ranks = [2, 3]
    hybrid_all_reduce(rank, world_size, gpu_ranks, cpu_ranks)