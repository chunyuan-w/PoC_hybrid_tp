export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so:${CONDA_PREFIX}/lib/libiomp5.so

numactl --physcpubind=0-27 --membind=0 python -u hybrid_using_share_memory.py 