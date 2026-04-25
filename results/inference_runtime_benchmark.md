# Runtime Benchmark Results

## Table A: Machine details

| host | CPU | RAM (GB) | GPU | OS | Python | PyTorch | CUDA |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ruby | Intel(R) Xeon(R) Gold 6258R CPU @ 2.70GHz | 1510.04 | NVIDIA A100-PCIE-40GB, NVIDIA A100-PCIE-40GB, NVIDIA A100-PCIE-40GB, NVIDIA A100-PCIE-40GB, NVIDIA A100-PCIE-40GB, NVIDIA A100-PCIE-40GB, NVIDIA A100-PCIE-40GB, NVIDIA A100-PCIE-40GB | Linux-4.18.0-513.24.1.el8_9.x86_64-x86_64-with-glibc2.28 | 3.9.23 | 2.8.0+cu128 | 12.8 |

## Table B: RTM baseline timings

| method | device | batch_size | latency_ms | throughput_sps |
| --- | --- | --- | --- | --- |
| 6S_cpu | cpu | 1 | 0.7569 | 1321.1233 |
| libRadtran_cpu | cpu | 1 | 37330.3574 | 0.0268 |

## Table C: CPU latency comparison (batch_size=1 only)

| method | fidelity_mode | model | batch_size | latency_ms | speedup_vs_libradtran |
| --- | --- | --- | --- | --- | --- |
| oracle_residual_axis-kan__m-baseline__k-baseline__act-relu__ln-0_pkan_cpu | oracle_residual | pkan | 1 | 95.6849 | 390.1385 |
| oracle_residual_axis-kan__m-baseline__k-large_deep__act-relu__ln-0_pkan_cpu | oracle_residual | pkan | 1 | 116.4518 | 320.5650 |
| oracle_residual_axis-kan__m-baseline__k-small__act-relu__ln-0_pkan_cpu | oracle_residual | pkan | 1 | 120.5471 | 309.6744 |
| oracle_residual_axis-kan__m-baseline__k-small__act-relu__ln-0_kan_cpu | oracle_residual | kan | 1 | 136.0292 | 274.4290 |
| oracle_residual_axis-kan__m-baseline__k-large_deep__act-relu__ln-0_kan_cpu | oracle_residual | kan | 1 | 144.9921 | 257.4648 |
| oracle_residual_axis-kan__m-baseline__k-baseline__act-relu__ln-0_kan_cpu | oracle_residual | kan | 1 | 169.1744 | 220.6619 |
| oracle_residual_axis-kan__m-baseline__k-large_deep__act-relu__ln-0_kan_cpu | oracle_residual | kan | 1 | 180.8518 | 206.4141 |
| oracle_residual_axis-kan__m-baseline__k-large_deep__act-relu__ln-0_pkan_cpu | oracle_residual | pkan | 1 | 181.6499 | 205.5071 |
| oracle_residual_axis-kan__m-baseline__k-balanced_deep__act-relu__ln-0_pkan_cpu | oracle_residual | pkan | 1 | 184.3507 | 202.4964 |
| oracle_residual_axis-kan__m-baseline__k-balanced_deep__act-relu__ln-0_kan_cpu | oracle_residual | kan | 1 | 184.7729 | 202.0337 |
| oracle_residual_axis-kan__m-baseline__k-baseline__act-relu__ln-0_pkan_cpu | oracle_residual | pkan | 1 | 194.6659 | 191.7663 |
| oracle_residual_axis-kan__m-baseline__k-baseline__act-relu__ln-0_kan_cpu | oracle_residual | kan | 1 | 219.3676 | 170.1726 |
| oracle_residual_axis-kan__m-baseline__k-shared_trunk_multihead__act-relu__ln-0_pkan_cpu | oracle_residual | pkan | 1 | 248.1148 | 150.4560 |
| oracle_residual_axis-kan__m-baseline__k-shared_trunk_multihead__act-relu__ln-0_pkan_cpu | oracle_residual | pkan | 1 | 251.0700 | 148.6850 |
| oracle_residual_axis-kan__m-baseline__k-shared_trunk_multihead__act-relu__ln-0_kan_cpu | oracle_residual | kan | 1 | 281.9728 | 132.3899 |
| oracle_residual_axis-kan__m-baseline__k-small__act-relu__ln-0_kan_cpu | oracle_residual | kan | 1 | 323.1533 | 115.5190 |
| oracle_residual_axis-kan__m-baseline__k-small__act-relu__ln-0_pkan_cpu | oracle_residual | pkan | 1 | 326.5278 | 114.3252 |
| oracle_residual_axis-kan__m-baseline__k-balanced_deep__act-relu__ln-0_kan_cpu | oracle_residual | kan | 1 | 501.3980 | 74.4526 |
| oracle_residual_axis-kan__m-baseline__k-balanced_deep__act-relu__ln-0_pkan_cpu | oracle_residual | pkan | 1 | 507.4091 | 73.5705 |
| oracle_residual_axis-kan__m-baseline__k-shared_trunk_multihead__act-relu__ln-0_kan_cpu | oracle_residual | kan | 1 | 568.8873 | 65.6200 |

## Table D: GPU latency comparison (batch_size=1 only)

| method | fidelity_mode | model | device | batch_size | latency_ms | speedup_vs_libradtran |
| --- | --- | --- | --- | --- | --- | --- |
| oracle_residual_axis-kan__m-baseline__k-baseline__act-relu__ln-0_kan_cuda-7 | oracle_residual | kan | cuda:7 | 1 | 2.9429 | 12684.7101 |
| oracle_residual_axis-kan__m-baseline__k-small__act-relu__ln-0_kan_cuda-7 | oracle_residual | kan | cuda:7 | 1 | 3.0161 | 12376.8952 |
| oracle_residual_axis-kan__m-baseline__k-baseline__act-relu__ln-0_kan_cuda-7 | oracle_residual | kan | cuda:7 | 1 | 3.3937 | 11000.0607 |
| oracle_residual_axis-kan__m-baseline__k-baseline__act-relu__ln-0_pkan_cuda-7 | oracle_residual | pkan | cuda:7 | 1 | 3.3937 | 10999.8316 |
| oracle_residual_axis-kan__m-baseline__k-baseline__act-relu__ln-0_pkan_cuda-7 | oracle_residual | pkan | cuda:7 | 1 | 3.4129 | 10938.1007 |
| oracle_residual_axis-kan__m-baseline__k-small__act-relu__ln-0_pkan_cuda-7 | oracle_residual | pkan | cuda:7 | 1 | 3.5706 | 10454.9892 |
| oracle_residual_axis-kan__m-baseline__k-balanced_deep__act-relu__ln-0_kan_cuda-7 | oracle_residual | kan | cuda:7 | 1 | 3.8292 | 9748.9391 |
| oracle_residual_axis-kan__m-baseline__k-small__act-relu__ln-0_pkan_cuda-7 | oracle_residual | pkan | cuda:7 | 1 | 3.8361 | 9731.2492 |
| oracle_residual_axis-kan__m-baseline__k-small__act-relu__ln-0_kan_cuda-7 | oracle_residual | kan | cuda:7 | 1 | 3.8659 | 9656.3069 |
| oracle_residual_axis-kan__m-baseline__k-balanced_deep__act-relu__ln-0_pkan_cuda-7 | oracle_residual | pkan | cuda:7 | 1 | 3.9303 | 9498.2142 |
| oracle_residual_axis-kan__m-baseline__k-large_deep__act-relu__ln-0_pkan_cuda-7 | oracle_residual | pkan | cuda:7 | 1 | 4.0468 | 9224.7503 |
| oracle_residual_axis-kan__m-baseline__k-large_deep__act-relu__ln-0_kan_cuda-7 | oracle_residual | kan | cuda:7 | 1 | 4.0610 | 9192.4744 |
| oracle_residual_axis-kan__m-baseline__k-large_deep__act-relu__ln-0_pkan_cuda-7 | oracle_residual | pkan | cuda:7 | 1 | 4.2259 | 8833.7451 |
| oracle_residual_axis-kan__m-baseline__k-large_deep__act-relu__ln-0_kan_cuda-7 | oracle_residual | kan | cuda:7 | 1 | 4.6205 | 8079.2266 |
| oracle_residual_axis-kan__m-baseline__k-balanced_deep__act-relu__ln-0_kan_cuda-7 | oracle_residual | kan | cuda:7 | 1 | 4.6736 | 7987.5043 |
| oracle_residual_axis-kan__m-baseline__k-balanced_deep__act-relu__ln-0_pkan_cuda-7 | oracle_residual | pkan | cuda:7 | 1 | 4.9122 | 7599.4640 |
| oracle_residual_axis-kan__m-baseline__k-shared_trunk_multihead__act-relu__ln-0_kan_cuda-7 | oracle_residual | kan | cuda:7 | 1 | 7.5535 | 4942.1204 |
| oracle_residual_axis-kan__m-baseline__k-shared_trunk_multihead__act-relu__ln-0_kan_cuda-7 | oracle_residual | kan | cuda:7 | 1 | 8.6812 | 4300.1459 |
| oracle_residual_axis-kan__m-baseline__k-shared_trunk_multihead__act-relu__ln-0_pkan_cuda-7 | oracle_residual | pkan | cuda:7 | 1 | 8.8590 | 4213.8198 |
| oracle_residual_axis-kan__m-baseline__k-shared_trunk_multihead__act-relu__ln-0_pkan_cuda-7 | oracle_residual | pkan | cuda:7 | 1 | 9.3540 | 3990.8654 |

## Table E: Throughput comparison (best batch size per method; amortized, not batch_size=1 latency)

| method | fidelity_mode | model | device | batch_size | total_elapsed_sec | latency_ms | samples_per_sec |
| --- | --- | --- | --- | --- | --- | --- | --- |
| oracle_residual_axis-kan__m-baseline__k-small__act-relu__ln-0_kan_cuda-7 | oracle_residual | kan | cuda:7 | 128 | 0.005243 | 0.0205 | 48827.10 |
| oracle_residual_axis-kan__m-baseline__k-baseline__act-relu__ln-0_pkan_cuda-7 | oracle_residual | pkan | cuda:7 | 128 | 0.005290 | 0.0207 | 48396.57 |
| oracle_residual_axis-kan__m-baseline__k-baseline__act-relu__ln-0_kan_cuda-7 | oracle_residual | kan | cuda:7 | 128 | 0.005904 | 0.0231 | 43362.48 |
| oracle_residual_axis-kan__m-baseline__k-large_deep__act-relu__ln-0_kan_cuda-7 | oracle_residual | kan | cuda:7 | 128 | 0.006606 | 0.0258 | 38753.67 |
| oracle_residual_axis-kan__m-baseline__k-small__act-relu__ln-0_pkan_cuda-7 | oracle_residual | pkan | cuda:7 | 128 | 0.006745 | 0.0263 | 37952.45 |
| oracle_residual_axis-kan__m-baseline__k-balanced_deep__act-relu__ln-0_kan_cuda-7 | oracle_residual | kan | cuda:7 | 128 | 0.007675 | 0.0300 | 33357.20 |
| oracle_residual_axis-kan__m-baseline__k-large_deep__act-relu__ln-0_pkan_cuda-7 | oracle_residual | pkan | cuda:7 | 128 | 0.007770 | 0.0304 | 32945.36 |
| oracle_residual_axis-kan__m-baseline__k-shared_trunk_multihead__act-relu__ln-0_kan_cuda-7 | oracle_residual | kan | cuda:7 | 128 | 0.013118 | 0.0512 | 19515.72 |
| oracle_residual_axis-kan__m-baseline__k-balanced_deep__act-relu__ln-0_pkan_cuda-7 | oracle_residual | pkan | cuda:7 | 128 | 0.013436 | 0.0525 | 19053.91 |
| oracle_residual_axis-kan__m-baseline__k-shared_trunk_multihead__act-relu__ln-0_pkan_cuda-7 | oracle_residual | pkan | cuda:7 | 128 | 0.023857 | 0.0932 | 10730.59 |
| oracle_residual_axis-kan__m-baseline__k-baseline__act-relu__ln-0_pkan_cpu | oracle_residual | pkan | cpu | 64 | 3.728282 | 14.5636 | 68.66 |
| oracle_residual_axis-kan__m-baseline__k-balanced_deep__act-relu__ln-0_kan_cpu | oracle_residual | kan | cpu | 128 | 5.113994 | 19.9765 | 50.06 |
| oracle_residual_axis-kan__m-baseline__k-baseline__act-relu__ln-0_kan_cpu | oracle_residual | kan | cpu | 64 | 5.289959 | 20.6639 | 48.39 |
| oracle_residual_axis-kan__m-baseline__k-large_deep__act-relu__ln-0_kan_cpu | oracle_residual | kan | cpu | 128 | 5.660469 | 22.1112 | 45.23 |
| oracle_residual_axis-kan__m-baseline__k-large_deep__act-relu__ln-0_pkan_cpu | oracle_residual | pkan | cpu | 128 | 5.775668 | 22.5612 | 44.32 |
| oracle_residual_axis-kan__m-baseline__k-balanced_deep__act-relu__ln-0_pkan_cpu | oracle_residual | pkan | cpu | 64 | 7.075593 | 27.6390 | 36.18 |
| oracle_residual_axis-kan__m-baseline__k-shared_trunk_multihead__act-relu__ln-0_kan_cpu | oracle_residual | kan | cpu | 128 | 8.266338 | 32.2904 | 30.97 |
| oracle_residual_axis-kan__m-baseline__k-small__act-relu__ln-0_kan_cpu | oracle_residual | kan | cpu | 128 | 8.858083 | 34.6019 | 28.90 |
| oracle_residual_axis-kan__m-baseline__k-small__act-relu__ln-0_pkan_cpu | oracle_residual | pkan | cpu | 128 | 9.041113 | 35.3168 | 28.32 |
| oracle_residual_axis-kan__m-baseline__k-shared_trunk_multihead__act-relu__ln-0_pkan_cpu | oracle_residual | pkan | cpu | 128 | 25.407298 | 99.2473 | 10.08 |
