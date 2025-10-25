#!/bin/bash

# 6B
sed -i '/# replace t_sin$/c\        a = 0.00462 # replace t_sin' vllm/core/scheduler_v2.py
sed -i '/# replace t_offset$/c\        b = 108.99 # replace t_offset' vllm/core/scheduler_v2.py
sed -i '/# replace f_ch a$/c\        a = 0.0408 # replace f_ch a' vllm/core/scheduler_v2.py
sed -i '/# replace f_ch c$/c\        c = 16.92 # replace f_ch c' vllm/core/scheduler_v2.py
sed -i '/# replace t_sin$/c\        a = 0.00462 # replace t_sin' vllm/core/scheduler_v2_infercept.py
sed -i '/# replace t_offset$/c\        b = 108.99 # replace t_offset' vllm/core/scheduler_v2_infercept.py
sed -i '/# replace f_ch a$/c\        a = 0.0408 # replace f_ch a' vllm/core/scheduler_v2_infercept.py
sed -i '/# replace f_ch c$/c\        c = 16.92 # replace f_ch c' vllm/core/scheduler_v2_infercept.py


# MARS, single API, performance, qps 3, 4, 5 and 6
CUDA_VISIBLE_DEVICES=3 python benchmarks/fixed_final_tput_bench_real.py --load-format dummy --qps 3.0 --msg results/single_api/6B_3.0_1800_MARS --window 1800 --api-policy V --policy-config V2 --chunk-fill --swap-space 32 --starvation-avoidance True --starvation-threshold 100 --starvation-quantum 100000 --exp-json diverse_oneapi_merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=3 python benchmarks/fixed_final_tput_bench_real.py --load-format dummy --qps 4.0 --msg results/single_api/6B_4.0_1800_MARS --window 1800 --api-policy V --policy-config V2 --chunk-fill --swap-space 32 --starvation-avoidance True --starvation-threshold 100 --starvation-quantum 100000 --exp-json diverse_oneapi_merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=3 python benchmarks/fixed_final_tput_bench_real.py --load-format dummy --qps 5.0 --msg results/single_api/6B_5.0_1800_MARS --window 1800 --api-policy V --policy-config V2 --chunk-fill --swap-space 32 --starvation-avoidance True --starvation-threshold 100 --starvation-quantum 100000 --exp-json diverse_oneapi_merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=3 python benchmarks/fixed_final_tput_bench_real.py --load-format dummy --qps 6.0 --msg results/single_api/6B_6.0_1800_MARS --window 1800 --api-policy V --policy-config V2 --chunk-fill --swap-space 32 --starvation-avoidance True --starvation-threshold 100 --starvation-quantum 100000 --exp-json diverse_oneapi_merged_exp_uniform.json

# InferCept, single API, performance, qps 3, 4, 5 and 6
CUDA_VISIBLE_DEVICES=3 python benchmarks/fixed_final_tput_bench_real.py --load-format dummy --qps 3.0 --msg results/single_api/6B_3.0_1800_InferCept --window 1800 --api-policy I --policy-config fcfs --chunk-fill --swap-space 32 --exp-json diverse_oneapi_merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=3 python benchmarks/fixed_final_tput_bench_real.py --load-format dummy --qps 4.0 --msg results/single_api/6B_4.0_1800_InferCept --window 1800 --api-policy I --policy-config fcfs --chunk-fill --swap-space 32 --exp-json diverse_oneapi_merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=3 python benchmarks/fixed_final_tput_bench_real.py --load-format dummy --qps 5.0 --msg results/single_api/6B_5.0_1800_InferCept --window 1800 --api-policy I --policy-config fcfs --chunk-fill --swap-space 32 --exp-json diverse_oneapi_merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=3 python benchmarks/fixed_final_tput_bench_real.py --load-format dummy --qps 6.0 --msg results/single_api/6B_6.0_1800_InferCept --window 1800 --api-policy I --policy-config fcfs --chunk-fill --swap-space 32 --exp-json diverse_oneapi_merged_exp_uniform.json


# vanilla vLLM, single API, performance, qps 3, 4, 5 and 6
sed -i '/# switch it for vanilla discard$/c\            self.waiting.append(seq_group) # switch it for vanilla discard' vllm/core/scheduler.py
CUDA_VISIBLE_DEVICES=3 python benchmarks/fixed_final_tput_bench_real.py --load-format dummy --qps 3.0 --msg results/single_api/6B_3.0_1800_vllm --window 1800 --api-policy D --policy-config fcfs --exp-json diverse_oneapi_merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=3 python benchmarks/fixed_final_tput_bench_real.py --load-format dummy --qps 4.0 --msg results/single_api/6B_4.0_1800_vllm --window 1800 --api-policy D --policy-config fcfs --exp-json diverse_oneapi_merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=3 python benchmarks/fixed_final_tput_bench_real.py --load-format dummy --qps 5.0 --msg results/single_api/6B_5.0_1800_vllm --window 1800 --api-policy D --policy-config fcfs --exp-json diverse_oneapi_merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=3 python benchmarks/fixed_final_tput_bench_real.py --load-format dummy --qps 6.0 --msg results/single_api/6B_6.0_1800_vllm --window 1800 --api-policy D --policy-config fcfs --exp-json diverse_oneapi_merged_exp_uniform.json