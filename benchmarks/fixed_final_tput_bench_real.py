
"""Benchmark offline inference throughput."""
import argparse
import random
import time
from typing import List, Dict, Tuple

import torch
import queue
import threading
from tqdm.rich import tqdm
from vllm import SamplingParams, LLMEngine, EngineArgs, utils
from vllm.outputs import RequestOutput
from vllm.model_executor.layers.attention import CACHE_EVENTS
import numpy as np
import csv
from datetime import datetime
import json
from pathlib import Path
import copy
import os

class Request:
    def __init__(self, request_id, prompt, sampling_params, prompt_token_ids):
        self.request_id = request_id
        self.prompt = prompt
        self.sampling_params = sampling_params
        self.prompt_token_ids = prompt_token_ids
        self.start_time = -1
        self.start_generate = -1
        self.end_time = -1
        self.arrival_time = -1
        self.start_length = 0
        self.end_length = 0
        self.pause_time = -1
        self.resume_time = -1
        self.finished = False
        self.api_times = []
        self.helper_to_calculate_prefill_time_except_for_first_prefill = -1
        self.prefill_time_except_for_first_prefill = 0

class APIExecutor:
    def __init__(self) -> None:
        self._queue = queue.Queue()
        self._total_apis = 0
        self.exp_json = {}
        self.exp_status = {}
        self.total_output_toks = {}
        self.args = None
        self.stop = None
    
    def _add_task(self, request_id: str, seq_id: int, api_time: float, ret_len: int):
        time.sleep(api_time)
        self._queue.put((request_id, seq_id, ret_len))
    
    def add_task(self, request_id: str, seq_id: int, api_time: float, ret_len: int):
        task = threading.Thread(target=self._add_task, args=(request_id, seq_id, api_time, ret_len))
        task.start()
        self._total_apis += 1
        return task
    
    def _get_results(self) -> Dict[str, Dict[int, int]]:
        results = {}
        current_num_ret = self._queue.qsize()
        for _ in range(current_num_ret):
            request_id, seq_id, ret_len = self._queue.get()
            if request_id not in results:
                results[request_id] = {}
            results[request_id][seq_id] = ret_len
        return results
    
    def get_new_sampling_params(self, request_id: str) -> Tuple[SamplingParams, int]:
        if request_id not in self.exp_status:
            self.exp_status[request_id] = 0
        else:
            self.exp_status[request_id] += 1
        
        experiments = self.exp_json[request_id]
        experiment_num = self.exp_status[request_id]
        # Format of each experiment
        if experiment_num >= len(experiments):
            experiment = {"prompt_tokens": 0, "completion_tokens": 0, "api_time": 0, 'api_token_length': 0}
        else:
            experiment = experiments[experiment_num]
        if "api_time" not in experiment:
            assert "api_token_length" not in experiment
            experiment["api_time"] = 0
            experiment["api_token_length"] = 0
        prompt_size, completion_tokens, api_exec_time, api_return_len = experiment["prompt_tokens"], experiment["completion_tokens"], experiment["api_time"], experiment["api_token_length"]
        api_invoke_interval = completion_tokens

        # added
        remain_length = 0
        for i in range(experiment_num+1, len(experiments)):
            remain_length += experiment.get("completion_tokens", 0)
        if api_exec_time == 0 and api_return_len == 0:
            api_max_calls = 0
        else:
            api_max_calls = 1
        
        new_sampling_params = SamplingParams(
            n=1,
            temperature=0.0,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=min(2048, self.total_output_toks[request_id]),
            
            stop=self.stop,
            use_api_simulator=True,
            
            api_return_length=api_return_len,
            api_invoke_interval=api_invoke_interval,
            api_exec_time=api_exec_time,
            api_max_calls=api_max_calls,
            remain_length=remain_length,

            # predicted values
            predicted_api_exec_time=max(1e-9, api_exec_time + np.random.normal(0, args.percentage_error_api_exec_time * api_exec_time)), # api length
            predicted_api_invoke_interval=max(1, api_invoke_interval + np.random.normal(0, args.percentage_error_request_length_before_api * api_invoke_interval)) # request length before api
        )
        return new_sampling_params, prompt_size
    
    def resume(self, vllm_engine: LLMEngine, requests, start_measure) -> None:
        api_rets = self._get_results()
        resume_time = time.perf_counter()
        for request_id, seq_id_to_ret_len in api_rets.items():
            begin_time = time.perf_counter()
            response = {}
            if start_measure:
                r: Request = requests[int(request_id)]
                r.resume_time = resume_time
                r.api_times.append((r.pause_time, r.resume_time))
                r.helper_to_calculate_prefill_time_except_for_first_prefill = r.resume_time
                r.pause_time = -1
                r.resume_time = -1
            for seq_id, ret_len in seq_id_to_ret_len.items():
                response[seq_id] = [0] * ret_len
            sampling_params, _ = self.get_new_sampling_params(request_id)
            vllm_engine.resume_request(request_id, response, sampling_params)
            end_time = time.perf_counter()
    
    def generate_exec_times(self, distro, num_prompts, seed):
        rng = np.random.default_rng(seed)
        if distro == 'N':
            # normal distribution
            return np.abs(rng.normal(loc=11, scale=3, size=(num_prompts,)))
        elif distro == 'U':
            # uniform distribution
            return rng.uniform(low=0.1, high=20, size=(num_prompts,))
        else:
            # Generate random numbers from gamma distribution
            right = np.abs(rng.gamma(shape=0.5, scale=4, size=(num_prompts,)))  # shorter api times
            left = np.abs(20-right)                                             # longer api times
            if distro == 'L':
                return left
            elif distro == 'R':
                return right
            elif distro == 'B':
                return np.concatenate([rng.choice(left, num_prompts//2),
                                       rng.choice(right, num_prompts//2)])
            else:
                return ValueError(f'Unsupported distribution: {distro}')
            


def run_vllm(
    args: argparse.Namespace,
) -> float:
    global total_input_tokens

    engine_args = EngineArgs.from_cli_args(args)
    engine = LLMEngine.from_engine_args(engine_args)
    if args.no_api:
        stop = []
    else:
        stop = utils.get_api_stop_strings()
    api_engine = APIExecutor()
    api_engine.args = args
    api_engine.stop = stop
    tasks = set()
    
    args.num_prompts = 4 * int(args.qps * args.window)

    # prepare input tokens
    with open(args.exp_json) as f :
        exp_json = json.load(f)

    # Use continuous int as request id now for easy tracking
    if args.no_api:
        keys_to_sample = list(exp_json.keys())
        assert len(keys_to_sample) == args.num_prompts
    else:
        keys_to_sample = random.choices(list(exp_json.keys()), k=args.num_prompts)

    new_exp_json = {}
    new_2_old_key = {}
    for i, key in enumerate(keys_to_sample):
        assert key in exp_json

        new_exp_json[str(i)] = exp_json[key].copy()
        new_2_old_key[i] = key
        
    exp_json = new_exp_json
    api_engine.exp_json = exp_json
    
    prompt_lens = []
    output_lens = []
    num_api_calls = []
    ret_lens = []
    api_exec_times = []

    # Unify chatbot with other workload

    for request_id, experiments in exp_json.items():
        ts = 0
        num_calls = 0
        for i, experiment in enumerate(experiments):

            if "prompt" in experiment and "prompt_tokens" not in experiment:
                experiment["prompt_tokens"] = len(engine.tokenizer.encode(experiment["prompt"]))
                if i == 0 and experiment['prompt_tokens'] >= 2048:
                    print(f'long prompt: {new_2_old_key[int(request_id)]}')
            if i == 0:
                prompt_lens.append(experiment['prompt_tokens'])

            if "completion" in experiment and "completion_tokens" not in experiment:
                experiment["completion_tokens"] = len(engine.tokenizer.encode(experiment["completion"]))
                if experiment['completion_tokens'] >= 2048:
                    print(f'long completion: {new_2_old_key[int(request_id)]}')

            if "api_token" in experiment and "api_token_length" not in experiment:
                experiment["api_token_length"] = len(engine.tokenizer.encode(experiment["api_token"]))
                if experiment["api_token_length"] >= 2048:
                    print(f'long ret: {new_2_old_key[int(request_id)]}')
            
            ts += experiment["completion_tokens"]
            if "api_token_length" in experiment:
                ts += experiment["api_token_length"]
                ret_lens.append(experiment["api_token_length"])
                num_calls += 1
            
            if "api_time" in experiment:
                api_exec_times.append(experiment['api_time'])
            api_engine.total_output_toks[request_id] = ts
        output_lens.append(min(ts, 2048 - prompt_lens[-1]))
        num_api_calls.append(num_calls)
    





    with open(f'./{args.msg}/data_dist.csv', 'w+') as f:
        f.write(f'Prompt Length,Output Length,Number of API Calls,API Return Length,API Execution Time\n')
        for i in range(len(prompt_lens)):
            f.write(f'{prompt_lens[i]},{output_lens[i]},{num_api_calls[i]},{ret_lens[i]},{api_exec_times[i]}\n')
    
    requests: List[Request] = []
    rng = np.random.default_rng(args.seed)
    arrival_times = []
    start_offset = 0

    for request_id, experiments in exp_json.items():
        sampling_params, prompt_size = api_engine.get_new_sampling_params(request_id)

        prompt_token_ids = [0] * prompt_size
        requests.append(Request(
            request_id=request_id,
            prompt=None,
            sampling_params=sampling_params,
            prompt_token_ids=prompt_token_ids
        ))
        offset = rng.exponential(1.0 / args.qps)
        start_offset += offset
        arrival_times.append(start_offset)

    pbar = tqdm(total=args.num_prompts, desc="Processed prompts")
    start = time.perf_counter()

    index = 0

    total_input_tokens = 0

    for r in requests[:int(args.qps)+1]:
        r.arrival_time = start
        total_input_tokens += len(r.prompt_token_ids)
        engine.add_request(r.request_id, r.prompt, r.sampling_params, r.prompt_token_ids, r.arrival_time)
        arrival_times.pop(0)
    index = int(args.qps)+1


    # Run the engine.
    outputs: List[RequestOutput] = []
    iter = 0
    torch.cuda.cudart().cudaProfilerStart()
    tokens_after_start = 0
    tokens_after_stop = 0
    start_measure_time = time.perf_counter()
    num_reqs = 0
    started_measure = True
    cpu_full = False

    last_step_finish_time = time.perf_counter()
    this_step_begin_time = time.perf_counter()

    while engine.has_unfinished_requests() or arrival_times:
        if engine.cpu_full:
            cpu_full = True
        curr_time = time.perf_counter()
        while arrival_times and start + arrival_times[0] <= curr_time:
            r = requests[index]
            r.arrival_time = start + arrival_times[0]
            total_input_tokens += len(r.prompt_token_ids)
            engine.add_request(r.request_id, r.prompt, r.sampling_params, r.prompt_token_ids, r.arrival_time)
            arrival_times.pop(0)
            if started_measure:
                r.start_time = curr_time
                r.start_length = 0
            index += 1
        
        if not engine.has_unfinished_requests:
            begin_time = time.perf_counter()
            api_engine.resume(engine)
            end_time = time.perf_counter()
            continue
        

        this_step_begin_time = time.perf_counter()
        begin_time = time.perf_counter()
        step_outputs = engine.step()
        last_step_finish_time = time.perf_counter()
        end_time = time.perf_counter()

        curr_time = time.perf_counter()

        can_stop = False
        if not started_measure and curr_time-start > 30:
            started_measure = True
        for output in step_outputs:
            r: Request = requests[int(output.request_id)]
            if r.start_generate == -1:
                r.start_generate = curr_time

            if r.helper_to_calculate_prefill_time_except_for_first_prefill != -1:
                r.prefill_time_except_for_first_prefill += curr_time - r.helper_to_calculate_prefill_time_except_for_first_prefill
                r.helper_to_calculate_prefill_time_except_for_first_prefill = -1

            if r.start_time != -1:
                r.end_time = curr_time
                r.end_length = output.outputs[0].token_ids.count(31548) # fix: exclude api outputs ("dummy", the output, is tokenized to 31548)
                
            else:
                if started_measure:
                    r.start_time = curr_time
                    r.start_length = len(output.outputs[0].token_ids)
            if output.finished:
                outputs.append(output)
                r.finished = True

                pbar.update(1)
            if output.paused:
                sampling_params: SamplingParams = engine.scheduler.paused[output.request_id][0].sampling_params

                r.pause_time = curr_time
                
                for (rid, sid) in output.paused:
                    begin_time = time.perf_counter()
                    task = api_engine.add_task(output.request_id, sid, sampling_params.api_exec_time, sampling_params.api_return_length)
                    end_time = time.perf_counter()
                    tasks.add(task)
        
        begin_time = time.perf_counter()
        api_engine.resume(engine, requests, started_measure)
        end_time = time.perf_counter()

        iter += 1
        if curr_time-start > args.window:
            can_stop = True
        if can_stop:
            break
    tokens_after_stop = engine.scheduler.get_tokens_have_seen() + sum(len(output.prompt_token_ids) + len(output.outputs[0].token_ids) for output in outputs)
    times_before_processing_dict = engine.scheduler.get_times_before_processing()
    times_before_processing = list(times_before_processing_dict.values())
    strategy_used_by_request_dict = engine.scheduler.get_strategy_used_by_request()
    predicted_waste_by_request_dict = engine.scheduler.get_predicted_waste()
    newest_sort_waste_dict = engine.scheduler.get_newest_sort_waste()

    num_reqs = len(outputs)
    torch.cuda.cudart().cudaProfilerStop()
    end = time.perf_counter()
    elapsed_time = end - start_measure_time
    pbar.close()
    actual_tokens = tokens_after_stop - tokens_after_start
    total_generated_tokens = engine.scheduler.get_output_tokens_have_seen() + sum(len(output.outputs[0].token_ids) for output in outputs)
    # parse latencies
    latencies = []
    queueing_latencies = []
    tpots = []
    tpots_dict = {}
    ttfts = []
    ttfts_dict = {}
    waiting_times = []
    waiting_times_dict = {}
    E2E_latencies = []
    E2E_latencies_dict = {}

    with open(f'./{args.msg}/api_trace.csv', 'w+') as t:
        t.write(f'Request ID: Start Time, End Time -- API Times\n')
        for r in requests:
            t.write(f'{r.request_id}: {r.start_time}, {r.end_time}\n -- {r.api_times}\n')

    with open(f'./{args.msg}/all_trace.csv', 'w+') as t:
        t.write(f'Request ID, Finished, Duration, API Time, Queueing, Start Length, End Length, final_norm_latency\n')

        for r in requests:
            if r.start_time == -1:
                continue
            if r.end_time == -1:
                continue
            if r.start_length == r.end_length:
                continue
            if not r.finished:
                continue



            end_time, end_length = r.end_time, r.end_length
            start_time, start_length = r.start_time, r.start_length
            start_generate = r.start_generate

            api_time = 0
            waiting_time = 0
            apis_list = copy.deepcopy(r.api_times)

            while apis_list:
                api_start, api_end = apis_list.pop(0)
                bumped_start = max(api_start, start_time)
                bumped_end = min(api_end, end_time)
                api_time += bumped_end - bumped_start
                waiting_time += bumped_end - bumped_start
            if start_generate != -1:
                queueing = start_generate - start_time
                tpot = (end_time - start_generate - api_time) / end_length
                ttft = start_generate - r.arrival_time
                
                queueing_latencies.append(queueing)
                tpots.append(tpot)
                tpots_dict[int(r.request_id)] = tpot
                ttfts.append(ttft)
                ttfts_dict[int(r.request_id)] = ttft
                waiting_times.append(waiting_time + r.prefill_time_except_for_first_prefill + ttft)
                waiting_times_dict[int(r.request_id)] = waiting_time + r.prefill_time_except_for_first_prefill + ttft

                E2E_latencies.append(end_time - r.arrival_time)
                E2E_latencies_dict[int(r.request_id)] = end_time - r.arrival_time
            else:
                queueing = end - start_time
            
            final_norm_latency = (end_time-start_time-api_time)/(end_length-start_length)
            latencies.append(final_norm_latency)

            t.write(f'{r.request_id}, {r.finished}, {r.end_time-r.start_time}, {api_time}, {queueing}, {start_length}, {end_length}, {final_norm_latency}\n')

    with open(f'./{args.msg}/per_request_log.csv', 'w+') as t:
        t.write(f'Request ID, Finished, Arrival Time, Start Time, Start Generate, [API Times], End Time, End Length, Strategy, w_p, w_d, w_s, w_p_real, w_d_real, w_s_real, Original Key, E2E Latency, TTFT, TPOT, Waiting Latency, Time before Processing, Last Sort Waste, API Exec Time, Predicted API Exec Time, API Invoke Interval, Predicted API Invoke Interval\n')
        for r in requests:

            r.end_length = r.end_length + len(r.api_times)

            start_time_calculated = -1
            if int(r.request_id) in times_before_processing_dict:
                start_time_calculated = r.arrival_time + times_before_processing_dict[int(r.request_id)]

            e2e_latency = -1
            if int(r.request_id) in E2E_latencies_dict:
                e2e_latency = E2E_latencies_dict[int(r.request_id)]

            ttft = -1
            if int(r.request_id) in ttfts_dict:
                ttft = ttfts_dict[int(r.request_id)]

            tpot = -1
            if int(r.request_id) in tpots_dict:
                tpot = tpots_dict[int(r.request_id)]

            waiting_time = -1
            if int(r.request_id) in waiting_times_dict:
                waiting_time = waiting_times_dict[int(r.request_id)]

            time_before_processing = -1
            if int(r.request_id) in times_before_processing_dict:
                time_before_processing = times_before_processing_dict[int(r.request_id)]

            strategy_used = '/'
            if int(r.request_id) in strategy_used_by_request_dict:
                strategy_used_list = strategy_used_by_request_dict[int(r.request_id)]
                strategy_used = '-'.join(strategy_used_list)

            predicted_waste = [[], [], []]
            real_waste = [[], [], []]
            if int(r.request_id) in predicted_waste_by_request_dict:
                predicted_waste_list = predicted_waste_by_request_dict[int(r.request_id)] # preserve, recompute, swap
                for i, waste in enumerate(predicted_waste_list):
                    predicted_waste[0].append(str(waste[0]))
                    predicted_waste[1].append(str(waste[1]))
                    predicted_waste[2].append(str(waste[2]))
                    real_waste[0].append(str(waste[3]))
                    real_waste[1].append(str(waste[4]))
                    real_waste[2].append(str(waste[5]))
                predicted_waste[0] = '-'.join(predicted_waste[0])
                predicted_waste[1] = '-'.join(predicted_waste[1])
                predicted_waste[2] = '-'.join(predicted_waste[2])
                real_waste[0] = '-'.join(real_waste[0])
                real_waste[1] = '-'.join(real_waste[1])
                real_waste[2] = '-'.join(real_waste[2])

            newest_sort_waste = -1
            if int(r.request_id) in newest_sort_waste_dict:
                newest_sort_waste = newest_sort_waste_dict[int(r.request_id)]
            

            api_times_str = str(r.api_times)
            api_times_str = api_times_str.replace(',', '-')
            api_times_str = api_times_str.replace(' ', '')

            t.write(f'{r.request_id}, {r.finished}, {r.arrival_time}, {start_time_calculated}, {r.start_generate}, {api_times_str}, {r.end_time}, {r.end_length}, {strategy_used}, {predicted_waste[0]}, {predicted_waste[1]}, {predicted_waste[2]}, {real_waste[0]}, {real_waste[1]}, {real_waste[2]}, {new_2_old_key[int(r.request_id)]}, {e2e_latency}, {ttft}, {tpot}, {waiting_time}, {time_before_processing}, {newest_sort_waste}, {r.sampling_params.api_exec_time}, {r.sampling_params.predicted_api_exec_time}, {r.sampling_params.api_invoke_interval}, {r.sampling_params.predicted_api_invoke_interval}\n')

    p50_queue = np.percentile(queueing_latencies, 50)
    p90_queue = np.percentile(queueing_latencies, 90)
    p99_queue = np.percentile(queueing_latencies, 99)
    p50_tpot = np.percentile(tpots, 50)
    p90_tpot = np.percentile(tpots, 90)
    p99_tpot = np.percentile(tpots, 99)
    
    p50_lat = np.percentile(latencies, 50)
    p90_lat = np.percentile(latencies, 90)
    p99_lat = np.percentile(latencies, 99)

    metrics = [p50_queue, p90_queue, p99_queue, p50_tpot, p90_tpot, p99_tpot, p50_lat, p90_lat, p99_lat]
    
    with open(f'./{args.msg}/iter_history.log', 'w+') as w:
        print(f'engine iters: {len(engine.iter_times)}, scheduler iters: {len(engine.scheduler.iter_history)}')
        for (i_s, i_e), b in zip(engine.iter_times, engine.scheduler.iter_history):
            if i_s is not None:
                b.iter_time = i_s.elapsed_time(i_e)
            w.write(f'{b}\n')

    with open(f'./{args.msg}/metric.log', 'w') as logfile:
        logfile.write(
            f'============ Serving Benchmark Result ============\n'
            
            f'Successful requests: {num_reqs}\n'
            f'Benchmark duration (s): {elapsed_time:.2f}\n'
            f'Total input tokens: {total_input_tokens}\n'
            f'Total generated tokens: {total_generated_tokens}\n'
            f'Request throughput (req/s): {num_reqs / elapsed_time:.4f}\n'
            f'Input token throughput (tok/s): {total_input_tokens / elapsed_time:.2f}\n'
            f'Output token throughput (tok/s): {total_generated_tokens / elapsed_time:.2f}\n'

            f'---------------Time before Processing-------------\n'
            f'Mean time before processing (ms): {np.mean(times_before_processing) * 1000 :.2f}\n'
            f'Median time before processing (ms): {np.median(times_before_processing) * 1000 :.2f}\n'
            f'P99 time before processing (ms): {np.percentile(times_before_processing, 99) * 1000 :.2f}\n'

            f'---------------Time to First Token----------------\n'
            f'Mean TTFT (ms): {np.mean(ttfts) * 1000 :.2f}\n'
            f'Median TTFT (ms): {np.median(ttfts) * 1000 :.2f}\n'
            f'P99 TTFT (ms): {np.percentile(ttfts, 99) * 1000 :.2f}\n'

            f'-----Time per Output Token (excl. 1st token)------\n'
            f'Mean TPOT (ms): {np.mean(tpots) * 1000 :.2f}\n'
            f'Median TPOT (ms): {np.median(tpots) * 1000 :.2f}\n'
            f'P99 TPOT (ms): {p99_tpot * 1000 :.2f}\n'
            
            f'----------------End-to-End Latency----------------\n'
            f'Mean E2E latencies (ms): {np.mean(E2E_latencies) * 1000 :.2f}\n'
            f'Median E2E latencies (ms): {np.median(E2E_latencies) * 1000 :.2f}\n'
            f'P99 E2E latencies (ms): {np.percentile(E2E_latencies, 99) * 1000 :.2f}\n'
            f'----------------Waiting Time----------------\n'
            f'Mean waiting times (ms): {np.mean(waiting_times) * 1000 :.2f}\n'
            f'Median waiting times (ms): {np.median(waiting_times) * 1000 :.2f}\n'
            f'P99 waiting times (ms): {np.percentile(waiting_times, 99) * 1000 :.2f}\n'

            f'==================================================\n'
        )

    return elapsed_time, engine.scheduler.total_tks, actual_tokens, cpu_full, num_reqs, np.mean(latencies), metrics


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)


    elapsed_time, total_tks, actual_tks, cpu_full, num_reqs, mean_normalized_lat, metrics = run_vllm(
        args,
    )
    
    torch.cuda.synchronize()
    with open(f'./{args.msg}/results.log', "a+") as logfile: 
        logfile.write(
            f"{args.api_policy},{args.distro},{args.qps},{mean_normalized_lat:.4f}\n"
            f"\t{metrics}\n"
        )
    with open(f'./{args.msg}/dump.log', "a+") as logfile: 
        logfile.write(
            f"###### RUN ########\n"
            f'actual tokens: {actual_tks}\n'
            f'total tokens: {total_tks}\n'
            f'distribution: {args.distro}\n'
            f'args: {args}\n'
            f'time: {elapsed_time:.2f} s\n'
            f"{actual_tks / elapsed_time:.2f} tokens/s\n"
            f"finished: {num_reqs}\n"
            f"Was CPU over 98% utilized: {cpu_full}\n"
        )

    if CACHE_EVENTS:
        times = []
        with open(f'./{args.msg}/swap_wait.csv', 'w+') as f:
            for s, e, t, i, o in CACHE_EVENTS:
                times.append(s.elapsed_time(e))
                f.write(f'{times[-1]}, {t}, {i}, {o}\n')
        print(f'cache events time mean: {np.mean(times):<10} max: {np.max(times):<10} min: {np.min(times):<10} total:  {np.sum(times):<10}')
        with open(f'./{args.msg}/dump.log', "a+") as logfile: 
            logfile.write(
                f'cache events time mean: {np.mean(times):<10} max: {np.max(times):<10} min: {np.min(times):<10} total:  {np.sum(times):<10}\n'
                f"###### RUN ########\n"
        )
        
    print(f"{args.api_policy},{args.distro},{args.qps},{num_reqs / elapsed_time:.4f} q/s, \n{metrics}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument(
        "--input-len", type=int, default=1024
    )
    parser.add_argument(
        "--output-len", type=int, default=1024
    )
    parser.add_argument(
        "--num-prompts", type=int, default=200000, help="Number of prompts to process."
    )
    parser.add_argument(
        "--api-ret-len", type=int, default=48, help="API return length."
    )
    parser.add_argument(
        "--api-max-calls", type=int, default=16, help="API max calls. -1 means no limit."
    )
    parser.add_argument(
        "--api-inv-offset", type=int, default=16, help="API invocation offset."
    )
    parser.add_argument(
        "--api-inv-mod", type=int, default=1, help="API invocation offset."
    )
    parser.add_argument(
        "--qps", type=float, default=1, help="Request arrival rate."
    )
    parser.add_argument(
        "--distro",
        type=str,
        choices=['R', 'L', 'N', 'U', 'B'],
        help='R=short, L=long, N=normal, U=uniform, B=bimodal',
    )
    parser.add_argument(
        "--msg",
        type=str,
        default='final_tput'
    )
    parser.add_argument(
        "--window",
        type=int,
        default=1800,
    )
    parser.add_argument(
        "--exp-json", type=str, required=True,
    )
    parser.add_argument(
        "--no-api", action='store_true',
        help='Run non-api bench, this strictly follows the order of input json'
    )
    parser.add_argument(
        "--percentage-error-api-exec-time", type=float, default=0,
    )
    parser.add_argument(
        "--percentage-error-request-length-before-api", type=float, default=0,
    )

    parser = EngineArgs.add_cli_args(parser)
    
    args = parser.parse_args()

    Path(f'./{args.msg}').mkdir(parents=True, exist_ok=True)
    main(args)
    
    
