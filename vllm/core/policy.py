from typing import List, Tuple

from vllm.sequence import SequenceGroup


class Policy:

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        raise NotImplementedError

    def sort_by_priority(
        self,
        now: float,
        seq_groups: List[SequenceGroup],
        running: bool = True,
        running_batch: int = 0,
        running_blocks: int = 0,
        use_cached_score: bool = False,
    ) -> List[SequenceGroup]:
        return sorted(
            seq_groups,
            key=lambda seq_group: self.get_priority(now, seq_group),
            reverse=True,
        )
    
# from vllm.core.block_manager import BlockSpaceManager, Device

class V2(Policy):

    def calculate_memory_time_blocks(self, num_blocks: int, running_batch: int, block_size:int) -> float:
        # taken from waste_discard function -- we care only about the time to compute the sequence and not the c_other part
        c_h = max(384 - running_batch, 1) # computational units for computing the sequence
        #n = max((block_size * num_blocks + c_h - 1) // c_h - 1, 1) #number of iterations to compute the sequence
        n = max((block_size * num_blocks + c_h - 1) // c_h, 1) #number of iterations to compute the sequence

        a = 0.1 # replace f_ch a
        c = 10 # replace f_ch c
        f_s = (a * 384 + c) / 1000 # forward pass time of one iteration
        
        memory_time = f_s * (1+n) * n / 2 * c_h # total time to compute the num_blocks
        return memory_time

    def get_priority_old(
        self,
        now: float,
        seq_group: SequenceGroup,
        # block_manager: BlockSpaceManager,
        running_batch: int,
        running_blocks: int,
        running: bool = True,
        blocksize: int = 64,
    ) -> float:
        if running:
            return now - seq_group.arrival_time
        cur_length = seq_group.get_seqs()[0].data.get_len()
        prompt_len = seq_group.get_seqs()[0].get_prompt_len()
        seq_blocks = (prompt_len + seq_group.sampling_params.predicted_api_invoke_interval + blocksize - 1) // blocksize
        api_start = seq_blocks * blocksize

        # quadratic
        a = 1e-4
        b = 1e-3
        before_exec_time = a * seq_group.sampling_params.predicted_api_invoke_interval * (seq_group.sampling_params.predicted_api_invoke_interval + prompt_len) + b * prompt_len ** 2
        before = (api_start * before_exec_time) / 2

        req_after = seq_group.sampling_params.predicted_api_invoke_interval + seq_group.sampling_params.api_return_length
        after_exec_time = seq_group.sampling_params.remain_length
        after = (api_start * after_exec_time) + ((req_after * after_exec_time) / 2) #TODO: recheck this

        api_exec_time = seq_group.sampling_params.predicted_api_exec_time
        api_call_complete = (seq_group.sampling_params.api_max_calls == 0)

        strategy = seq_group.sampling_params.strategy

        if strategy == 'preserve':
            # BUG to fix : api = #memory_of_before * api_exec_time
            # lets fix this together: before is the meomry of the before part over time, so if the before contains m tokens, then the memory of before is m * api_start/2
            api = api_start * api_exec_time
            return before + api + after
        elif strategy == 'recompute':
            integral_recompute_weight = 0.1
            api = api_exec_time * integral_recompute_weight

            if api_call_complete:
                return before + after
            else:
                return before + api + before + after
        elif strategy == 'swap':
            integral_swap_weight = 0.1
            api = api_exec_time * integral_swap_weight
            cpu_to_gpu_transfer_rate = 2 #TODO: fix this hyperparameter
            swap = api_start * (api_start / cpu_to_gpu_transfer_rate) / 2

            if api_call_complete:
                return swap + after
            else:
                return before + swap + api + swap + after
        raise ValueError()
        # return seq_group.get_seqs()[0].expected_out_len

    def get_priority( # ranking function
        self,
        now: float,
        seq_group: SequenceGroup,
        # block_manager: BlockSpaceManager,
        running_batch: int,
        running_blocks: int,
        running: bool = True,
        blocksize: int = 64,
        use_cached_score: bool = False
    ) -> float:
        if running:
            return now - seq_group.arrival_time
        if use_cached_score and seq_group.newest_sort_waste is not None:
            return -seq_group.newest_sort_waste
        # else:
        #     return -seq_group.sampling_params.waste
        before_seq_blocks = (seq_group.get_seqs()[0].get_prompt_len() + seq_group.sampling_params.predicted_api_invoke_interval + blocksize - 1) // blocksize

        before = self.calculate_memory_time_blocks(before_seq_blocks, running_batch, blocksize)

        # TODO: seq_group.sampling_params.api_invoke_interval should be #tokens in the after part
        after_seq_blocks = (seq_group.sampling_params.predicted_api_invoke_interval + seq_group.sampling_params.api_return_length + blocksize - 1) // blocksize
        after = self.calculate_memory_time_blocks(after_seq_blocks, running_batch, blocksize)

        api_exec_time = seq_group.sampling_params.predicted_api_exec_time
        api_call_complete = (seq_group.sampling_params.api_max_calls == 0)

        strategy = seq_group.sampling_params.strategy # first decide strategy, then sort

        if strategy == 'preserve':
            before_tokens = before_seq_blocks * blocksize
            api_memory = before_tokens * api_exec_time
            score = before + api_memory + after
        elif strategy == 'recompute':
            integral_recompute_weight = 0
            api = api_exec_time * integral_recompute_weight

            before_after_seq_blocks = before_seq_blocks + after_seq_blocks
            # TODO: make sure the claculation is correct when we have decode phases of the after part, its not like the before part
            before_after_memory_time = self.calculate_memory_time_blocks(before_after_seq_blocks, running_batch, blocksize)
            if api_call_complete:
                score = before_after_memory_time
            else:
                score = before + api + before_after_memory_time
        elif strategy == 'swap':
            integral_swap_weight = 0.1
            api = api_exec_time * integral_swap_weight
            cpu_to_gpu_transfer_rate = 2 #TODO: fix this hyperparameter
            swap = before_seq_blocks * (before_seq_blocks / cpu_to_gpu_transfer_rate) / 2

            if api_call_complete:
                score = swap + after
            else:
                score = before + swap + api + swap + after
        seq_group.newest_sort_waste = score
        return -score
        raise ValueError()

    def sort_by_priority(
        self,
        now: float,
        seq_groups: List[SequenceGroup],
        # block_manager: BlockSpaceManager,
        running: bool = True,
        running_batch: int = 0,
        running_blocks: int = 0,
        use_cached_score: bool = False,
        # blocksize: int = 64
    ) -> List[SequenceGroup]:
        # print([sg.inflight_length for sg in seq_groups])
        # running_batch = sum([sg.inflight_length for sg in seq_groups])
        # running_blocks = sum([sg.blocks for sg in seq_groups])
        return sorted(
            seq_groups,
            key=lambda seq_group: self.get_priority(now, seq_group, running_batch, running_blocks, running, use_cached_score=use_cached_score),
            reverse=True,
        )


class FCFS(Policy):

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        return now - seq_group.arrival_time
    
class Chunked_FCFS(Policy):

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> Tuple[int, float]:
        return -seq_group.get_seqs()[0].data.logical_query_len, now - seq_group.arrival_time

class LongestRemainingAPIFirst(Policy):

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        return seq_group.predicted_api_remaining_time(now)

class ShortestJobFirst(Policy):
    
        def get_priority(
            self,
            now: float,
            seq_group: SequenceGroup,
        ) -> float:
            return -seq_group.sampling_params.remain_langth


class PolicyFactory:

    _POLICY_REGISTRY = {
        'fcfs': FCFS,
        'c-fcfs': Chunked_FCFS,
        'lra': LongestRemainingAPIFirst,
        'sjf': ShortestJobFirst,
        'V2': V2
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        if policy_name not in cls._POLICY_REGISTRY:
            raise ValueError(f'Policy {policy_name} not found')
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
