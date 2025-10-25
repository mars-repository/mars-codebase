import argparse
import dataclasses
from dataclasses import dataclass
from typing import Optional, Tuple

from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig)

def str_to_bool(value):
    if value.lower() in ['true', 't', 'yes', 'y', '1']:
        return True
    elif value.lower() in ['false', 'f', 'no', 'n', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected, got {value}.')


@dataclass
class EngineArgs:
    """Arguments for vLLM engine."""
    model: str
    tokenizer: Optional[str] = None
    tokenizer_mode: str = 'auto'
    trust_remote_code: bool = False
    download_dir: Optional[str] = None
    load_format: str = 'auto'
    dtype: str = 'auto'
    seed: int = 0
    max_model_len: Optional[int] = None
    worker_use_ray: bool = False
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    # block_size: int = 64
    swap_space: int = 4  # GiB
    gpu_memory_utilization: float = 0.9
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: int = 256
    disable_log_stats: bool = False
    revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    quantization: Optional[str] = None
    chunk_fill: bool = False
    chunk_size: int = 32
    resize_model: bool = False
    n_layer: int = 12
    n_embed: int = 4096
    n_head: int = 16
    api_policy: str = 'P'
    heuristic_coef: Optional[float] = None
    discard_policy: str = 'inter'
    swap_limit_const: float = 1.0
    policy_config: str = ''
    starvation_avoidance: bool = False
    starvation_threshold: int = 0
    starvation_quantum: int = 0
    skip_sorting_for_this_number_of_iterations: int = 0

    def __post_init__(self):
        if self.tokenizer is None:
            self.tokenizer = self.model

    @staticmethod
    def add_cli_args(
            parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Shared CLI arguments for vLLM engine."""
        # Model arguments
        parser.add_argument(
            '--model',
            type=str,
            # default='facebook/opt-125m',
            default='EleutherAI/gpt-j-6b',
            # default = 'meta-llama/Llama-2-7b',
            help='name or path of the huggingface model to use')
        parser.add_argument(
            '--tokenizer',
            type=str,
            default=EngineArgs.tokenizer,
            help='name or path of the huggingface tokenizer to use')
        parser.add_argument(
            '--revision',
            type=str,
            default=None,
            help='the specific model version to use. It can be a branch '
            'name, a tag name, or a commit id. If unspecified, will use '
            'the default version.')
        parser.add_argument(
            '--tokenizer-revision',
            type=str,
            default=None,
            help='the specific tokenizer version to use. It can be a branch '
            'name, a tag name, or a commit id. If unspecified, will use '
            'the default version.')
        parser.add_argument('--tokenizer-mode',
                            type=str,
                            default=EngineArgs.tokenizer_mode,
                            choices=['auto', 'slow'],
                            help='tokenizer mode. "auto" will use the fast '
                            'tokenizer if available, and "slow" will '
                            'always use the slow tokenizer.')
        parser.add_argument('--trust-remote-code',
                            action='store_true',
                            help='trust remote code from huggingface')
        parser.add_argument('--download-dir',
                            type=str,
                            default=EngineArgs.download_dir,
                            help='directory to download and load the weights, '
                            'default to the default cache dir of '
                            'huggingface')
        parser.add_argument(
            '--load-format',
            type=str,
            default=EngineArgs.load_format,
            choices=['auto', 'pt', 'safetensors', 'npcache', 'dummy'],
            help='The format of the model weights to load. '
            '"auto" will try to load the weights in the safetensors format '
            'and fall back to the pytorch bin format if safetensors format '
            'is not available. '
            '"pt" will load the weights in the pytorch bin format. '
            '"safetensors" will load the weights in the safetensors format. '
            '"npcache" will load the weights in pytorch format and store '
            'a numpy cache to speed up the loading. '
            '"dummy" will initialize the weights with random values, '
            'which is mainly for profiling.')
        parser.add_argument(
            '--dtype',
            type=str,
            default=EngineArgs.dtype,
            choices=[
                'auto', 'half', 'float16', 'bfloat16', 'float', 'float32'
            ],
            help='data type for model weights and activations. '
            'The "auto" option will use FP16 precision '
            'for FP32 and FP16 models, and BF16 precision '
            'for BF16 models.')
        parser.add_argument('--max-model-len',
                            type=int,
                            default=None,
                            help='model context length. If unspecified, '
                            'will be automatically derived from the model.')
        # Parallel arguments
        parser.add_argument('--worker-use-ray',
                            action='store_true',
                            help='use Ray for distributed serving, will be '
                            'automatically set when using more than 1 GPU')
        parser.add_argument('--pipeline-parallel-size',
                            '-pp',
                            type=int,
                            default=EngineArgs.pipeline_parallel_size,
                            help='number of pipeline stages')
        parser.add_argument('--tensor-parallel-size',
                            '-tp',
                            type=int,
                            default=EngineArgs.tensor_parallel_size,
                            help='number of tensor parallel replicas')
        # NOTE(): deprecated, use deepspeed kernel to decide the block size
        # KV cache arguments
        # parser.add_argument('--block-size',
        #                     type=int,
        #                     default=EngineArgs.block_size,
        #                     choices=[8, 16, 32, 64],
        #                     help='token block size')
        # TODO(): Support fine-grained seeds (e.g., seed per request).
        parser.add_argument('--seed',
                            type=int,
                            default=EngineArgs.seed,
                            help='random seed')
        parser.add_argument('--swap-space',
                            type=int,
                            default=EngineArgs.swap_space,
                            help='CPU swap space size (GiB) per GPU')
        parser.add_argument('--gpu-memory-utilization',
                            type=float,
                            default=EngineArgs.gpu_memory_utilization,
                            help='the percentage of GPU memory to be used for'
                            'the model executor')
        parser.add_argument('--max-num-batched-tokens',
                            type=int,
                            default=EngineArgs.max_num_batched_tokens,
                            help='maximum number of batched tokens per '
                            'iteration')
        parser.add_argument('--max-num-seqs',
                            type=int,
                            default=EngineArgs.max_num_seqs,
                            help='maximum number of sequences per iteration')
        parser.add_argument('--disable-log-stats',
                            action='store_true',
                            help='disable logging statistics')
        # Quantization settings.
        parser.add_argument('--quantization',
                            '-q',
                            type=str,
                            choices=['awq', None],
                            default=None,
                            help='Method used to quantize the weights')
        parser.add_argument('--chunk-fill',
                            action='store_true')
        parser.add_argument('--chunk-size',
                            type=int,
                            default=EngineArgs.chunk_size)
        parser.add_argument('--api-policy',
                            type=str,
                            choices=['D', 'P', 'S', 'H', 'V', 'T', 'W', 'G', 'H-S', 'H-D', 'H-B', 'I'],
                            help='D=Discard, P=Preserve, S=Swap+passive, H=Heuristic, V=Vulcan, T=Test, W=Waste, G=Greedy, I=InferCept',
                            default=EngineArgs.api_policy)
        parser.add_argument('--heuristic-coef',
                            type=float,
                            default=None)
        parser.add_argument('--discard-policy',
                            type=str,
                            choices=['inter', 'intra'],
                            help='inter=inter-request, intra=intra-request',
                            default=EngineArgs.discard_policy)
        parser.add_argument('--swap-limit-const',
                            type=float,
                            default=1.0)
        parser.add_argument('--resize-model', action='store_true')
        parser.add_argument('--n-layer', type=int, default=12)
        parser.add_argument('--n-embed', type=int, default=4096)
        parser.add_argument('--n-head', type=int, default=16)
        # newly added
        parser.add_argument('--policy-config',
                            type=str,
                            help='See available policies in policy.py')
        parser.add_argument('--starvation-avoidance', type=str_to_bool, help='Avoid starvation in the scheduler, only for V', default="False")
        parser.add_argument('--starvation-threshold', type=int, help='Threshold for starvation avoidance, only for V', default=0)
        parser.add_argument('--starvation-quantum', type=int, help='Quantum for starvation avoidance, only for V', default=0)
        parser.add_argument('--skip-sorting-for-this-number-of-iterations', type=int, help='Skip sorting for this number of iterations', default=0)
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> 'EngineArgs':
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        return engine_args

    def create_engine_configs(
        self,
    ) -> Tuple[ModelConfig, CacheConfig, ParallelConfig, SchedulerConfig]:
        model_config = ModelConfig(self.model, self.tokenizer,
                                   self.tokenizer_mode, self.trust_remote_code,
                                   self.download_dir, self.load_format,
                                   self.dtype, self.seed, self.revision,
                                   self.tokenizer_revision, self.max_model_len,
                                   self.quantization,
                                   self.resize_model,
                                   self.n_layer,
                                   self.n_embed,
                                   self.n_head
                                   )
        cache_config = CacheConfig(
            -1, self.gpu_memory_utilization, self.swap_space,
            getattr(model_config.hf_config, 'sliding_window', None))
        parallel_config = ParallelConfig(self.pipeline_parallel_size,
                                         self.tensor_parallel_size,
                                         self.worker_use_ray)
        scheduler_config = SchedulerConfig(self.max_num_batched_tokens,
                                           self.max_num_seqs,
                                           model_config.max_model_len,
                                           self.chunk_fill,
                                           self.chunk_size,
                                           self.api_policy,
                                           self.heuristic_coef,
                                           self.discard_policy,
                                           self.swap_limit_const,
                                           self.policy_config,
                                           self.starvation_avoidance,
                                           self.starvation_threshold,
                                           self.starvation_quantum,
                                           self.skip_sorting_for_this_number_of_iterations
                                           )
        return model_config, cache_config, parallel_config, scheduler_config


@dataclass
class AsyncEngineArgs(EngineArgs):
    """Arguments for asynchronous vLLM engine."""
    engine_use_ray: bool = False
    disable_log_requests: bool = False
    max_log_len: Optional[int] = None

    @staticmethod
    def add_cli_args(
            parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = EngineArgs.add_cli_args(parser)
        parser.add_argument('--engine-use-ray',
                            action='store_true',
                            help='use Ray to start the LLM engine in a '
                            'separate process as the server process.')
        parser.add_argument('--disable-log-requests',
                            action='store_true',
                            help='disable logging requests')
        parser.add_argument('--max-log-len',
                            type=int,
                            default=None,
                            help='max number of prompt characters or prompt '
                            'ID numbers being printed in log. '
                            'Default: unlimited.')
        return parser
