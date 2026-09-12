import os
import numpy as np
import torch
import platform
import yaml

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sota_root = os.path.join(root_dir, "sota", "Jailbreak")
ROOT_DIR = os.getenv("LLMGE_ROOT_DIR", root_dir)
SLURM_CONFIG_DIR = os.getenv(
    "LLMGE_SLURM_CONFIG_DIR",
    os.path.join(ROOT_DIR, "slurm-config"),
)
CLUSTER = os.getenv("LLMGE_CLUSTER", "pace-ice")
DATA_PATH = os.path.join(ROOT_DIR, "sota/Jailbreak/behaviors")
SOTA_ROOT = os.getenv("LLMGE_SOTA_ROOT", sota_root)

# individual is a trait-segmented prompt-builder module (prompt.py)
SEED_NETWORK = os.getenv(
    "LLMGE_SEED_NETWORK",
    os.path.join(SOTA_ROOT, "prompt.py"),
)
MODEL = "prompt"

# mutator LLM used by server.py for mutation/crossover
MODEL_PATH = "/storage/ice-shared/vip-vvk/llm_storage/meta-llama/Llama-3.3-70B-Instruct/"
VARIANT_DIR = os.path.join(SOTA_ROOT, "models/llmge_models")
TRAIN_FILE = os.path.join(SOTA_ROOT, "eval.py")
LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "1648"))
ISLAND_TEMP_SCRIPT = os.path.join("src", "island_temp_script_{ISLAND_NUM}.sh")

SLURM_OUTPUT_PATH = "run_job_outputs/"

DEFAULT_PROMPT_GROUP = "Jailbreak/traits"
PROMPTS = f"templates/{DEFAULT_PROMPT_GROUP}/**/*.txt"
CONSTANT_RULES_PATH = "templates/Jailbreak/ConstantRules.txt"

OUTPUT_DIR = "jailbreak_test"
PORT = int(os.getenv("LLMGE_PORT", "8137"))

LLM_MODEL = 'llama3.3'
PACE_ICE = True

# available mutator LLM identifiers
LLM_QWEN = 'qwen25'
LLM_MIXTRAL = 'mixtral'
LLM_LLAMA3 = 'llama3'
LLM_GEMMA2 = 'gemma2'
LLM_GEMMA3 = 'gemma3'
LLM_DEEPSEEK = 'deepseek'
LLM_GEMINI = 'gemini'

# llms allowed for island runs
ISLAND_LLMS = [LLM_QWEN, LLM_MIXTRAL, LLM_DEEPSEEK, LLM_LLAMA3, LLM_GEMMA2, LLM_GEMMA3, LLM_GEMINI]

ENVIRONMENT_DIR = os.path.join(ROOT_DIR, ".venv")
LOCAL_LLM = os.getenv("LOCAL_LLM", "true").lower() in ("true", "1", "yes")
HOSTNAME_DIR = os.path.join(ROOT_DIR, "hostname.log")

# multi-island settings
GLOBAL_DATA_PATH = "global_data"
PROMPT_GROUP_TEMPLATE = "templates/{prompt_group}/**/*.txt"
MAX_ISLANDS = len(ISLAND_LLMS)

SLURM_MIXT_INPUT_X = SEED_NETWORK
SLURM_MIXT_INPUT_Y = os.path.join(SOTA_ROOT, "models/Menghao/model_x.py")
SLURM_MIXT_OUTPUT = os.path.join(SOTA_ROOT, "models/Menghao/model_z.py")
SLURM_MIXT_TOP_P = 0.15
SLURM_MIXT_TEMPERATURE = 0.1
SLURM_MIXT_APPLY_QUALITY_CONTROL = True
SLURM_MIXT_BIT = 8

ISLAND_CONTROLLER_RUN_NAME = "jailbreak_islands_run1"
ISLAND_CONTROLLER_NUM_ISLANDS = 2
ISLAND_CONTROLLER_LLMS = "llama3"
ISLAND_CONTROLLER_PROMPT_GROUPS = "Jailbreak/traits"

# Jailbreak target + judge
TARGET_MODEL_PATH = os.getenv(
    "JB_TARGET_MODEL_PATH",
    "/storage/ice-shared/vip-vvk/llm_storage/meta-llama/Llama-3.1-8B-Instruct/",
)
TARGET_SERVED_NAME = os.getenv("JB_TARGET_SERVED_NAME", "target")
TARGET_VLLM_PORT = int(os.getenv("JB_TARGET_VLLM_PORT", "8001"))

JUDGE_MODEL_PATH = os.getenv(
    "JB_JUDGE_MODEL_PATH",
    "/storage/ice-shared/vip-vvk/llm_storage/meta-llama/Llama-Guard-3-8B/",
)
JUDGE_SERVED_NAME = os.getenv("JB_JUDGE_SERVED_NAME", "guard")
JUDGE_VLLM_PORT = int(os.getenv("JB_JUDGE_VLLM_PORT", "8002"))

BEHAVIORS_PATH = os.getenv("JB_BEHAVIORS_PATH", os.path.join(SOTA_ROOT, "behaviors/curated.csv"))
N_BEHAVIORS_PER_EVAL = int(os.getenv("JB_N_BEHAVIORS_PER_EVAL", "8"))
BEHAVIOR_SEED = int(os.getenv("JB_BEHAVIOR_SEED", "0"))

JB_CACHE_DIR = os.getenv("JB_CACHE_DIR", os.path.join(os.path.expanduser("~"), "scratch", "jb_cache"))

# Load slurm templates from slurm_config.yaml
_slurm_config_path = os.path.join(SLURM_CONFIG_DIR, 'slurm_config.yaml')
if os.path.exists(_slurm_config_path):
    with open(_slurm_config_path, 'r') as _f:
        _slurm_config = yaml.safe_load(_f)
    LLM_GPU = _slurm_config.get('gpu_selection', 'H200|H100')
    PYTHON_BASH_SCRIPT_TEMPLATE = _slurm_config.get('python_bash_script', '')
    LLM_BASH_SCRIPT_TEMPLATE = _slurm_config.get('llm_bash_script', '')
    ISLANDS_BASH_SCRIPT_TEMPLATE = _slurm_config.get('islands_bash_script', '')
else:
    LLM_GPU = 'H200|H100'
    PYTHON_BASH_SCRIPT_TEMPLATE = ''
    LLM_BASH_SCRIPT_TEMPLATE = ''
    ISLANDS_BASH_SCRIPT_TEMPLATE = ''

INFERENCE_SUBMISSION = False

LOCAL = os.getenv("LOCAL", "false").lower() in ("true", "1", "yes")
if LOCAL:
    RUN_COMMAND = 'bash'
    DELAYED_CHECK = os.getenv("DELAYED_CHECK", "false").lower() in ("true", "1", "yes")
else:
    RUN_COMMAND = 'sbatch'
    DELAYED_CHECK = os.getenv("DELAYED_CHECK", "true").lower() in ("true", "1", "yes")
MACOS = platform.system() == "Darwin"
RUNLINE_AMP = ''
if torch.mps.is_available():
    DEVICE = 'mps'
    MACOS = True
    RUNLINE_AMP = "-amp"
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

# resolves to {MODEL}_{gene_id}
RUNLINE_TMP = '{}_{}'
EVAL_RUNLINE = "uv run python {} --model {} --variant_dir {VARIANT_DIR}"
EVAL_NO_PROGRESS_TIMEOUT_SECONDS = int(os.getenv("LLMGE_EVAL_NO_PROGRESS_TIMEOUT_SECONDS", str(40 * 60)))
"""
Evolution Constants/Params
"""
# obj1 = harm_success_rate to maximize, obj2 = avg prompt tokens / cost to minimize
FITNESS_WEIGHTS = (1.0, -1.0)
INVALID_FITNESS_MAX = tuple([float(x*np.inf*-1) for x in FITNESS_WEIGHTS])
PLACEHOLDER_FITNESS = tuple([int(x*9999999999*-1) for x in FITNESS_WEIGHTS])
NUM_EOT_ELITES = 10
GENERATION = 0
PROB_QC = 0.0
PROB_EOT = 0.25
num_generations = 3  # small for first validation
start_population_size = 16
population_size = 16
crossover_probability = 0.35
mutation_probability = 0.8
num_elites = 4
hof_size = 100
max_gen_attempts = 5
migration_gen = 5
"""
Misc. Non-sense
"""
DNA_TXT = """
LLM-Guided Evolution Jailbreak

⠀⠀⣿⡇⠀⠀⠀⠀⠀⠀⠀⢀⣠⣤⣶⣶⠶⣶⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⣀⣹⣟⣛⣛⣻⣿⣿⣿⡾⠟⢉⣴⠟⢁⣴⠋⣹⣷⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠈⠛⠛⣿⠉⢉⣩⠵⠚⠁⢀⡴⠛⠁⣠⠞⠁⣰⠏⠸⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⢻⣷⠋⠁⠀⢀⡴⠋⠀⢀⡴⠋⠀⣼⠃⠀⡼⢿⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⢻⣆⣠⡴⠋⠀⠀⣠⠟⠀⢀⡾⠁⠀⡼⠁⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠻⣯⡀⠀⢀⡼⠃⠀⢠⡟⠀⢀⡾⠁⢀⣾⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠙⠻⣶⣟⡀⠀⣰⠏⠀⢀⡾⠁⠀⣼⢹⣿⣀⣤⣤⣴⠶⢿⡿⠛⢛⣷⢶⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠻⠿⠶⠶⠾⠷⠶⠿⠛⢻⣟⠉⣥⠟⠁⣠⠟⠀⢠⠞⠁⣄⡿⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣿⠞⠁⢀⡴⠋⠀⣴⠋⠀⣰⠟⠀⣤⡾⣷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⡄⢠⠞⠁⢀⡾⠁⢀⡼⠃⢀⡴⠋⠀⢸⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣷⠋⠀⣰⠏⠀⣠⠟⠀⣰⠟⠁⢀⡴⠛⣿⠀⠀⣀⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠻⣧⡼⠃⢀⡼⠋⢠⡞⠁⣠⣞⣋⣤⣶⣿⡟⠛⣿⠛⠛⣻⠟⠷⢶⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠻⣦⣾⣤⣴⣯⡶⠾⠟⠛⠉⠉⠉⣿⡇⢠⡏⠀⣰⠏⠀⢀⣼⠋⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⡾⠀⢰⠏⠀⢠⡞⠁⠀⣠⠞⢻⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣷⠇⢠⠏⠀⣰⠋⠀⣠⠞⠁⠀⢀⣿⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡟⢠⠟⢀⡼⠁⣠⠞⠁⣀⣴⢾⣿⣤⣿⣦⣄⣀⡀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣿⡟⣠⠏⣠⠞⣁⣴⣾⣿⣿⣿⣿⣿⣿⡏⢹⡏⠛⠳⣦⣄⡀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⢷⣾⣷⠿⠿⠛⠉⠀⠀⠈⠳⣬⣿⡟⣾⠁⠀⣼⠃⠉⠻⠆
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢿⣧⡏⠀⣼⠃⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⠁⡼⠁⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣟⡼⠁⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡿⠁⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢙⣃⠀⠀
"""
