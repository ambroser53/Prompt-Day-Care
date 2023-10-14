<p align="center" width="100%">
<img src="assets/lpb.jpg" alt="Llama Prompt Breeder" style="width: 200px; height:200px; display: block; margin: auto; border-radius: 50%;">
</p>

# LLaMA Day Care

A repository recreating the PromptBreeder Evolutionary Algorithm from the [DeepMind Paper](https://arxiv.org/abs/2309.16797?fbclid=IwAR1o-VI0DSwNOawBAQAcv0adoDakSWrgwPuLxWqJhLdCbouuZBA0Gm7Sy8I) in Python using LMQL as the backend.

The idea behind a PromptBreeder is that specific prompting techniques can cause massive improvements for LLMs on specific tasks. PromptBreeder uses a Binary Tournament Genetic Algorithm along with the adaptation capabilities of LLM to overcome the limitations of human-designed prompts. It creates generations of algorithms that optimise the prompts for a specific task given a specific fitness score metric to evaluate against.

LMQL stands for Language Model Query Language and allows for much more finegrained control over LLM inference. This allows for the creation of a much more powerful PromptBreeder, since parts of the prompts can be generated individually and output aspects such as chain of thought reasoning can be separated from constrained task specific decision output. In addition, it provides a platform for a more modularised approach to LLM based development.

## How to use

TODO include conda and venv environments

For all arguments, the default is what was used for the specific paper. All task-specific requirements, prompt structures and fitness scorers must be tailor-made to fit your specific task. An example task of answer quality is available in the example day_care folder.

| Argument                   | Description                                                                                                                                                                                                                                                                                                       |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--model_name_or_path`	      | Model to use (See [LQML's Model Documentation](https://lmql.ai/docs/models/))                                                                                                                                                                                                                                     |
| `--day_care_dir`	            | Directory of breeding ground (See [Day Care Structure](#day-care-structure))                                                                                                                                                                                                                                      |
| `--task`	                    | Task to put into the day care for breeding (required)                                                                                                                                                                                                                                                             |
| `--decoder_options_override` | 	Decoder options string to use instead of task description definition. Task specific options can be set in `task_description.json` file in the task's `day_care` directory and can be set separately for mutation and fitness testing. (See [LMQL Decoding Options](https://lmql.ai/docs/language/decoding.html)) |
|`--fitness_test_suite` | 	Fitness test suite to use. Each fitness test sample must have all relevant fields for the task's fitness test prompt as well as gold labels under the key headings `"gold_"+task_name`.                                                                                                                          |
| `--is_async`	| Whether to use async, For this to take effect `client_batch_size` must also `>1`.                                                                                                                                                                                                                                 |
| `--client_batch_size` | 	Batch size for client-side async querying (LMQL async querying)                                                                                                                                                                                                                                                  |
| `--server_batch_size`	| Batch size for use in on the inference with the model (LMQL async querying)                                                                                                                                                                                                                                       |
| `--population_size` | 	Number of units in the population                                                                                                                                                                                                                                                                                |
| `--initial_prompts_per_unit` | 	Number of prompts to generate for each unit in the initial population                                                                                                                                                                                                                                            |
| `--evolution_generations` | 	Number of generations to evolve the population (i.e. carry out binary tournament genetic algorithm on every pair in the population)                                                                                                                                                                              |
| `--num_eda_samples` | 	Number of samples to use for Estimation of Distribution (EDA) based mutations                                                                                                                                                                                                                                    |
| `--num_lamarckian_reasonings` | 	Number of reasonings to use for Lamarckian mutation operations                                                                                                                                                                                                                                                   |
| `--chunk_size` | 	Number of tokens that are generated speculatively, in one LLM call                                                                                                                                                                                                                                               |
| `--test_CoT` | 	Whether to test chain of thought prompt variants during fitness testing.                                                                                                                                                                                                                                         |
| `--test_noCoT` | 	Whether to test non-Chain of Thought prompt variants during fitness testing.                                                                                                                                                                                                                                     |
| `--verbose` | 	Whether to use verbose mode                                                                                                                                                                                                                                                                                      |
| `--seed` | 	Random seed                                                                                                                                                                                                                                                                                                      |
|`--continue_run_number` | 	Continue from specified run number                                                                                                                                                                                                                                                                               |
| `--debug_mode` | 	Whether to use debug mode. Debug mode uses a specific directory and cycles through all the mutation operations sequentially.                                                                                                                                                                                     |
| `--use_wandb` |	Whether to use wandb |
| `--wandb_project` |	Wandb project name |

## Day Care Structure
Day Care folders must contain the following files:
- `breeder.lmql` - The LMQL file that contains the breeding operations in LMQL format for each of the supported mutation operators (currently the eight from the original PromptBreeder paper) as well as the fitness test. Arguments unique to the queries must be specified here by type.
- `mutation_prompts.json` - The JSON file that contains the list of prompts used for mutation operations. Currently, this uses those specified in the original paper.
- `thinking_styles.json` - The JSON file containing thinking styles used for zero-order mutation operations. Currently, this uses those specified in the original paper.

Day Care folder must also contain a directory for each task, the name of the directory corresponds to the name of the task as it appears in the fitness test suite. Each task directory must contain a `task_description.json` with the following fields:
- `task_description` - a string fully describing the task 
- `fitness_scorer` - a dictionary with the `type` field that corresponds to the fitness scorer type defined within utils.fitness_scorers.py and any other arguments that the fitness scorer requires as additional fields.
- `decoder_options` - string in [LMQL decoder options format](https://lmql.ai/docs/language/decoding.html) that specifies the decoder options to use for the task. This can be overridden by the `--decoder_options_override` argument.
- `fitness_decoder_options` - string in [LMQL decoder options format](https://lmql.ai/docs/language/decoding.html) that specifies the decoder options to use specifically for the fitness test. This can be overridden by the `--decoder_options_override` argument.
- `fixed_system_prompt` - string that specifies the fixed system prompt to use for fitness test inference instead if bred system prompts. Ensure that if this is set for a task that the breeder.lmql is not specified to generate new system prompts to save on inference.

An example `day_care` folder is provided.