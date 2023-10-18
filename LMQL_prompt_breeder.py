import os
import pickle
import shutil
from glob import glob
import re

import time
from itertools import compress

import lmql
import json
import hjson
import random
import argparse
import hashlib

import nest_asyncio
from tqdm import tqdm
from utils import fitness_scorers
from utils.bert_filterer import BertFilterer

import torch
from transformers import BitsAndBytesConfig
import asyncio

# Logging and Debugging
import wandb
import logging
from utils.logging import CustomFormatter

custom_handler = logging.StreamHandler()
custom_handler.setFormatter(CustomFormatter())

file_handler = logging.FileHandler('generation_pipeline.log')
file_handler.setFormatter(CustomFormatter())

logging.root.handlers = []
logging.basicConfig(
    level=logging.WARNING,
    handlers=[
        file_handler,
        custom_handler
    ])


def format_specification(specification_str, model_name_or_path, decoder_options, batch_size):
    specification_str = specification_str.replace('{model_name_or_path}', '"' + model_name_or_path + '"')
    specification_str = specification_str.replace('{decoder_options}', decoder_options)
    specification_str = specification_str.replace('{batch_size}', str(batch_size))
    return specification_str


# sync up an async function, so it doesn't have to be waited for
def sync(fn, *args, **kwargs):
    return asyncio.get_event_loop().run_until_complete(fn(*args, **kwargs))


def get_query_args(query_arg_spec, inputs, **additional_kwargs):
    '''
    Creates a dictionary of arguments for a query from a specification and a dictionary of inputs.
    Ensures there are no missing inputs but will leave them as None if they can't be found (desirable if they are optional in LMQL code).

    :param query_arg_spec: list of arguments names for the query
    :param inputs: current inputs for the query
    :param additional_kwargs: optional additional arguments from elsewhere to introduce if they are present
    :return: new_args: dictionary of arguments for the query
    '''
    new_args = {}
    for arg in query_arg_spec:
        if arg == 'model_kwargs':
            continue

        if arg not in inputs.keys() and arg in additional_kwargs.keys():
            new_args[arg] = additional_kwargs[arg]
        elif arg not in inputs.keys():
            new_args[arg] = None
        else:
            new_args[arg] = inputs[arg]
    return new_args


def is_birth_defect(newborn):
    if newborn is None:
        return True
    for baby_key in newborn.keys():
        if isinstance(newborn[baby_key], str):
            if len(newborn[baby_key].strip()) < 5:
                return True
    return False


def clean_newborn(new_born):
    for baby_key in new_born.keys():
        if isinstance(new_born[baby_key], str):
            new_born[baby_key] = new_born[baby_key].strip()
    return new_born


def get_genome_by_id(population, genome_id):
    return next((g for g in population if g['id'] == genome_id), None)


def get_next_run_number(path: str) -> int:
    runs = glob(os.path.join(path, '**/RUN_[0-9]*'), recursive=True)
    dirs = [int(re.search(r'RUN_([0-9]+)$', run).group(1)) for run in runs if os.path.isdir(run)]
    return max(dirs) + 1 if dirs else 0


class DayCare:
    def __init__(
            self,
            day_care_dir,
            task,
            model_name_or_path,
            population_size,
            initial_prompts_per_unit,
            evolution_generations,
            num_eda_samples,
            num_lamarckian_reasonings,
            test_CoT,
            test_noCoT,
            seed,
            **kwargs
    ):
        # set RUN and output directory
        if kwargs.get('debug_mode', False):
            self.output_dir = day_care_dir + task + '/DEBUG/'
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            if os.path.isfile(self.output_dir + 'complete'):
                shutil.rmtree(self.output_dir)
                os.makedirs(self.output_dir)
            self.create_config(day_care_dir, task, model_name_or_path, population_size, initial_prompts_per_unit,
                               evolution_generations, num_eda_samples, num_lamarckian_reasonings, seed, kwargs)
        elif kwargs.get('continue_run_number', None) is not None:
            self.output_dir = day_care_dir + task + '/RUN_' + str(kwargs.get('continue_run_number')) + '/'
            if not os.path.exists(self.output_dir):
                raise FileNotFoundError('Run number specified for continue_run_number does not exist: ' + str(
                    kwargs.get('continue_run_number')))

            # load config
            with open(self.output_dir + '_config.json') as f:
                config = json.load(f)

            # unpack config
            day_care_dir = config.pop('day_care_dir')
            task = config.pop('task')
            model_name_or_path = config.pop('model_name_or_path')
            population_size = config.pop('population_size')
            initial_prompts_per_unit = config.pop('initial_prompts_per_unit')
            evolution_generations = config.pop('evolution_generations')
            num_eda_samples = config.pop('num_eda_samples')
            num_lamarckian_reasonings = config.pop('num_lamarckian_reasonings')
            seed = config.pop('seed')
            kwargs = config
        else:
            next_run_num = get_next_run_number(os.path.join(day_care_dir, task))
            self.output_dir = day_care_dir + task + '/RUN_' + str(next_run_num) + '/'
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            self.create_config(day_care_dir, task, model_name_or_path, population_size, initial_prompts_per_unit,
                               evolution_generations, num_eda_samples, num_lamarckian_reasonings, seed, kwargs)

        self.hasher = hashlib.sha3_224()
        self.model_kwargs = {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'  # {'fp4', 'nf4'}
            )
        }

        if not test_CoT and not test_noCoT:
            raise ValueError('At least one of test_CoT and test_noCoT must be True')
        self.test_CoT = test_CoT
        self.test_noCoT = test_noCoT

        if kwargs.get('is_async', False) and kwargs.get('client_batch_size', 1) > 1:
            self.is_async = True
        else:
            self.is_async = False
        self.task = task
        self.chunk_size = kwargs.get('chunk_size', 3)
        self.breeder = self._get_breeder(day_care_dir, model_name_or_path, **kwargs)
        self.num_eda_samples = num_eda_samples
        self.num_lamarckian_reasonings = num_lamarckian_reasonings
        self.evolution_generations = evolution_generations

        if not os.path.exists(day_care_dir + '/' + self.task + '/task_description.json'):
            raise FileNotFoundError(
                'task_description.json not found in day care directory of specified task: ' + self.task)
        else:
            with open(day_care_dir + '/' + self.task + '/task_description.json') as f:
                self.task_description = json.load(f)

        # load mutation prompts and thinking styles
        with open(day_care_dir + '/mutation_prompts.json') as f:
            self.mutation_prompts = json.load(f)
        with open(day_care_dir + '/thinking_styles.json') as f:
            self.thinking_styles = json.load(f)

        # load fitness scorer
        # - new fitness scorers can be added to utils.fitness_scorers
        # - the fitness scorer must be specified in the task_description.json of the task being bred
        fitness_scorer_type = self.task_description['fitness_scorer'].pop("type")
        self.fitness_scorer = eval("fitness_scorers." + fitness_scorer_type + "FitnessScorer")(self.task,
                                                                                               **self.task_description[
                                                                                                   'fitness_scorer'])

        # INITIALISE POPULATION
        if not os.path.exists(self.output_dir + 'generation_0_log.json'):
            if self.is_async:
                asyncio.run(self._initialise_population(population_size, initial_prompts_per_unit, kwargs))
            else:
                sync(self._initialise_population(population_size, initial_prompts_per_unit, kwargs))

        # GENOME RECORDS
        # a record of all previous generations is kept
        self.generations = {}

        # a record of correct reasonings given a task prompt is kept
        self.correct_reasonings = {}

        # a tally of the best genome in each generation is kept
        self.elite_history = {}

        # all new mutation prompts that pass the bar are kept
        self.new_mutation_prompts = []

        # set seed
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # get the bert filterer
        self.bert_filterer = BertFilterer('bert-base-uncased')

    def create_config(self, day_care_dir, task, model_name_or_path, population_size, initial_prompts_per_unit,
                      evolution_generations, num_eda_samples, num_lamarckian_reasonings, seed, kwargs):
        config_complete = dict({
            'day_care_dir': day_care_dir,
            'task': task,
            'model_name_or_path': model_name_or_path,
            'population_size': population_size,
            'initial_prompts_per_unit': initial_prompts_per_unit,
            'evolution_generations': evolution_generations,
            'num_eda_samples': num_eda_samples,
            'num_lamarckian_reasonings': num_lamarckian_reasonings,
            'seed': seed
        }, **kwargs)
        # save config
        with open(self.output_dir + '_config.json', 'w') as f:
            json.dump(config_complete, f, indent=4)

    async def _initialise_population(self, population_size, initial_prompts_per_unit, kwargs):
        self.population = []
        # time the generation of each prompt and the initial population as a whole
        timer = time.time()
        start = time.time()

        async_tasks = []
        self.p_par = tqdm(range(population_size * initial_prompts_per_unit))

        if self.is_async:
            self.sem = asyncio.Semaphore(kwargs.get('client_batch_size'))

        self.genome_id = 0

        for i in range(population_size):

            for j in range(initial_prompts_per_unit):
                if self.is_async:
                    async_tasks.append(self._async(self.mutate, 'mutation_zero_order_direct', regen_defects=False))
                else:
                    genome = sync(self.mutate('mutation_zero_order_direct', regen_defects=True))
                    self.genome_to_dir(genome, 0)
                    self.population.append(genome)

                    gen_time = time.time() - timer
                    timer = time.time()
                    if kwargs.get('verbose', False):
                        print(genome)
                        print('time to generate: ', gen_time)
                    if wandb.run is not None:
                        wandb.log({
                            'time_to_generate': gen_time,
                        })

        if self.is_async:
            self.population = await asyncio.gather(*async_tasks)

        if kwargs.get('verbose', False):
            print('mean time to generate: ', (time.time() - start) / (population_size * initial_prompts_per_unit))
            print('total time to generate: ', time.time() - start)
        if wandb.run is not None:
            wandb.log({
                'mean_time_to_generate': (time.time() - start) / (population_size * initial_prompts_per_unit),
                'total_time_to_generate': time.time() - start,
            })

    def genome_to_dir(self, genome, gen):
        if not os.path.exists(self.output_dir + 'generation_' + str(gen) + '_population/'):
            os.makedirs(self.output_dir + 'generation_' + str(gen) + '_population/')
        unique_id = genome['id']
        with open(self.output_dir + 'generation_' + str(gen) + '_population/' + unique_id + '.json', 'w') as f:
            json.dump(genome, f)

    def from_dir_genome(self, genome, gen):
        unique_id = genome['id']
        if os.path.exists(self.output_dir + 'generation_' + str(gen) + '_population/' + unique_id + '.json'):
            with open(self.output_dir + 'generation_' + str(gen) + '_population/' + unique_id + '.json') as f:
                return json.load(f)
        else:
            return genome

    def get_genome_id(self, genome):
        self.hasher.update(pickle.dumps(self.str_format_genome(genome)))
        unique_id = self.hasher.hexdigest()
        return unique_id

    async def breed(self, fitness_suite, **kwargs):
        mutation_operators = [
            'mutation_zero_order_direct',
            'mutation_first_order_direct',
            'mutation_eda_random',
            'mutation_eda_ranked',
            'mutation_eda_lineage',
            'hypermutation_zero_order_direct',
            'hypermutation_first_order_direct',
        ]
        if self.test_CoT:
            mutation_operators.append('mutation_lamarckian')

        if kwargs.get('debug_mode', False):
            # in debug mode go through each mutation sequentially
            if os.path.exists(self.output_dir + 'current_operator.json'):
                with open(self.output_dir + 'current_operator.json') as f:
                    mutation_i = json.load(f)['mutation_i']
            else:
                mutation_i = 0

        if self.is_async:
            self.sem = asyncio.Semaphore(kwargs.get('client_batch_size'))

        # time generations
        start = time.time()

        # EVOLVE POPULATION
        for gen in tqdm(range(self.evolution_generations)):
            if os.path.exists(self.output_dir + 'generation_' + str(gen) + '_log.json') and os.path.exists(
                    self.output_dir + 'generation_' + str(gen) + '_stats.json'):
                self.load_generation(gen)
                continue
            elif os.path.exists(self.output_dir + 'generation_' + str(gen) + '_log.json'):
                # load preliminary generation log
                with open(self.output_dir + 'generation_' + str(gen) + '_log.json') as f:
                    self.population = json.load(f)
                # update self.population with new fitness scores
                for pop_i in range(len(self.population)):
                    if 'fitness' not in self.population[pop_i]:
                        self.population[pop_i] = self.from_dir_genome(self.population[pop_i], gen)
            else:
                # save preliminary generation log to run folder
                with open(self.output_dir + 'generation_' + str(gen) + '_log.json', 'w') as f:
                    json.dump(self.population, f)

            await self.update_fitnesses(fitness_suite, gen)

            async_mutation_tasks = []
            next_population = []

            # put population into random pairs
            random.shuffle(self.population)
            pairs = [(self.population[i], self.population[i + 1]) for i in range(0, len(self.population), 2)]
            self.p_par = tqdm(range(len(pairs)))

            for genome1, genome2 in pairs:
                if kwargs.get('debug_mode', False):
                    mutation_op = mutation_operators[mutation_i % len(mutation_operators)]
                    mutation_i += 1
                else:
                    # get a random mutation operator
                    mutation_op = random.choice(mutation_operators)

                if self.is_async:
                    # async
                    bte_task = self._async(self.binary_tournament_evolve, genome1, genome2, fitness_suite, mutation_op,
                                           gen)
                    async_mutation_tasks.append(bte_task)
                else:
                    elite_genome, mutant = sync(
                        self.binary_tournament_evolve(genome1, genome2, fitness_suite, mutation_op, gen))

                    # update population
                    next_population.append(mutant)
                    next_population.append(elite_genome)

                if kwargs.get('debug_mode', False):
                    with open(self.output_dir + 'current_operator.json', 'w') as f:
                        json.dump({'mutation_i': mutation_i}, f)

            if self.is_async:
                # gather all tasks with semaphore batch limit
                generation_results = await asyncio.gather(*async_mutation_tasks)
                for elite_genome, mutant in generation_results:
                    next_population.append(mutant)
                    next_population.append(elite_genome)

            # get stats on last generation
            population_fitness = [g['fitness']['best'] for g in self.population if 'fitness' in g.keys()]
            population_CoT_fitness = [g['fitness']['CoT'] for g in self.population if 'fitness' in g.keys()]
            population_noCoT_fitness = [g['fitness']['noCoT'] for g in self.population if 'fitness' in g.keys()]

            gen_stats = {
                'mean_fitness': sum(population_fitness) / len(population_fitness),
                'max_fitness': max(population_fitness),
                'mean_CoT_fitness': sum(population_CoT_fitness) / len(population_CoT_fitness),
                'mean_noCoT_fitness': sum(population_noCoT_fitness) / len(population_noCoT_fitness),
                'best_genome': self.str_format_genome(self.fitness_scorer.sort(self.population, descending=True)[0]),
            }

            if kwargs.get('verbose', False):
                print("Generation " + str(gen + 1) + " stats: ")
                print(gen_stats)
            if wandb.run is not None:
                wandb.log(gen_stats)
                wandb.log({
                    'mean_time_for_bte': (time.time() - start) / len(pairs),
                    'total_time_for_generation': time.time() - start,
                })

            # update tally of best genomes
            self.update_elite_history(next_population, gen + 1)

            # update generation log
            self.generations[gen + 1] = next_population
            # save generation log to run folder
            with open(self.output_dir + 'generation_' + str(gen) + '_log.json', 'w') as f:
                json.dump(self.population, f)
            with open(self.output_dir + 'generation_' + str(gen) + '_stats.json', 'w') as f:
                json.dump(gen_stats, f)
            # save random state
            with open(self.output_dir + 'generation_' + str(gen) + '_random_state.pkl', 'wb') as f:
                pickle.dump(random.getstate(), f)
            with open(self.output_dir + 'generation_' + str(gen) + '_torch_random_state.pkl', 'wb') as f:
                pickle.dump(torch.get_rng_state(), f)

            # update population
            self.population = next_population

        # save generation log to run folder
        with open(self.output_dir + 'generation_log.json', 'w') as f:
            json.dump(self.generations, f)

        with open(self.output_dir + 'elite_history.json', 'w') as f:
            json.dump(self.elite_history, f)

        with open(self.output_dir + 'new_mutation_prompts.json', 'w') as f:
            json.dump(self.new_mutation_prompts, f)

        with open(self.output_dir + 'complete', 'w') as f:
            f.write('')

    async def update_fitnesses(self, fitness_suite, gen):
        # collect async tasks
        async_fitness_tasks = []
        for genome in self.population:
            if 'fitness' not in genome.keys():
                async_fitness_tasks.extend(self.get_fitness_tests(genome, fitness_suite))

        start = time.time()

        fitness_results = {}
        for genome_id, fitness, CoT in await asyncio.gather(*async_fitness_tasks, return_exceptions=True):
            if isinstance(fitness, Exception):
                logging.error(f"Error Type: {type(fitness).__name__}, Error: {fitness}")
                continue

            # get genome by id
            if genome_id not in fitness_results.keys():
                fitness_results[genome_id] = {
                    'CoT': [],
                    'noCoT': [],
                }

            fitness_results[genome_id]['CoT' if CoT else 'noCoT'].append(fitness)

        if wandb.run is not None:
            wandb.log({
                'mean_time_for_fitness_tests': (time.time() - start) / len(async_fitness_tasks),
                'total_time_for_fitness_tests': time.time() - start,
            })

        for genome_id, fitnesses in fitness_results.items():
            genome = get_genome_by_id(self.population, genome_id)

            CoT_fitness_evals = fitnesses['CoT']
            noCoT_fitness_evals = fitnesses['noCoT']

            # add in correct reasonings
            if self.test_CoT:
                for correct_fitness in list(
                        compress(CoT_fitness_evals, self.fitness_scorer.get_correct(CoT_fitness_evals))):
                    self.correct_reasonings[genome_id] = correct_fitness['REASONING']

            if self.test_CoT:
                CoT = self.fitness_scorer(CoT_fitness_evals)
            else:
                CoT = self.fitness_scorer.worst_fitness()
            if self.test_noCoT:
                noCoT = self.fitness_scorer(noCoT_fitness_evals)
            else:
                noCoT = self.fitness_scorer.worst_fitness()
            best = CoT if self.fitness_scorer.compare(CoT, noCoT) else noCoT

            genome['fitness'] = {
                "CoT": CoT,
                "noCoT": noCoT,
                "best": best
            }
            print(genome['fitness'])

            if self.fitness_scorer.compare(genome['fitness']['CoT'], genome['fitness']['noCoT']):
                genome['CoT'] = True
            else:
                genome['CoT'] = False
            self.genome_to_dir(genome, gen)

    async def binary_tournament_evolve(self, genome1, genome2, fitness_suite, mutation_op, generation_num):

        if self.fitness_scorer.compare(genome1['fitness']['best'], genome2['fitness']['best']):
            elite_genome = genome1
            inferior_genome = genome2
        else:
            elite_genome = genome2
            inferior_genome = genome1

        # mutate elite genome
        mutant = await self.mutate(mutation_op, elite_genome, fitness_suite)

        if mutant is None:
            return elite_genome, inferior_genome
        else:
            self.genome_to_dir(mutant, generation_num)
            return elite_genome, mutant

    async def _async(self, func, *args, **kwargs):
        async with self.sem:
            return await func(*args, **kwargs)

    def load_generation(self, gen):
        with open(self.output_dir + 'generation_' + str(gen) + '_log.json') as f:
            self.population = json.load(f)
        self.update_elite_history(self.population, gen)
        self.generations[gen] = self.population
        # load random state
        with open(self.output_dir + 'generation_' + str(gen) + '_random_state.pkl', 'rb') as f:
            random.setstate(pickle.load(f))
        with open(self.output_dir + 'generation_' + str(gen) + '_torch_random_state.pkl', 'rb') as f:
            torch.set_rng_state(pickle.load(f))

    ################################  MUTATION FUNCTIONS  ################################
    async def mutate(self, mutation_op, genome=None, fitness_suite=None, regen_defects=False):
        if mutation_op == "mutation_zero_order_direct":
            new_mutant = await self.mutation_zero_order_direct()
        elif mutation_op == "mutation_first_order_direct":
            new_mutant = await self.mutation_first_order_direct(genome)
        elif mutation_op == "mutation_eda_random":
            new_mutant = await self.mutation_eda_random()
        elif mutation_op == "mutation_eda_ranked":
            new_mutant = await self.mutation_eda_ranked()
        elif mutation_op == "mutation_eda_lineage":
            new_mutant = await self.mutation_eda_lineage()
        elif mutation_op == "mutation_lamarckian":
            new_mutant = await self.mutation_lamarckian(genome)
        elif mutation_op == "hypermutation_zero_order_direct":
            new_mutant = await self.hypermutation_zero_order_direct(genome, fitness_suite)
        elif mutation_op == "hypermutation_first_order_direct":
            new_mutant = await self.hypermutation_first_order_direct(genome, fitness_suite)
        else:
            raise NotImplementedError('Mutation operator not implemented: ' + mutation_op)

        if is_birth_defect(new_mutant) and not regen_defects and False:
            self.p_par.update(1)
            return None
        elif is_birth_defect(new_mutant) and regen_defects:
            return await self.mutate("mutation_zero_order_direct", genome, fitness_suite, regen_defects=True)
        else:
            self.p_par.update(1)
            new_mutant['id'] = self.get_genome_id(new_mutant)
            return clean_newborn(new_mutant)

    async def hypermutation_first_order_direct(self, genome, fitness_suite):
        unit = {
            "mutation_prompt": self.mutation_prompts[str(random.choice(list(self.mutation_prompts.keys())))]
        }
        unit = get_query_args(self.breeder['hypermutation_first_order_direct_args'], unit)
        new_mutation_prompt = await self.run_query('hypermutation_first_order_direct', unit)
        new_mutation_prompt = new_mutation_prompt['MUTATED_TASK_PROMPT']
        if len(new_mutation_prompt.strip()) < 10:
            return None
        else:
            return await self._test_hypermutation(genome, new_mutation_prompt, fitness_suite)

    async def hypermutation_zero_order_direct(self, genome, fitness_suite):
        unit = {
            "thinking_style": self.thinking_styles[str(random.choice(list(self.thinking_styles.keys())))],
            "task_description": self.task_description['task_description'],
        }
        unit = get_query_args(self.breeder['hypermutation_zero_order_direct_args'], unit)
        new_mutation_prompt = await self.run_query('hypermutation_zero_order_direct', unit)
        new_mutation_prompt = new_mutation_prompt['MUTATION_PROMPT']
        if len(new_mutation_prompt.strip()) < 10:
            return None
        else:
            return await self._test_hypermutation(genome, new_mutation_prompt, fitness_suite)

    async def _test_hypermutation(self, genome, new_mutation_prompt, fitness_suite):
        unit = {
            "mutation_prompt": new_mutation_prompt,
            "thinking_style": (genome['thinking_style'] if 'thinking_style' in genome.keys() else None),
            "task_prompt": genome['task_prompt'],
        }
        # now run a mutation with the new mutation prompt
        alternate_mutant = await self.mutation_first_order_direct(genome, unit=unit)
        if is_birth_defect(alternate_mutant):
            return None
        else:
            # TODO put in fitness test for new task prompt to decide whether to keep the mutation prompt
            return alternate_mutant

    async def mutation_lamarckian(self, genome):
        if (genome_id := genome['id']) not in self.correct_reasonings.keys():
            genome_id = random.choice(list(self.correct_reasonings.keys()))
        str_genome = self.str_format_genome(get_genome_by_id(self.population, genome_id))
        task_prompt_correct_reasonings = self.correct_reasonings[genome_id]
        input_reasonings = random.sample(task_prompt_correct_reasonings,
                                         min(self.num_lamarckian_reasonings, len(task_prompt_correct_reasonings)))
        unit = {
            "mutation_prompt": self.mutation_prompts[str(random.choice(list(self.mutation_prompts.keys())))],
            "correct_reasonings": input_reasonings,
            "task_prompt": str_genome,
        }
        unit = get_query_args(self.breeder['mutation_lamarckian_args'], unit)
        return await self.run_query('mutation_lamarckian', unit)

    async def mutation_eda_lineage(self):
        random_elite_keys = random.sample(self.elite_history.keys(),
                                          min(self.num_eda_samples, len(self.elite_history.keys())))
        random_elites = {k: self.elite_history[k] for k in random_elite_keys}
        # deduplicate list of elites (give one attempt at replacement)
        ids = set()
        for k, d in random_elites.items():
            if id(d) in ids:
                replacement_key = random.choice(list(self.elite_history.keys()))
                if id(self.elite_history[replacement_key]) not in ids:
                    random_elite_keys.append(replacement_key)
                random_elite_keys.pop(random_elite_keys.index(k))
            ids.add(id(d))
        elite_lineage = [self.elite_history[k] for k in sorted(random_elite_keys)]
        unit = {
            "eda_method": "lineage",
            "mutation_prompt": self.mutation_prompts[str(random.choice(list(self.mutation_prompts.keys())))],
            "task_prompts": [self.str_format_genome(g) for g in elite_lineage],
        }
        unit = get_query_args(self.breeder['mutation_eda_based_args'], unit)

        return await self.run_query('mutation_eda_based', unit)

    async def mutation_eda_ranked(self):
        samples_with_fitness = [g for g in self.population if 'fitness' in g.keys()]
        random_samples = self.get_eda_samples(samples_with_fitness)
        ranked_samples = [self.str_format_genome(g) for g in self.fitness_scorer.sort(random_samples)]
        unit = {
            "eda_method": "ranked",
            "mutation_prompt": self.mutation_prompts[str(random.choice(list(self.mutation_prompts.keys())))],
            "task_prompts": ranked_samples,
        }
        unit = get_query_args(self.breeder['mutation_eda_based_args'], unit)
        return await self.run_query('mutation_eda_based', unit)

    async def mutation_eda_random(self):
        random_samples = self.get_eda_samples(self.population)
        unit = {
            "eda_method": "random",
            "mutation_prompt": self.mutation_prompts[str(random.choice(list(self.mutation_prompts.keys())))],
            "task_prompts": [self.str_format_genome(g) for g in random_samples],
        }
        unit = get_query_args(self.breeder['mutation_eda_based_args'], unit)
        return await self.run_query('mutation_eda_based', unit)

    async def mutation_first_order_direct(self, genome, unit=None):
        unit = {
            "mutation_prompt": self.mutation_prompts[str(random.choice(list(self.mutation_prompts.keys())))],
            "task_prompt": self.str_format_genome(genome),
        } if unit is None else unit
        unit = get_query_args(self.breeder['mutation_direct_args'], unit)
        return await self.run_query('mutation_direct', unit)

    async def mutation_zero_order_direct(self):
        unit = {
            "mutation_prompt": self.mutation_prompts[str(random.choice(list(self.mutation_prompts.keys())))],
            "thinking_style": self.thinking_styles[str(random.choice(list(self.thinking_styles.keys())))],
            "task_description": self.task_description['task_description'],
        }
        unit = get_query_args(self.breeder['mutation_direct_args'], unit)
        return await self.run_query('mutation_direct', unit)

    ####################################################

    def update_elite_history(self, population, current_generation):
        elite = self.fitness_scorer.sort([d for d in population if 'fitness' in d], descending=True)[0]
        self.elite_history[current_generation] = elite

    def get_eda_samples(self, samples):
        '''
        Gets a random sample of genomes from the population, does BERT filtering on them and returns them as a list of strings
        :param samples: base distribution of samples to choose from
        :return: random sample of genomes from the population (full genome dictionaries)
        '''
        random_samples = {self.str_format_genome(g): g for g in
                          random.sample(samples, min(len(samples), self.num_eda_samples * 2))}
        filtered_samples = [genome for dismiss, genome in
                            zip(self.bert_filterer(random_samples.keys()), random_samples.values()) if not dismiss]
        random_filtered_samples = random.sample(filtered_samples, min(self.num_eda_samples, len(filtered_samples)))
        return random_filtered_samples

    def str_format_genome(self, genome, CoT=True):
        CoT = genome['CoT'] if 'CoT' in genome.keys() else CoT
        return (("SYSTEM_PROMPT: " + genome['SYSTEM_PROMPT'] + " " if not self.task_description[
            'fixed_system_prompt'] else "")
                + "TASK_PROMPT: " + genome['TASK_PROMPT'] + (
                    " REASONING_PROMPT: " + genome['REASONING_PROMPT'] if CoT else ""))

    def get_fitness_tests(self, specimen, fitness_suite):
        async def run_fitness_test(genome_id, test_args, q, CoT):
            test_args['CoT'] = CoT
            test_result = await self.run_query('fitness_test', test_args)
            test_result[self.task] = test_result['REQUIREMENT']
            test_result['gold_' + self.task] = q['gold_' + self.task]
            return genome_id, test_result, CoT

        specimen_args = {
            "task_prompt": specimen['TASK_PROMPT'],
            "reasoning_prompt": specimen['REASONING_PROMPT'],
        }
        if not self.task_description['fixed_system_prompt']:
            specimen_args["system_prompt"] = specimen['SYSTEM_PROMPT']

        genome_id = specimen['id']

        fitness_tasks = []

        print('Running fitness tests for ' + self.str_format_genome(specimen) + '...')
        for q in fitness_suite:
            test_args = get_query_args(self.breeder['fitness_test_args'], specimen_args, **q)

            if self.test_CoT:
                # run fitness test with CoT
                fitness_tasks.append(self._async(run_fitness_test, genome_id, test_args, q, True))

            if self.test_noCoT:
                # run fitness test without CoT
                fitness_tasks.append(self._async(run_fitness_test, genome_id, test_args, q, False))

        return fitness_tasks

    def _get_breeder(self, day_care_dir, model_name_or_path, **kwargs):
        # get task specification
        with open(day_care_dir + self.task + '/task_description.json') as f:
            self.task_description = json.load(f)

        # load prompt specification
        with open(day_care_dir + "breeder.lmql") as f:
            breeder_spec = hjson.load(f)

        breeder_queries = {}
        try:
            fitness_test_template = breeder_spec['fitness_test_template']
        except KeyError:
            raise NotImplementedError('fitness_test_template not found in breeder.lmql')

        breeder_queries['fitness_test'] = self.load_query(fitness_test_template,
                                                          breeder_spec['fitness_test_template_args'] + ["model_kwargs",
                                                                                                        "CoT"],
                                                          model_name_or_path, fitness=True, **kwargs)
        breeder_queries['fitness_test_args'] = breeder_spec['fitness_test_template_args']

        mutation_operators = [key for key in breeder_spec.keys() if
                              (key.startswith('mutation_') or key.startswith('hypermutation_')) and not key.endswith(
                                  '_args')]
        for mutation_op in mutation_operators:
            breeder_queries[mutation_op] = self.load_query(breeder_spec[mutation_op],
                                                           breeder_spec[mutation_op + '_args'] + ["model_kwargs"],
                                                           model_name_or_path, **kwargs)
            breeder_queries[mutation_op + '_args'] = breeder_spec[mutation_op + '_args']
        return breeder_queries

    def load_query(self, query_spec, query_inputs, model_name_or_path, fitness=False, **kwargs):
        specification = format_specification(query_spec, model_name_or_path, (
            kwargs.get("decoder_options_override") if kwargs.get('decoder_options_override', None) else
            self.task_description[('fitness_' if fitness else '') + 'decoder_options']
        ), kwargs.get('server_batch_size', 1))
        req = lmql.query(specification, input_variables=query_inputs, is_async=self.is_async, chunksize=self.chunk_size)
        return req

    async def run_query(self, operation, arguments):
        if self.is_async:
            result = await self.breeder[operation](**arguments, model_kwargs=self.model_kwargs, verbose=True)
            result = result[0].variables
            result.update(arguments)
            return result
        else:
            result = self.breeder[operation](**arguments, model_kwargs=self.model_kwargs, verbose=True)[0].variables
            result.update(arguments)
            return result


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", help="Model to use (default: %(default)s)",
                        default="meta-llama/Llama-2-13b-chat-hf")
    parser.add_argument('--day_care_dir', help='Directory of breeding ground (default: %(default)s)',
                        default='./data/lmql_specifications/day_care/')
    parser.add_argument('--task', help='Task to put into the day care for breeding', required=True)
    parser.add_argument('--decoder_options_override',
                        help='Decoder options string to use instead of task description definition (default: %(default)s)',
                        default=None)
    parser.add_argument('--client_batch_size',
                        help='Batch size for client-side async querying (LMQL async querying) (default: %(default)s)',
                        default=1, type=int)
    parser.add_argument('--server_batch_size',
                        help='Batch size for use in on the inference with the model (LMQL async querying) (default: %(default)s)',
                        default=1, type=int)
    parser.add_argument('--fitness_test_suite', help='Fitness test suite to use (default: %(default)s)',
                        default='./data/fitness_test_suite.json')
    parser.add_argument('--population_size', help='Number of units in the population (default: %(default)s)',
                        default=50, type=int)
    parser.add_argument('--initial_prompts_per_unit',
                        help='Number of prompts to generate for each unit in the initial population (default: %(default)s)',
                        default=2, type=int)
    parser.add_argument('--evolution_generations',
                        help='Number of generations to evolve the population (default: %(default)s)', default=20,
                        type=int)
    parser.add_argument('--num_eda_samples',
                        help='Number of samples to use for EDA based mutations (default: %(default)s)', default=5,
                        type=int)
    parser.add_argument('--num_lamarckian_reasonings',
                        help='Number of reasonings to use for Lamarckian mutations (default: %(default)s)', default=2,
                        type=int)
    parser.add_argument('--chunk_size',
                        help='Number of tokens that are generated speculatively, in one LLM call (default: %(default)s)',
                        default=3, type=int)
    parser.add_argument('--test_CoT', help='Whether to test CoT (default: %(default)s)', action='store_true')
    parser.add_argument('--test_noCoT', help='Whether to test noCoT (default: %(default)s)', action='store_true')
    parser.add_argument('--verbose', help='Whether to use verbose mode (default: %(default)s)', action='store_true')
    parser.add_argument('--is_async', help='Whether to use async (default: %(default)s)', action='store_true')
    parser.add_argument('--seed', help='Random seed (default: %(default)s)', default=42, type=int)
    parser.add_argument('--continue_run_number', help='Continue from specified run number (default: %(default)s)',
                        default=None, type=int)
    parser.add_argument('--debug_mode', help='Whether to use debug mode (default: %(default)s)', action='store_true')
    parser.add_argument('--use_wandb', help='Whether to use wandb (default: %(default)s)', action='store_true')
    parser.add_argument('--wandb_project', help='Wandb project name (default: %(default)s)', default='prompt_breeder')
    args = parser.parse_args()
    return args


def main(args):
    # load input json
    with open(args.fitness_test_suite) as f:
        fitness_test_data = json.load(f)

    wandb_name = None
    wandb_id = None
    if args.use_wandb:
        # attempt to continue wandb run
        if args.continue_run_number is not None:
            output_dir = args.day_care_dir + '/' + args.task + '/RUN_' + str(args.continue_run_number) + '/'
            # load config
            with open(output_dir + '_config.json') as f:
                config = json.load(f)
            if 'wandb_id' in config.keys():
                wandb_id = config['wandb_id']
                wandb.init(args.wandb_project, config=config, id=wandb_id, resume='allow')

        # otherwise create new run
        if wandb.run is None:
            wandb.init(args.wandb_project, config=vars(args))
            wandb_name = wandb.run.name
            wandb_id = wandb.run.id

    # run day care main loop
    day_care = DayCare(wandb_id=wandb_id, wandb_name=wandb_name, **vars(args))
    if args.is_async:
        asyncio.run(day_care.breed(fitness_test_data, **vars(args)))
    else:
        sync(day_care.breed(fitness_test_data, **vars(args)))


if __name__ == '__main__':
    args = get_args()
    # fixes for multiprocessing
    torch.multiprocessing.set_start_method('spawn')
    nest_asyncio.apply()
    print("Verbose on? " + str(args.verbose))
    main(args)
