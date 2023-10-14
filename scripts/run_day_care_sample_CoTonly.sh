python ../LMQL_prompt_breeder.py \
     --task answerability \
     --fitness_test_suite ../data/squad_reqs_fitness_test_suite.json \
     --use_wandb \
     --verbose \
     --test_CoT \
     --population 25 \
     --day_care_dir ../data/lmql_specifications/day_care/