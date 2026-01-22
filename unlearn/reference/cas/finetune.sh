#!/bin/bash

#SBATCH --job-name=c_full_e2eu                                                      # Job name
#SBATCH --output=/home/a5m/scasper.a5m/owsg/outfiles/c_full_e2eu_%A_%a.out            # Standard output and error log
#SBATCH --array=59                 					                               # Array range
#SBATCH --time=24:00:00                                                                # Time limit hrs:min:sec
#SBATCH --reservation=aisi                                                             # Reservation
#SBATCH --gpus=4                                                                       # Number of GPUs

echo "Job ID: $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

cd /lus/lfs1aip1/home/a5m/scasper.a5m/
# conda init
source miniforge3/bin/activate
conda activate owsg
cd /lus/lfs1aip1/home/a5m/scasper.a5m/owsg

source model_names.sh
model=${PARAM_LIST[$SLURM_ARRAY_TASK_ID]}

if [[ "$model" == *"ga-1-in-41"* ]]; then
    revision='main'
elif [[ "$model" == *"rmu"* || "$model" == *"5percent"* || "$model" == *"lie_o"* || "$model" == *"smollm"* || "$model" == *"deepfried"* || "$model" == *"deep_fry"* || "$model" == *"replace"* || "$model" == *"early"* || "$model" == *"Eleuther"* ]]; then
    revision='main'
else
    revision='annealing_step_11921'
fi

# Lora and full finetuning attacks just on the forget set
# python finetune.py --model_name=${model} --revision=${revision} --lr=2e-5 --num_train_examples=24453 --include_bio_remove=True --epochs=2 --lora=True --open_book_eval=True --eval_every=500
python finetune.py --model_name=${model} --revision=${revision} --lr=2e-5 --num_train_examples=24453 --include_bio_remove=True --epochs=2 --open_book_eval=True --eval_every=500

# Lora and full finetuning attacks just on the forget set with warmup
# python finetune.py --model_name=${model} --revision=${revision} --lr=2e-5 --num_train_examples=24453 --include_bio_remove=True --epochs=2 --lora=True --open_book_eval=True --eval_every=500 --lr_warmup=1000
# python finetune.py --model_name=${model} --revision=${revision} --lr=2e-5 --num_train_examples=24453 --include_bio_remove=True --epochs=2 --open_book_eval=True --eval_every=500 --lr_warmup=1000

# Lora finetune on wikitext and forget data mixed
# python finetune.py --model_name=${model} --revision=${revision} --lr=2e-5 --num_train_examples=24453 --include_text_retain=True --include_bio_remove=True --epochs=2 --lora=True --open_book_eval=True --eval_every=500
# python finetune.py --model_name=${model} --revision=${revision} --lr=2e-5 --num_train_examples=24453 --include_text_retain=True --include_bio_remove=True --epochs=2 --open_book_eval=True --eval_every=500

# Lora finetune on wikitext and forget data mixed with warmup
# python finetune.py --model_name=${model} --revision=${revision} --lr=2e-5 --num_train_examples=24453 --include_text_retain=True --include_bio_remove=True --epochs=2 --lora=True --open_book_eval=True --eval_every=500 --lr_warmup=1000
# python finetune.py --model_name=${model} --revision=${revision} --lr=2e-5 --num_train_examples=24453 --include_text_retain=True --include_bio_remove=True --epochs=2 --open_book_eval=True --eval_every=500 --lr_warmup=1000

# Lora finetune on benign only
# python finetune.py --model_name=${model} --revision=${revision} --lr=2e-5 --num_train_examples=29444 --include_text_retain=True --epochs=3 --open_book_eval=True --eval_every=500 --max_chunks=4 --lora=True
# python finetune.py --model_name=${model} --revision=${revision} --lr=2e-5 --num_train_examples=29444 --include_text_retain=True --epochs=3 --open_book_eval=True --eval_every=500 --max_chunks=4

# cmu models
# python finetune.py --model_name=${model} --revision=${revision} --lr=1e-5 --num_train_examples=3900 --epochs=5 --include_compliance_remove=True --lora=True --lr_warmup=250
# python finetune.py --model_name=${model} --revision=${revision} --lr=1e-5 --num_train_examples=3900 --epochs=5 --include_compliance_remove=True --lr_warmup=250
