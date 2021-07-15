python preprocess_with_state.py \
    --data_path ${1} \
    --data_name train

python preprocess_with_state.py \
    --data_path ${2} \
    --data_name eval

python run_simpletod_plus.py \
    --output_dir ./simpletod_plus_7epoch_1e-3/ \
    --model_name_or_path=gpt2 \
    --model_type=gpt2 \
    --train_file=./cache/train.jsonl \
    --validation_file=./cache/eval.jsonl \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 6 \
    --num_train_epochs 7 \
    --learning_rate 1e-3