python preprocess_use_res_as_input.py \
    --data_path ${1} \
    --data_name train_with_res

python preprocess_use_res_as_input.py \
    --data_path ${2} \
    --data_name eval_with_res

python run_simpletod_plus.py \
    --output_dir ./simpletod_plus_chitchat_inlast_domainmodify_7epoch_1e-3/ \
    --model_name_or_path=gpt2 \
    --model_type=gpt2 \
    --train_file=./cache/train_with_res.jsonl \
    --validation_file=./cache/eval_with_res.jsonl \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 6 \
    --num_train_epochs 7 \
    --learning_rate 1e-3