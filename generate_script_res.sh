python preprocess_use_res_as_input.py \
    --data_path ${1} \
    --data_name test \
    --do_test

python generate_simpletod_with_res.py \
    --model_type=gpt2 \
    --model_name_or_path ./simpletod_plus_chitchat_inlast_domainmodify_7epoch_1e-3/ \
    --test_file=./cache/test.jsonl \
    --output_path ./test_prediction.json \
    --per_device_test_batch_size 1

python post_preprocess.py \
    --output_path ${2} \
    --data_path ./test_prediction.json \
    --test_dir ${1} \
    --chit_chat_state True \
    --do_chit_chat