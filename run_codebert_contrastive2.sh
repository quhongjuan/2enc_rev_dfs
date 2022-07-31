lang=java_bert_cont2 #programming language
lr=5e-5
batch_size=2
beam_size=1
source_length=200
target_length=200
data_dir=./data
output_dir=./${lang}
train_file=./data/train_pdg.json
dev_file=./data/dev_pdg.json
test_file=./data/test_pdg.json
eval_steps=2901 #400 for ruby, 600 for javascript, 1000 for others
train_steps=871400 #20000 for ruby, 30000 for javascript, 50000 for others
pretrained_model=./codebert-base #Roberta: roberta-base
#pretrained_model=./${lang}/checkpoint-best-bleu/
gradient_accumulation_steps=1
device=cuda:0

# CUDA_VISIBLE_DEVICES=0
# debug test --do_test --do_eval --model_type roberta --model_name_or_path ./codebert-base --load_model_path ./java/checkpoint-best-bleu/pytorch_model.bin --train_filename ./data/train.json --test_filename ./data/test/json --dev_filename ./data/dev.json --output_dir ./java --max_source_length 200 --max_target_length 200 --beam_size 1 --train_batch_size 2 --eval_batch_size 2 --learning_rate 5e-5 --train_steps 50000 --eval_steps 2000
# debug train --do_train --do_eval --model_type roberta --model_name_or_path ./codebert-base --train_filename ./data/train.json --test_filename ./data/test/json --dev_filename ./data/dev.json --output_dir ./java --max_source_length 200 --max_target_length 200 --beam_size 1 --train_batch_size 2 --eval_batch_size 2 --learning_rate 5e-5 --train_steps 50000 --eval_steps 2000
python3 run_codebert_contrastive2.py --device $device --do_train --do_eval --gradient_accumulation_steps $gradient_accumulation_steps --model_type roberta --model_name_or_path $pretrained_model --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --train_steps $train_steps --eval_steps $eval_steps
