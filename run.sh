python run.py --do_train --do_valid --evaluate_train \
  --model TransE -n 128 -b 512 -d 100 -g 30 -a 1.0 -adv \
  -lr 0.0001 --max_steps 2 --cpu_num 2 --test_batch_size 32 \
  --ntriples_eval_train 200