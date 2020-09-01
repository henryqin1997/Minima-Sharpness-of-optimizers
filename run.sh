#!/bin/sh
echo First group:
CUDA_VISIBLE_DEVICES=4 python3 onecycle_lb_make_ckpt.py --optimizer sgd --max-lr 0.001 --batch-size 8192 &
CUDA_VISIBLE_DEVICES=5 python3 onecycle_lb_make_ckpt.py --optimizer sgdwm --max-lr 0.001 --batch-size 8192 &
CUDA_VISIBLE_DEVICES=6 python3 onecycle_lb_make_ckpt.py --optimizer novograd --max-lr 0.2 --batch-size 8192 &
CUDA_VISIBLE_DEVICES=7 python3 onecycle_lb_make_ckpt.py --optimizer rmsprop --max-lr 0.0003 --batch-size 8192 &
wait

echo Second group:
CUDA_VISIBLE_DEVICES=4 python3 onecycle_lb_make_ckpt.py --optimizer adam --max-lr 0.01 --batch-size 8192 &
CUDA_VISIBLE_DEVICES=5 python3 onecycle_lb_make_ckpt.py --optimizer lamb --max-lr 0.02 --batch-size 8192 &
CUDA_VISIBLE_DEVICES=6 python3 onecycle_lb_make_ckpt.py --optimizer radam --max-lr 0.0033 --batch-size 8192 --pct-start 0.5 &
CUDA_VISIBLE_DEVICES=7 python3 onecycle_lb_make_ckpt.py --optimizer radam --max-lr 0.0033 --batch-size 8192 --pct-start 0.15
