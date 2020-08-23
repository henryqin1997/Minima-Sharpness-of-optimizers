# Minima-Sharpness-of-optimizers
Introduce a new benchmark to measure the generalization of different optimizers

## To run one-cycle lr scheduling on CIFAR10 RESNET50
python3 onecycle_cifar_res.py --optimizer sgd --max-lr 0.2

optimizer option: sgd sgdwm adam radam novograd rmsprop lars lamb novograd
