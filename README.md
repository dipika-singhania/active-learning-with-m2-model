# active-learning-with-m2-model

# How to run the file
python main.py --dataset FashionMNIST --tensorboard

# How to see tensorboard logs
tensorboard --logdir=/mnt/data/captioning_dataset/active_learning/results/FashionMNIST/tb_logs/  --port=8008

ssh -N -L localhost:8900:localhost:8008 dipika16@vacherin.d2.comp.nus.edu.sg

#
