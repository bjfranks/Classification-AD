#!/usr/bin/env bash


mkdir ../log/cifar10_oe_cifar100;

methods=( hsc bce );
n_classes=( 1 2 4 8 16 32 64 100 );
sizes=( 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 );

for seed in $(seq 1 10);
  do
    for k in "${n_classes[@]}";
      do
        for size in "${sizes[@]}";
          do
            for exp in $(seq 0 9);
              do
                for method in "${methods[@]}";
                  do
                    mkdir ../log/cifar10_oe_cifar100/${method};
                    mkdir ../log/cifar10_oe_cifar100/${method}/oe_size_${size};
                    mkdir ../log/cifar10_oe_cifar100/${method}/oe_size_${size}/${k}_oe_classes;
                    mkdir ../log/cifar10_oe_cifar100/${method}/oe_size_${size}/${k}_oe_classes/${exp}_vs_rest;
                    mkdir ../log/cifar10_oe_cifar100/${method}/oe_size_${size}/${k}_oe_classes/${exp}_vs_rest/seed_${seed};
                    python main.py cifar10 cifar10_LeNet ../log/cifar10_oe_cifar100/${method}/oe_size_${size}/${k}_oe_classes/${exp}_vs_rest/seed_${seed} ../data --rep_dim 256 --objective ${method} --outlier_exposure True --oe_dataset_name cifar100 --oe_size ${size} --oe_n_classes ${k} --device cuda:0 --seed ${seed} --lr 0.001 --n_epochs 200 --lr_milestone 100 --lr_milestone 150 --batch_size 128 --data_augmentation True --data_normalization True --normal_class ${exp};
                  done
              done
          done
      done
  done
