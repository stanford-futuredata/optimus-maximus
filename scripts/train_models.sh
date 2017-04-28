!/usr/bin/env bash

set -x

./nomad_train_netflix.sh 0.5 10 ~/data/netflix_prize_dataset/nomad-format ~/models/nomad/Netflix-10-reg-0.5
./nomad_train_netflix.sh 0.05 10 ~/data/netflix_prize_dataset/nomad-format ~/models/nomad/Netflix-10-reg-0.05
./nomad_train_netflix.sh 0.005 10 ~/data/netflix_prize_dataset/nomad-format ~/models/nomad/Netflix-10-reg-0.005
./nomad_train_netflix.sh 0.0005 10 ~/data/netflix_prize_dataset/nomad-format ~/models/nomad/Netflix-10-reg-0.0005

./nomad_train_netflix.sh 0.5 25 ~/data/netflix_prize_dataset/nomad-format ~/models/nomad/Netflix-25-reg-0.5
./nomad_train_netflix.sh 0.05 25 ~/data/netflix_prize_dataset/nomad-format ~/models/nomad/Netflix-25-reg-0.05
./nomad_train_netflix.sh 0.005 25 ~/data/netflix_prize_dataset/nomad-format ~/models/nomad/Netflix-25-reg-0.005
./nomad_train_netflix.sh 0.0005 25 ~/data/netflix_prize_dataset/nomad-format ~/models/nomad/Netflix-25-reg-0.0005

./nomad_train_netflix.sh 0.5 50 ~/data/netflix_prize_dataset/nomad-format ~/models/nomad/Netflix-50-reg-0.5
./nomad_train_netflix.sh 0.05 50 ~/data/netflix_prize_dataset/nomad-format ~/models/nomad/Netflix-50-reg-0.05
./nomad_train_netflix.sh 0.005 50 ~/data/netflix_prize_dataset/nomad-format ~/models/nomad/Netflix-50-reg-0.005
./nomad_train_netflix.sh 0.0005 50 ~/data/netflix_prize_dataset/nomad-format ~/models/nomad/Netflix-50-reg-0.0005

./nomad_train_netflix.sh 0.5 100 ~/data/netflix_prize_dataset/nomad-format ~/models/nomad/Netflix-100-reg-0.5
./nomad_train_netflix.sh 0.05 100 ~/data/netflix_prize_dataset/nomad-format ~/models/nomad/Netflix-100-gold-standard
./nomad_train_netflix.sh 0.005 100 ~/data/netflix_prize_dataset/nomad-format ~/models/nomad/Netflix-100-reg-0.005
./nomad_train_netflix.sh 0.0005 100 ~/data/netflix_prize_dataset/nomad-format ~/models/nomad/Netflix-100-reg-0.0005

./nomad_train.sh 1 10 ~/data/yahoo_music_R2_dataset/nomad-format ~/models/nomad/R2-10-reg-1
./nomad_train.sh 0.1 10 ~/data/yahoo_music_R2_dataset/nomad-format ~/models/nomad/R2-10-reg-0.1
./nomad_train.sh 0.01 10 ~/data/yahoo_music_R2_dataset/nomad-format ~/models/nomad/R2-10-reg-0.01
./nomad_train.sh 0.001 10 ~/data/yahoo_music_R2_dataset/nomad-format ~/models/nomad/R2-10-reg-0.001
./nomad_train.sh 0.0001 10 ~/data/yahoo_music_R2_dataset/nomad-format ~/models/nomad/R2-10-reg-0.0001

./nomad_train.sh 1 25 ~/data/yahoo_music_R2_dataset/nomad-format ~/models/nomad/R2-25-reg-1
./nomad_train.sh 0.1 25 ~/data/yahoo_music_R2_dataset/nomad-format ~/models/nomad/R2-25-reg-0.1
./nomad_train.sh 0.01 25 ~/data/yahoo_music_R2_dataset/nomad-format ~/models/nomad/R2-25-reg-0.01
./nomad_train.sh 0.001 25 ~/data/yahoo_music_R2_dataset/nomad-format ~/models/nomad/R2-25-reg-0.001
./nomad_train.sh 0.0001 25 ~/data/yahoo_music_R2_dataset/nomad-format ~/models/nomad/R2-25-reg-0.0001

./nomad_train.sh 1 50 ~/data/yahoo_music_R2_dataset/nomad-format ~/models/nomad/R2-50-reg-1
./nomad_train.sh 0.1 50 ~/data/yahoo_music_R2_dataset/nomad-format ~/models/nomad/R2-50-reg-0.1
./nomad_train.sh 0.01 50 ~/data/yahoo_music_R2_dataset/nomad-format ~/models/nomad/R2-50-reg-0.01
./nomad_train.sh 0.001 50 ~/data/yahoo_music_R2_dataset/nomad-format ~/models/nomad/R2-50-reg-0.001
./nomad_train.sh 0.0001 50 ~/data/yahoo_music_R2_dataset/nomad-format ~/models/nomad/R2-50-reg-0.0001

./nomad_train.sh 1 100 ~/data/yahoo_music_R2_dataset/nomad-format ~/models/nomad/R2-100-reg-1
./nomad_train.sh 0.1 100 ~/data/yahoo_music_R2_dataset/nomad-format ~/models/nomad/R2-100-reg-0.1
./nomad_train.sh 0.01 100 ~/data/yahoo_music_R2_dataset/nomad-format ~/models/nomad/R2-100-reg-0.01
./nomad_train.sh 0.001 100 ~/data/yahoo_music_R2_dataset/nomad-format ~/models/nomad/R2-100-reg-0.001
./nomad_train.sh 0.0001 100 ~/data/yahoo_music_R2_dataset/nomad-format ~/models/nomad/R2-100-reg-0.0001

./nomad_train.sh 1 10 ~/data/yahoo_KDD_cup_2011_dataset/nomad-format ~/models/nomad/KDD-10-reg-1
./nomad_train.sh 0.1 10 ~/data/yahoo_KDD_cup_2011_dataset/nomad-format ~/models/nomad/KDD-10-reg-0.1
./nomad_train.sh 0.01 10 ~/data/yahoo_KDD_cup_2011_dataset/nomad-format ~/models/nomad/KDD-10-reg-0.01
./nomad_train.sh 0.001 10 ~/data/yahoo_KDD_cup_2011_dataset/nomad-format ~/models/nomad/KDD-10-reg-0.001
./nomad_train.sh 0.0001 10 ~/data/yahoo_KDD_cup_2011_dataset/nomad-format ~/models/nomad/KDD-10-reg-0.0001

./nomad_train.sh 1 25 ~/data/yahoo_KDD_cup_2011_dataset/nomad-format ~/models/nomad/KDD-25-reg-1
./nomad_train.sh 0.1 25 ~/data/yahoo_KDD_cup_2011_dataset/nomad-format ~/models/nomad/KDD-25-reg-0.1
./nomad_train.sh 0.01 25 ~/data/yahoo_KDD_cup_2011_dataset/nomad-format ~/models/nomad/KDD-25-reg-0.01
./nomad_train.sh 0.001 25 ~/data/yahoo_KDD_cup_2011_dataset/nomad-format ~/models/nomad/KDD-25-reg-0.001
./nomad_train.sh 0.0001 25 ~/data/yahoo_KDD_cup_2011_dataset/nomad-format ~/models/nomad/KDD-25-reg-0.0001

./nomad_train.sh 1 50 ~/data/yahoo_KDD_cup_2011_dataset/nomad-format ~/models/nomad/KDD-50-reg-1
./nomad_train.sh 0.1 50 ~/data/yahoo_KDD_cup_2011_dataset/nomad-format ~/models/nomad/KDD-50-reg-0.1
./nomad_train.sh 0.01 50 ~/data/yahoo_KDD_cup_2011_dataset/nomad-format ~/models/nomad/KDD-50-reg-0.01
./nomad_train.sh 0.001 50 ~/data/yahoo_KDD_cup_2011_dataset/nomad-format ~/models/nomad/KDD-50-reg-0.001
./nomad_train.sh 0.0001 50 ~/data/yahoo_KDD_cup_2011_dataset/nomad-format ~/models/nomad/KDD-50-reg-0.0001

./nomad_train.sh 1 100 ~/data/yahoo_KDD_cup_2011_dataset/nomad-format ~/models/nomad/KDD-100-gold-standard
./nomad_train.sh 0.1 100 ~/data/yahoo_KDD_cup_2011_dataset/nomad-format ~/models/nomad/KDD-100-reg-0.1
./nomad_train.sh 0.01 100 ~/data/yahoo_KDD_cup_2011_dataset/nomad-format ~/models/nomad/KDD-100-reg-0.01
./nomad_train.sh 0.001 100 ~/data/yahoo_KDD_cup_2011_dataset/nomad-format ~/models/nomad/KDD-100-reg-0.001
./nomad_train.sh 0.0001 100 ~/data/yahoo_KDD_cup_2011_dataset/nomad-format ~/models/nomad/KDD-100-reg-0.0001

