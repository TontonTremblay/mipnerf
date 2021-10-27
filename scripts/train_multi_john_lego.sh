#!/bin/bash
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Script for training on the multiscale Blender dataset.

SCENE=lego
EXPERIMENT=debug
# TRAIN_DIR=/Users/barron/tmp/nerf_results/$EXPERIMENT/$SCENE
TRAIN_DIR=/home/jtremblay/code/mipnerf/out_google_0/
# DATA_DIR=/home/jtremblay/code/mipnerf/multiplescale/lego/
DATA_DIR=/media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/falling_google_scenes/00000/mip/
mkdir $TRAIN_DIR

python -m train \
  --data_dir=$DATA_DIR \
  --train_dir=$TRAIN_DIR \
  --gin_file=configs/multiblender.gin \
  --logtostderr
