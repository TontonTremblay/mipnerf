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

# Script for training on the Blender dataset.

SCENE=lego
EXPERIMENT=debug
# TRAIN_DIR=/home/jtremblay/code/mipnerf/lego_v8/
TRAIN_DIR=/home/jtremblay/code/mipnerf/google0_1/
# DATA_DIR=/media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/nvisii_mvs_10/lego/V8/mip/
DATA_DIR=/media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/falling_google_scenes/00000/mip/

mkdir $TRAIN_DIR
rm $TRAIN_DIR/*
python -m train \
  --data_dir=$DATA_DIR \
  --train_dir=$TRAIN_DIR \
  --gin_file=configs/blender.gin \
  --logtostderr
