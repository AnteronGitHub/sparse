py_dir              := ./src
py_main             := $(py_dir)/edge-deep-learning.py
py_segnet           := $(py_dir)/segnet.py
py_training         := $(py_dir)/training.py
py_training_requirements  := requirements-training.txt
py_stats            := $(py_dir)/collect-statistics.py
py_cache            := $(shell find . -iname __pycache__)
py_venv             := venv

data_dir            := data

samples_dir            := $(data_dir)/samples
samples_init_script    := ./scripts/init-test-data.sh
sample_classification  := orange_0.jpg
sample_segmentation    := city_0.jpg

stats_dir           := $(data_dir)/stats

build_dir := build

jetson_install_pytorch_script        := ./scripts/jetson-install-pytorch.sh
jetson_install_torchvision_script    := ./scripts/jetson-install-torchvision.sh

$(samples_dir)/$(sample_classification):
	$(samples_init_script) $(samples_dir) $(sample_classification)

$(samples_dir)/$(sample_segmentation):
	$(samples_init_script) $(samples_dir) $(sample_segmentation)

$(stats_dir):
	mkdir -p $(stats_dir)

$(py_venv): $(py_venv)/touchfile

$(py_venv)/touchfile: $(py_training_requirements)
	python3 -m venv $(py_venv)
	$(py_venv)/bin/pip install -r $(py_training_requirements)
	touch $(py_venv)/touchfile

.PHONY: jetson-dependencies
jetson-dependencies:
	$(jetson_install_pytorch_script)
	$(jetson_install_torchvision_script)

.PHONY: run-classification
run-classification: $(samples_dir)/$(sample_classification)
	python3 $(py_main) $(samples_dir)/$(sample_classification)

.PHONY: run-segnet
run-segnet: $(samples_dir)/$(sample_segmentation)
	python3 $(py_segnet) $(samples_dir)/$(sample_segmentation)

.PHONY: run-training
run-training:
	python3 $(py_training)

.PHONY: run-training-venv
run-training-venv: $(py_venv) $(py_training_requirements)
	$(py_venv)/bin/python $(py_training)

.PHONY: run
run:
	make run-training

.PHONY: run-venv
run-venv:
	make run-training-venv

.PHONY: collect-stats
collect-stats: $(stats_dir)
	python3 $(py_stats)

.PHONY: clean
clean:
	rm -rf $(data_dir) $(py_cache) $(py_venv) $(build_dir)
