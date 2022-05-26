py_dir              := ./src
py_jetson_demo      := $(py_dir)/jetson_demo.py
py_segnet           := $(py_dir)/segnet.py
py_client           := $(py_dir)/split_training_client.py
py_client_requirements  := requirements-client.txt
py_server           := $(py_dir)/split_training_server.py
py_stats            := ./collect_statistics.py
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

$(py_venv)/touchfile: $(py_client_requirements)
	python3 -m venv $(py_venv)
	$(py_venv)/bin/pip install -r $(py_client_requirements)
	touch $(py_venv)/touchfile

.PHONY: jetson-dependencies
jetson-dependencies:
	$(jetson_install_pytorch_script)
	$(jetson_install_torchvision_script)

.PHONY: run-jetson-demo
run-jetson-demo: $(samples_dir)/$(sample_classification)
	python3 $(py_jetson_demo) $(samples_dir)/$(sample_classification)

.PHONY: run-segnet
run-segnet: $(samples_dir)/$(sample_segmentation)
	python3 $(py_segnet) $(samples_dir)/$(sample_segmentation)

.PHONY: run-server
run-server:
	python3 $(py_server)

.PHONY: run-client
run-client:
	python3 $(py_client)

.PHONY: run-client-venv
run-client-venv: $(py_venv) $(py_client_requirements)
	$(py_venv)/bin/python $(py_client)

.PHONY: run
run:
	make run-client

.PHONY: run-venv
run-venv:
	make run-client-venv

.PHONY: collect-stats
collect-stats:
	python3 $(py_stats)

.PHONY: clean
clean:
	rm -rf $(data_dir) $(py_cache) $(py_venv) $(build_dir)
