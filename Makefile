uid := $(shell id -u)

pycache := $(shell find $(abspath .) -iname __pycache__)

sparse_src_dir  := $(abspath ./sparse_framework)
sparse_data_dir := $(abspath ./data)
sparse_run_dir  := $(abspath ./run)
sparse_py       := $(shell find $(sparse_src_dir) -iname *.py)

docker_image      := sparse/pytorch
dockerfile        := Dockerfile
docker_build_file := .DOCKER

docker_run := docker run \
               --network host \
		       -v $(sparse_src_dir):/usr/lib/sparse_framework \
		       -v $(sparse_data_dir):/data \
		       -v $(sparse_run_dir):/run/sparse \
		       -v $(abspath .):/app

ifneq (,$(shell uname -a | grep tegra))
	docker_base_image=nvcr.io/nvidia/l4t-pytorch:r34.1.0-pth1.12-py3
else
	docker_base_image=pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
endif

$(sparse_data_dir):
	mkdir $(sparse_data_dir)

$(docker_build_file): $(sparse_py) $(dockerfile)
	docker build . --no-cache --build-arg BASE_IMAGE=$(docker_base_image) -t $(docker_image)
	touch $(docker_build_file)

.PHONY: all
all: $(sparse_data_dir) $(docker_build_file)
	make -C examples/split_learning all

.PHONY: docker
docker: $(docker_build_file)
	make -C examples/split_learning docker
	make -C examples/split_inference docker

.PHONY: run
run: | $(sparse_data_dir) $(docker_build_file)
	make run-learning-aio

# Learning
.PHONY: run-sparse-monitor
run-sparse-monitor: | $(sparse_data_dir) $(docker_build_file)
	$(docker_run) \
		--name sparse_monitor_server \
		-d $(docker_image)

.PHONY: run-learning-aio
run-learning-aio: | $(sparse_data_dir) $(docker_build_file)
	make -C examples/split_learning run-aio

.PHONY: run-learning-data-source
run-learning-data-source: | $(sparse_data_dir) $(docker_build_file)
	make -C examples/split_learning run-data-source

.PHONY: run-learning-unsplit
run-learning-unsplit: | $(sparse_data_dir) $(docker_build_file)
	make -C examples/split_learning run-unsplit-final

.PHONY: run-learning-split
run-learning-split: | $(sparse_data_dir) $(docker_build_file)
	make -C examples/split_learning run-split-final
	make -C examples/split_learning run-split-intermediate

.PHONY: run-learning-split-final
run-learning-split-final: | $(sparse_data_dir) $(docker_build_file)
	make -C examples/split_learning run-split-final

.PHONY: run-learning-split-client
run-learning-split-client: | $(sparse_data_dir) $(docker_build_file)
	make -C examples/split_learning run-split-client

.PHONY: run-learning-split-intermediate
run-learning-split-intermediate: | $(sparse_data_dir) $(docker_build_file)
	make -C examples/split_learning run-split-intermediate

# Inference
.PHONY: run-inference-aio
run-inference-aio: | $(sparse_data_dir) $(docker_build_file)
	make -C examples/split_inference run-aio

.PHONY: run-inference-data-source
run-inference-data-source: | $(sparse_data_dir) $(docker_build_file)
	make -C examples/split_inference run-data-source

.PHONY: run-inference-unsplit
run-inference-unsplit: | $(sparse_data_dir) $(docker_build_file)
	make -C examples/split_inference run-unsplit-final

.PHONY: run-inference-split
run-inference-split: | $(sparse_data_dir) $(docker_build_file)
	make -C examples/split_inference run-split-final
	make -C examples/split_inference run-split-intermediate

.PHONY: run-inference-split-final
run-inference-split-final: | $(sparse_data_dir) $(docker_build_file)
	make -C examples/split_inference run-split-final

.PHONY: run-inference-split-client
run-inference-split-client: | $(sparse_data_dir) $(docker_build_file)
	make -C examples/split_inference run-split-client

.PHONY: run-inference-split-intermediate
run-inference-split-intermediate: | $(sparse_data_dir) $(docker_build_file)
	make -C examples/split_inference run-split-intermediate

.PHONY: clean
clean:
	make -iC examples/split_learning clean
	make -iC examples/split_inference clean
	docker container prune -f
	docker image prune -f
	sudo rm -rf $(pycache) $(docker_build_file)
