uid := $(shell id -u)

pycache := $(shell find . -iname __pycache__)

sparse_src := src
sparse_py  := $(shell find $(sparse_src) -iname *.py)

docker_image      := sparse/pytorch
dockerfile        := Dockerfile
docker_build_file := .DOCKER

ifneq (,$(shell uname -a | grep tegra))
	docker_base_image=nvcr.io/nvidia/l4t-pytorch:r34.1.0-pth1.12-py3
else
	docker_base_image=pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
endif

.PHONY: all
all: $(docker_build_file)
	make -C examples/split_learning all

.PHONY: run
run: | $(docker_build_file)
	make run-learning-aio

# Learning
.PHONY: run-learning-monitor
run-learning-monitor: | $(docker_build_file)
	make -C examples/split_learning run-monitor

.PHONY: run-learning-aio
run-learning-aio: | $(docker_build_file)
	make -C examples/split_learning run-aio

.PHONY: run-learning-data-source
run-learning-data-source: | $(docker_build_file)
	make -C examples/split_learning run-data-source

.PHONY: run-learning-unsplit
run-learning-unsplit: | $(docker_build_file)
	make -C examples/split_learning run-unsplit-final

.PHONY: run-learning-split
run-learning-split: | $(docker_build_file)
	make -C examples/split_learning run-split-final
	make -C examples/split_learning run-split-intermediate

.PHONY: run-learning-split-final
run-learning-split-final: | $(docker_build_file)
	make -C examples/split_learning run-split-final

.PHONY: run-learning-split-intermediate
run-learning-split-intermediate: | $(docker_build_file)
	make -C examples/split_learning run-split-intermediate

# Inference
.PHONY: run-inference-monitor
run-inference-monitor: | $(docker_build_file)
	make -C examples/split_inference run-monitor

.PHONY: run-inference-aio
run-inference-aio: | $(docker_build_file)
	make -C examples/split_inference run-aio

.PHONY: run-inference-data-source
run-inference-data-source: | $(docker_build_file)
	make -C examples/split_inference run-data-source

.PHONY: run-inference-unsplit
run-inference-unsplit: | $(docker_build_file)
	make -C examples/split_inference run-unsplit-final

.PHONY: run-inference-split
run-inference-split: | $(docker_build_file)
	make -C examples/split_inference run-split-final
	make -C examples/split_inference run-split-intermediate

.PHONY: run-inference-split-final
run-inference-split-final: | $(docker_build_file)
	make -C examples/split_inference run-split-final

.PHONY: run-inference-split-client
run-inference-split-client: | $(docker_build_file)
	make -C examples/split_inference run-split-client

.PHONY: run-inference-split-intermediate
run-inference-split-intermediate: | $(docker_build_file)
	make -C examples/split_inference run-split-intermediate

.PHONY: clean
clean:
	make -iC examples/split_learning clean
	make -iC examples/split_inference clean
	docker container prune -f
	docker image prune -f
	sudo rm -rf $(pycache) $(docker_build_file)

$(docker_build_file): $(sparse_py) $(dockerfile)
	docker build . --build-arg BASE_IMAGE=$(docker_base_image) -t $(docker_image)
	touch $(docker_build_file)
