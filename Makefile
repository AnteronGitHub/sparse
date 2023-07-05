pycache := $(shell find $(abspath .) -iname __pycache__)

sparse_src_dir  := $(abspath ./sparse_framework)
sparse_py       := $(shell find $(sparse_src_dir) -iname '*.py')

docker_image      := sparse/pytorch
dockerfile        := Dockerfile
docker_build_file := .DOCKER

ifneq (,$(shell uname -a | grep tegra))
	docker_base_image=nvcr.io/nvidia/l4t-pytorch:r34.1.0-pth1.12-py3
else
	docker_base_image=pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
endif

$(docker_build_file): $(sparse_py) $(dockerfile)
	docker build . --no-cache --build-arg BASE_IMAGE=$(docker_base_image) -t $(docker_image)
	touch $(docker_build_file)

.PHONY: docker clean run-experiment clean-experiment

docker: $(docker_build_file)
	make -C examples/splitnn docker
	make -C examples/deprune docker

clean:
	make -iC examples/splitnn clean
	make -iC examples/deprune clean
	docker container prune -f
	docker image prune -f
	sudo rm -rf $(pycache) $(docker_build_file)

run-experiment:
	scripts/run-experiment.sh

clean-experiment:
	scripts/clean-experiment.sh

