pycache := $(shell find $(abspath .) -iname __pycache__)

sparse_src_dir  := $(abspath ./sparse_framework)
sparse_py       := $(shell find $(sparse_src_dir) -iname '*.py')
sparse_stats_dir := /var/lib/sparse/stats

docker_image      := sparse/pytorch
dockerfile        := Dockerfile
docker_build_file := .DOCKER

docker_image_graphs      := sparse/graphs
dockerfile_graphs        := Dockerfile.graphs
docker_build_file_graphs := .DOCKER_GRAPHS
src_graphs               := make_graphs.py
py_requirements_graphs   := requirements_graphs.txt

ifneq (,$(shell uname -a | grep tegra))
	docker_base_image=nvcr.io/nvidia/l4t-pytorch:r34.1.0-pth1.12-py3
	docker_py_requirements=requirements_jetson.txt
else
	docker_base_image=pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
	docker_py_requirements=requirements.txt
endif

$(docker_build_file): $(sparse_py) $(dockerfile)
	docker build . --no-cache \
                 --build-arg BASE_IMAGE=$(docker_base_image) \
								 --build-arg PY_REQUIREMENTS=$(docker_py_requirements) \
								 -t $(docker_image)
	docker image prune -f
	touch $(docker_build_file)

$(docker_build_file_graphs): $(py_requirements_graphs) $(dockerfile_graphs)
	docker build . -f $(dockerfile_graphs) \
								 -t $(docker_image_graphs)
	docker image prune -f
	touch $(docker_build_file_graphs)

.PHONY: docker clean run run-experiment clean-experiment graphs

docker: $(docker_build_file)
	make -C examples/splitnn docker
	make -C examples/deprune docker

clean:
	make -iC examples/splitnn clean
	make -iC examples/deprune clean
	docker container prune -f
	docker image prune -f
	sudo rm -rf $(pycache) $(docker_build_file)

run run-experiment:
	scripts/run-experiment.sh

clean-experiment:
	scripts/clean-experiment.sh

graphs: $(docker_build_file_graphs)
	docker run --rm -v $(sparse_stats_dir):$(sparse_stats_dir) -v $(abspath $(src_graphs)):/app/$(src_graphs) -it $(docker_image_graphs)
