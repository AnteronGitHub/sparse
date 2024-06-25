pycache := $(shell find $(abspath .) -iname __pycache__)

sparse_src_dir  := $(abspath ./sparse_framework)
sparse_py       := $(shell find $(sparse_src_dir) -iname '*.py')
sparse_stats_dir := /var/lib/sparse/stats

docker_image      := anterondocker/sparse-framework
dockerfile        := Dockerfile
docker_build_file := .DOCKER

docker_tag_graphs        := graphs
dockerfile_graphs        := Dockerfile.graphs
docker_build_file_graphs := .DOCKER_GRAPHS
src_graphs               := make_graphs.py
py_requirements_graphs   := requirements_graphs.txt

ifneq (,$(shell uname -a | grep tegra))
	docker_base_image=nvcr.io/nvidia/l4t-pytorch:r34.1.0-pth1.12-py3
	docker_tag=l4t-pytorch
	docker_py_requirements=requirements_jetson.txt
else
	docker_base_image=pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
	docker_tag=pytorch
	docker_py_requirements=requirements.txt
endif

$(docker_build_file): $(sparse_py) $(dockerfile)
	docker build . --no-cache \
                 --build-arg BASE_IMAGE=$(docker_base_image) \
								 --build-arg PY_REQUIREMENTS=$(docker_py_requirements) \
								 -t $(docker_image):$(docker_tag)
	docker build . -f $(dockerfile_graphs) \
		-t $(docker_image):$(docker_tag_graphs)
	docker image prune -f
	touch $(docker_build_file)

.PHONY: docker clean run run-experiment clean-experiment graphs

docker: $(docker_build_file)
	#make -C examples/splitnn docker
	# Deprecated
#	make -C examples/deprune docker

clean:
	scripts/delete_worker.sh

run:
	scripts/deploy_worker.sh

run-experiment:
	scripts/run-experiment.sh

clean-experiment:
	scripts/clean-experiment.sh

graphs: $(docker_build_file)
	docker run --rm -v $(sparse_stats_dir):$(sparse_stats_dir) \
		              -v $(abspath $(src_graphs)):/app/$(src_graphs) \
									-it $(docker_image):$(docker_tag_graphs)
