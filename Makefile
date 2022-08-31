uid := $(shell id -u)

pycache := $(shell find . -iname __pycache__)

sparse_src := src
sparse_py  := $(shell find $(sparse_src) -iname *.py)

docker_image      := sparse/pytorch
dockerfile        := Dockerfile
docker_build_file := .DOCKER

.PHONY: all
all: $(docker_build_file)
	make -C examples/split_learning all

.PHONY: run
run: | $(docker_build_file)
	make run-learning-aio

.PHONY: run-monitor
run-monitor: | $(docker_build_file)
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

.PHONY: clean
clean:
	make -iC examples/split_learning clean
	docker container prune -f
	docker image prune -f
	rm -rf $(pycache) $(docker_build_file)

$(docker_build_file): $(sparse_py) $(dockerfile)
	docker build . -t $(docker_image)
	touch $(docker_build_file)
