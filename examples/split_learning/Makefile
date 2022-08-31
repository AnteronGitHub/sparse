uid := $(shell id -u)
base_image_file := $(abspath ../../.DOCKER)
py_sources := $(shell find . -iname *.py)

docker_image := split_learning
docker_build_file := .DOCKER

.PHONY: all
all: $(docker_build_file)

.PHONY: run-aio
run-aio: | $(docker_build_file)
	docker run \
               -u $(uid) \
			   -v $(abspath ../../src):/usr/lib/sparse \
			   -v $(abspath .):/app \
			   --rm -it $(docker_image)

.PHONY: run-monitor
run-monitor: | $(docker_build_file)
	docker run --name sparse_monitor \
               -u $(uid) \
			   -v $(abspath ../../src):/usr/lib/sparse \
			   -v $(abspath .):/app \
			   -d $(docker_image) \
			   python benchmark.py --suite monitor

.PHONY: run-data-source
run-data-source: | $(docker_build_file)
	docker run --name split_learning_source \
               --network host \
			   -u $(uid) \
			   -v $(abspath ../../src):/usr/lib/sparse \
			   -v $(abspath .):/app \
			   -d $(docker_image) \
			   python benchmark.py --suite offload_source

.PHONY: run-unsplit-final
run-unsplit-final: | $(docker_build_file)
	docker run --name split_learning_final \
               --network host \
			   -u $(uid) \
			   -v $(abspath ../../src):/usr/lib/sparse \
			   -v $(abspath .):/app \
			   -d $(docker_image) \
			   python benchmark.py --suite offload_final

.PHONY: run-split-final
run-split-final: | $(docker_build_file)
	docker run --name split_learning_final \
               --network host \
			   -u $(uid) \
			   -e WORKER_LISTEN_PORT=50008 \
			   -v $(abspath ../../src):/usr/lib/sparse \
			   -v $(abspath .):/app \
			   -d $(docker_image) \
			   python benchmark.py --suite offload_final --model VGG_server

.PHONY: run-split-intermediate
run-split-intermediate: | $(docker_build_file)
	docker run --name split_learning_intermediate \
               --network host \
			   -u $(uid) \
			   -e MASTER_UPSTREAM_PORT=50008 \
			   -v $(abspath ../../src):/usr/lib/sparse \
			   -v $(abspath .):/app \
			   -d $(docker_image) \
			   python benchmark.py --suite offload_intermediate --model VGG_client

.PHONY: clean
clean:
	docker kill sparse_monitor split_learning_source split_learning_final split_learning_intermediate
	rm -f $(docker_build_file)

$(docker_build_file): $(py_sources) $(base_image_file)
	docker build . -t $(docker_image)
	touch $(docker_build_file)

