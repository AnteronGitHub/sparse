uid             := $(shell id -u)
base_image_file := $(abspath ../../.DOCKER)
py_sources      := $(shell find . -iname *.py)

docker_image      := split_inference
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
	docker run --name inference_monitor \
               -u $(uid) \
			   -v $(abspath ../../src):/usr/lib/sparse \
			   -v $(abspath .):/app \
			   -d $(docker_image) \
			   python benchmark.py --suite monitor

.PHONY: run-data-source
run-data-source: | $(docker_build_file)
	docker run --name split_inference_source \
               --network host \
			   -u $(uid) \
			   -v $(abspath ../../src):/usr/lib/sparse \
			   -v $(abspath .):/app \
			   -d $(docker_image) \
			   python benchmark.py --suite offload_source

.PHONY: run-unsplit-final
run-unsplit-final: | $(docker_build_file)
	docker run --name split_inference_final \
               --network host \
			   -u $(uid) \
			   -v $(abspath ../../src):/usr/lib/sparse \
			   -v $(abspath .):/app \
			   -d $(docker_image) \
			   python benchmark.py --suite offload_final --model YOLOv3

.PHONY: run-split-final
run-split-final: | $(docker_build_file)
	docker run --name split_inference_final \
               --network host \
			   -u $(uid) \
			   -e WORKER_LISTEN_PORT=50008 \
			   -v $(abspath ../../src):/usr/lib/sparse \
			   -v $(abspath .):/app \
			   -d $(docker_image) \
			   python benchmark.py --suite offload_final --model YOLOv3_server

.PHONY: run-split-intermediate
run-split-intermediate: | $(docker_build_file)
	docker run --name split_inference_intermediate \
               --network host \
			   -u $(uid) \
			   -e MASTER_UPSTREAM_PORT=50008 \
			   -v $(abspath ../../src):/usr/lib/sparse \
			   -v $(abspath .):/app \
			   -d $(docker_image) \
			   python benchmark.py --suite offload_intermediate --model YOLOv3_client

.PHONY: clean
clean:
	docker kill inference_monitor split_inference_source split_inference_final split_inference_intermediate
	rm -f $(docker_build_file)

$(docker_build_file): $(py_sources) $(base_image_file)
	docker build . -t $(docker_image)
	touch $(docker_build_file)

