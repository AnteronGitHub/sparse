base_image_file    := $(abspath ../../.DOCKER)
docker_base_image  := anterondocker/sparse-framework
py_sources 			   := $(shell find . -iname '*.py')

docker_build_file  := .DOCKER
docker_image       := anterondocker/splitnn

ifneq (,$(shell uname -a | grep tegra))
	docker_tag=l4t-pytorch
else
	docker_tag=pytorch
endif

$(docker_build_file): $(py_sources) $(base_image_file)
	docker build . --build-arg TAGNAME=$(docker_tag) \
                 -t $(docker_image):$(docker_tag)
	touch $(docker_build_file)

.PHONY: docker clean

docker: $(docker_build_file)

clean:
	sudo rm -f $(docker_build_file)

