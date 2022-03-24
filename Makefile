py_dir              := ./src
py_main             := $(py_dir)/edge-deep-learning.py
py_segnet           := $(py_dir)/segnet.py
py_stats            := $(py_dir)/collect-statistics.py
py_cache            := $(shell find . -iname __pycache__)

data_dir            := data

samples_dir            := $(data_dir)/samples
samples_init_script    := ./scripts/init-test-data.sh
sample_classification  := orange_0.jpg
sample_segmentation    := city_0.jpg

stats_dir           := $(data_dir)/stats

$(samples_dir)/$(sample_classification):
	$(samples_init_script) $(samples_dir) $(sample_classification)

$(samples_dir)/$(sample_segmentation):
	$(samples_init_script) $(samples_dir) $(sample_segmentation)

$(stats_dir):
	mkdir -p $(stats_dir)

.PHONY: run-classification
run-classification: $(samples_dir)/$(sample_classification)
	python3 $(py_main) $(samples_dir)/$(sample_classification)

.PHONY: run-segnet
run-segnet: $(samples_dir)/$(sample_segmentation)
	python3 $(py_segnet) $(samples_dir)/$(sample_segmentation)

.PHONY: run
run:
	make run-classification

.PHONY: collect-stats
collect-stats: $(stats_dir)
	python3 $(py_stats)

.PHONY: clean
clean:
	rm -rf $(data_dir) $(py_cache)
