py_dir              := ./src
py_main             := $(py_dir)/edge-deep-learning.py
py_stats            := $(py_dir)/collect-statistics.py

data_dir            := data

samples_dir         := $(data_dir)/samples
samples_init_script := ./scripts/init-test-data.sh
samples_filename    := orange_0.jpg

stats_dir           := $(data_dir)/stats

$(samples_dir):
	$(samples_init_script) $(samples_dir) $(samples_filename)

$(stats_dir):
	mkdir -p $(stats_dir)

.PHONY: run
run: $(samples_dir)
	python3 $(py_main) $(samples_dir)/$(samples_filename)

.PHONY: collect-stats
collect-stats: $(stats_dir)
	python3 $(py_stats)

.PHONY: clean
clean:
	rm -rf $(data_dir)
