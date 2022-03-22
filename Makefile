py_main          := ./src/edge-deep-learning.py

data_init_script := ./scripts/init-test-data.sh
data_dir         := data
data_filename    := orange_0.jpg

$(data_dir):
	$(data_init_script) $(data_dir) $(data_filename)

.PHONY: run
run: $(data_dir)
	python3 $(py_main) $(data_dir)/$(data_filename)

.PHONY: clean
clean:
	rm -rf $(data_dir)
