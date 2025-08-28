CXX = g++
CXXFLAGS = -std=c++17 -O3 -fopenmp
BUILD_DIR = build
TARGETS = $(BUILD_DIR)/main_naive $(BUILD_DIR)/main_ours $(BUILD_DIR)/main_noGF $(BUILD_DIR)/main_noLF $(BUILD_DIR)/main_noFB

all: $(BUILD_DIR) $(TARGETS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/main_naive: main_naive.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BUILD_DIR)/main_ours: main_ours.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BUILD_DIR)/main_noGF: main_noGF.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BUILD_DIR)/main_noLF: main_noLF.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BUILD_DIR)/main_noFB: main_noFB.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean