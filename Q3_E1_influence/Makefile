CXX = g++
CXXFLAGS = -std=c++17 -O3 -fopenmp
TARGETS = influence_IC_each_node influence_SIR_each_node influence_LT_each_node

all: $(TARGETS)

influence_IC_each_node: influence_IC_each_node.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

influence_SIR_each_node: influence_SIR_each_node.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

influence_LT_each_node: influence_LT_each_node.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean