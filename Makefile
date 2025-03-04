CC := g++
SRCDIR := src
BUILDDIR := build
TARGET := automata

EXTRA_FLAGS := $(EXTRA_FLAGS)
ifdef CPU_ONLY
EXTRA_FLAGS := $(EXTRA_FLAGS) -DCPU_ONLY
endif
ifdef HEADLESS_ONLY
EXTRA_FLAGS := $(EXTRA_FLAGS) -DHEADLESS_ONLY
endif
ifdef NH_RADIUS
EXTRA_FLAGS := $(EXTRA_FLAGS) -DNH_RADIUS=$(NH_RADIUS)
endif
ifdef AUTOMATA_TYPE_BIT
EXTRA_FLAGS := $(EXTRA_FLAGS) -DAUTOMATA_TYPE_BIT
endif

COMMON_FLAGS := -Wall -O3 # Optimization level 3
CFLAGS := -std=c++17 -fopenmp $(COMMON_FLAGS) $(EXTRA_FLAGS) # -g
CUFLAGS := --compiler-options "$(COMMON_FLAGS)" $(EXTRA_FLAGS) -m64 -gencode arch=compute_75,code=sm_75
# my rtx 2080 is compute capability 7.5 https://developer.nvidia.com/cuda-gpus
#-gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80


SOURCES := $(shell find $(SRCDIR) -type f -name *.cpp)
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.cpp=.o))

ifdef CPU_ONLY
CUSOURCES := $()
CUOBJECTS := $()
else
CUSOURCES := $(shell find $(SRCDIR) -type f -name *.cu)
CUOBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(CUSOURCES:.cu=.o))
endif

LIB := -lboost_program_options -fopenmp -lfmt
ifndef HEADLESS_ONLY
LIB := $(LIB) -lGL -lGLU -lglut -lGLEW
endif
ifndef CPU_ONLY
# if lcudart cannot be found, check if the env var 
# LIBRARY_PATH is pointing to the correct cuda install location
LIB := $(LIB) -lcudart
endif

INC := $()
ifndef CPU_ONLY
INC := $(INC) -I/usr/local/cuda/include
endif

run: $(TARGET)
	@echo "\033[1;37mRunning" $(TARGET) "\033[0m"; 
	./$(TARGET) --render

profile: $(TARGET)
	@echo "\033[1;37mProfiling" $(TARGET) "\033[0m"; 
	nsys profile --stats=true --trace=cuda --cudabacktrace=true --cuda-memory-usage=true -o report -f true ./$(TARGET) -b

$(TARGET): $(OBJECTS) $(CUOBJECTS)
	@echo "\033[1;37mLinking" $(TARGET) "\033[0m"
	$(CC) $^ -o $(TARGET) $(LIB)
	@echo "\033[1;37mCompiled successfully\033[0m"

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp
	@echo "\033[1;37mBuilding" $@ "\033[0m"
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) $(INC) -c -o $@ $<

$(BUILDDIR)/%.o: $(SRCDIR)/%.cu
	@echo "\033[1;37mBuilding" $@ "\033[0m"
	@mkdir -p $(BUILDDIR)
	nvcc -ccbin $(CC) $(INC) $(CUFLAGS) -c -o $@ $<

clean:
	@echo "\033[1;37mCleaning...\033[0m"; 
	$(RM) -r $(BUILDDIR) $(TARGET) *.qdrep *.sqlite callgrind.out*


.PHONY: clean