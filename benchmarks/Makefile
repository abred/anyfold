CXXFLAGS = -c -Wall -std=c++11 -DHAS_OPENCL -DPROJECT_ROOT_DIR="\"$(shell pwd)/../\""
INCLUDE = -I../include -I/opt/AMDAPPSDK-2.9-1/include
LDFLAGS = ../build/src/libanyfold.a -L/opt/AMDAPPSDK-2.9-1/lib/x86_64 -lOpenCL


SOURCES = benchmarks.cpp \


OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE = benchmarks

# all: $(SOURCES) $(EXECUTABLE)
all: debug

debug: CXXFLAGS += -g -DDEBUG
debug: $(EXECUTABLE)

release: CXX_FLAGS += -O2
release: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) 
	$(CXX)  $(OBJECTS) -o $@ $(LDFLAGS)

.cpp.o:
	$(CXX) $(CXXFLAGS) $(INCLUDE) $< -o $@
# oder %.o : %.cpp

clean:
	rm *.o
