TARGET := face_detection
SRC := main.cpp

PKG_CONFIG ?= pkg-config
OPENCV_CFLAGS := $(shell $(PKG_CONFIG) --cflags opencv4)
OPENCV_LIBS := $(shell $(PKG_CONFIG) --libs opencv4)

CXX := g++
CXXFLAGS ?=
LDFLAGS ?=

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -o $@ $^ $(LDFLAGS) $(OPENCV_LIBS)

.PHONY: run clean

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
