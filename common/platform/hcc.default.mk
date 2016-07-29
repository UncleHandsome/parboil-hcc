# (c) 2011 The Board of Trustees of the University of Illinois.

# Cuda-related definitions common to all benchmarks

########################################
# Variables
########################################

# c.default is the base along with CUDA configuration in this setting
include $(PARBOIL_ROOT)/common/platform/c.default.mk

# Programs
HCC_BIN=$(HCC_BUILD_PATH)/compiler/bin/clang++

# Flags
PLATFORM_CXXFLAGS=-hc -std=c++amp -stdlib=libc++ -I$(HCC_SOURCE_PATH)/include -O3
PLATFORM_LDFLAGS=-lm -lpthread -L$(HCC_BUILD_PATH)/lib -Wl,--rpath=$(HCC_BUILD_PATH)/lib\
                 -lc++ -lc++abi -lm -ldl -lpthread -Wl,--whole-archive -lmcwamp -Wl,--no-whole-archive
