NVCC = nvcc
NVCC_FLAGS = -std=c++11 -arch=sm_80 -lineinfo -lcublas -lcusparse 


##################################################################

## Project file structure ##

# Source file directory:
SRC_DIR = src

# Object file directory:
OBJ_DIR = .

# Include header file directory
INC_DIR = include


##################################################################

## Compile ##

matmul_benchmark: matrixmul.o
	@$(NVCC) $(NVCC_FLAGS) $< -o $@
	
# Compile CUDA source files to object files
%.o : %.cu
	@$(NVCC) $(NVCC_FLAGS) -x cu -c $< -o $@

clean:
	@rm -f $(OBJ_DIR)/*.o
