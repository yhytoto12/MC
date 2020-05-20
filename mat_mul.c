#include "mat_mul.h"

#include <stdio.h>
#include <CL/cl.h>

#define CHECK_ERROR(err) \
  if (err != CL_SUCCESS) { \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
    exit(EXIT_FAILURE); \
  }

#define BS 56
#define ITEMS 8
#define MAX_DEV 4 

static cl_int err;
static cl_platform_id platform;
static cl_device_id device[MAX_DEV];
static cl_context context;
static cl_command_queue queue[MAX_DEV];
static cl_program program;
static cl_kernel kernel[MAX_DEV];
static cl_mem a_d[MAX_DEV];
static cl_mem b_d[MAX_DEV];
static cl_mem c_d[MAX_DEV];
static int ndev;
static size_t MD[MAX_DEV];

void mat_mul(float *A, float *B, float *C, int M, int N, int K) {
  // Write to GPU; A (cpu) -> a_d (gpu), B (cpu) -> b_d (gpu)
	size_t nextA = (size_t)A;
  for (int i = 0; i < ndev; i++) {
    err = clEnqueueWriteBuffer(queue[i], a_d[i], CL_TRUE, 0, MD[i] * K * sizeof(float), (void *)nextA, 0, NULL, NULL);
		CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue[i], b_d[i], CL_TRUE, 0, K * N * sizeof(float), B, 0, NULL, NULL);
    CHECK_ERROR(err);
		nextA += MD[i] * K * sizeof(float);
  }

  // Setup kernel arguments
  for (int i = 0; i < ndev; i++) {
    err = clSetKernelArg(kernel[i], 0, sizeof(cl_mem), &a_d[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[i], 1, sizeof(cl_mem), &b_d[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[i], 2, sizeof(cl_mem), &c_d[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[i], 3, sizeof(int), &MD[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[i], 4, sizeof(int), &N);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[i], 5, sizeof(int), &K);
    CHECK_ERROR(err);
  }


  // Setup global work size and local work size
  size_t gws[2] = {(M + ndev - 1) / ndev, (N + ITEMS - 1) / ITEMS}, lws[2] = {BS, BS / ITEMS};
  for (int i = 0; i < 2; ++i) {
    // By OpenCL spec, global work size should be MULTIPLE of local work size
    // Formula below achieve it
    // e.g., gws = 25, lws = 16, then (25 + 16 - 1) / 16 * 16 = 40 / 16 * 16 = 2 * 16 = 32
    gws[i] = (gws[i] + lws[i] - 1) / lws[i] * lws[i];
	}
  // Run kernel
  for(int i = 0; i < ndev; i++) {
    err = clEnqueueNDRangeKernel(queue[i], kernel[i], 2, NULL, gws, lws, 0, NULL, NULL);
    CHECK_ERROR(err);
  }

  // Read from GPU; c_d (gpu) -> C (cpu)
	size_t nextC = (size_t)C;
  for (int i = 0 ; i < ndev; i++) {
    err = clEnqueueReadBuffer(queue[i], c_d[i], CL_TRUE, 0, MD[i] * N * sizeof(float), (void *)nextC, 0, NULL, NULL);
    CHECK_ERROR(err);
		nextC += MD[i] * N * sizeof(float);
  }

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  for(int i = 0; i < ndev; i++) {
    err = clFinish(queue[i]);
    CHECK_ERROR(err);
  }
}

static void print_platform_info(cl_platform_id platform) {
  size_t sz;
  char *buf;
  CHECK_ERROR(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &sz));
  buf = (char*)malloc(sz);
  CHECK_ERROR(clGetPlatformInfo(platform, CL_PLATFORM_NAME, sz, buf, NULL));
  printf("Detected OpenCL platform: %s\n", buf);
  free(buf);
}

static void print_device_info(cl_device_id* device) {
  size_t sz;
  char *buf;
  for (int i = 0; i < ndev; i++) {
    CHECK_ERROR(clGetDeviceInfo(device[i], CL_DEVICE_NAME, 0, NULL, &sz));
    buf = (char *)malloc(sz);
    CHECK_ERROR(clGetDeviceInfo(device[i], CL_DEVICE_NAME, sz, buf, NULL));
    printf("Detected OpenCL device: %s\n", buf);
    free(buf);
  }
}

static cl_program create_and_build_program_with_source(cl_context context, cl_device_id *device, const char *file_name) {
  FILE *file = fopen(file_name, "rb");
  if (file == NULL) {
    printf("Failed to open %s\n", file_name);
    exit(EXIT_FAILURE);
  }
  fseek(file, 0, SEEK_END);
  size_t source_size = ftell(file);
  rewind(file);
  char *source_code = (char*)malloc(source_size + 1);
  fread(source_code, sizeof(char), source_size, file);
  source_code[source_size] = '\0';
  fclose(file);
  // printf("source code :\n%s", source_code); // for debug
  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_code, &source_size, &err);
  CHECK_ERROR(err);
  free(source_code);
  err = clBuildProgram(program, ndev, device, NULL, NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE) {
    size_t log_size;
    for (int i = 0; i < ndev; i++) {
      CHECK_ERROR(clGetProgramBuildInfo(program, device[i], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));
      char *log = (char *)malloc(log_size + 1);
      CHECK_ERROR(clGetProgramBuildInfo(program, device[i], CL_PROGRAM_BUILD_LOG, log_size, log, NULL));
      log[log_size] = 0;
      printf("Compile error:\n%s\n", log);
      free(log);
    }
  }
  CHECK_ERROR(err);
  return program;
}

void mat_mul_init(float *A, float *B, float *C, int M, int N, int K) {

  // Get OpenCL platform
  err = clGetPlatformIDs(1, &platform, NULL);
  CHECK_ERROR(err);
  print_platform_info(platform);

  // Get OpenCL device
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, (unsigned int *) &ndev);
  CHECK_ERROR(err);
	ndev = (ndev < M ? ndev : M);								// ndev = min(ndev, M, MAX_DEV)
	ndev = (ndev < MAX_DEV ? ndev : MAX_DEV);
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, ndev, device, NULL);
	CHECK_ERROR(err);
  print_device_info(device);

  // Create OpenCL context
  context = clCreateContext(NULL, ndev, device, NULL, NULL, &err);
  CHECK_ERROR(err);

  // Create OpenCL command queue
	for (int i = 0; i < ndev; i++) {
  	queue[i] = clCreateCommandQueue(context, device[i], 0, &err);
	  CHECK_ERROR(err);
  }

  // Compile program from "kernel.cl"
  program = create_and_build_program_with_source(context, device, "kernel.cl");

  // Extract kernel from compiled program
  for(int i = 0; i < ndev; i++) {
    kernel[i] = clCreateKernel(program, "sgemm", &err);
    CHECK_ERROR(err);
  }

	for(int i = 0; i < ndev; i++) {
		MD[i] = M * (i + 1) / ndev - M * i / ndev;
	}

  // Create GPU buffers
  for(int i = 0; i < ndev; i++) {
    a_d[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, MD[i] * K * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    b_d[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, K * N * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    c_d[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, MD[i] * N * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
  }
  



  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  for(int i = 0; i < ndev; i++) {
    err = clFinish(queue[i]);
    CHECK_ERROR(err);
  }
}

void mat_mul_final(float *A, float *B, float *C, int M, int N, int K) {
  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  for (int i = 0; i < ndev; i++) {
    err = clFinish(queue[i]);
    CHECK_ERROR(err);
  }
}
