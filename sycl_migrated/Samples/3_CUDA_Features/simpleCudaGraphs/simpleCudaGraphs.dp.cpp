/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* DPCT_ORIG #include <cooperative_groups.h>*/
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
/* DPCT_ORIG #include <cuda_runtime.h>*/
#include <helper_cuda.h>
#include <vector>
#include <chrono>

/* DPCT_ORIG namespace cg = cooperative_groups;*/

#define THREADS_PER_BLOCK 512
#define GRAPH_LAUNCH_ITERATIONS 3

typedef struct callBackData {
  const char *fn_name;
  double *data;
} callBackData_t;

/* DPCT_ORIG __global__ void reduce(float *inputVec, double *outputVec, size_t
   inputSize, size_t outputSize) {*/
void reduce(float *inputVec, double *outputVec, size_t inputSize,
            size_t outputSize, const sycl::nd_item<3> &item_ct1, double *tmp) {
/* DPCT_ORIG   __shared__ double tmp[THREADS_PER_BLOCK];*/

/* DPCT_ORIG   cg::thread_block cta = cg::this_thread_block();*/
  auto cta = item_ct1.get_group();
/* DPCT_ORIG   size_t globaltid = blockIdx.x * blockDim.x + threadIdx.x;*/
  size_t globaltid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2);

  double temp_sum = 0.0;
/* DPCT_ORIG   for (int i = globaltid; i < inputSize; i += gridDim.x *
 * blockDim.x) {*/
  for (int i = globaltid; i < inputSize;
       i += item_ct1.get_group_range(2) * item_ct1.get_local_range(2)) {
    temp_sum += (double)inputVec[i];
  }
/* DPCT_ORIG   tmp[cta.thread_rank()] = temp_sum;*/
  tmp[item_ct1.get_local_linear_id()] = temp_sum;

/* DPCT_ORIG   cg::sync(cta);*/
  /*
  DPCT1065:0: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

/* DPCT_ORIG   cg::thread_block_tile<32> tile32 =
 * cg::tiled_partition<32>(cta);*/
  sycl::sub_group tile32 = item_ct1.get_sub_group();

  double beta = temp_sum;
  double temp;

/* DPCT_ORIG   for (int i = tile32.size() / 2; i > 0; i >>= 1) {*/
  for (int i = item_ct1.get_sub_group().get_local_linear_range() / 2; i > 0;
       i >>= 1) {
/* DPCT_ORIG     if (tile32.thread_rank() < i) {*/
    if (item_ct1.get_sub_group().get_local_linear_id() < i) {
/* DPCT_ORIG       temp = tmp[cta.thread_rank() + i];*/
      temp = tmp[item_ct1.get_local_linear_id() + i];
      beta += temp;
/* DPCT_ORIG       tmp[cta.thread_rank()] = beta;*/
      tmp[item_ct1.get_local_linear_id()] = beta;
    }
/* DPCT_ORIG     cg::sync(tile32);*/
    /*
    DPCT1065:2: Consider replacing sycl::sub_group::barrier() with
    sycl::sub_group::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.get_sub_group().barrier();
  }
/* DPCT_ORIG   cg::sync(cta);*/
  /*
  DPCT1065:1: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

/* DPCT_ORIG   if (cta.thread_rank() == 0 && blockIdx.x < outputSize) {*/
  if (item_ct1.get_local_linear_id() == 0 &&
      item_ct1.get_group(2) < outputSize) {
    beta = 0.0;
/* DPCT_ORIG     for (int i = 0; i < cta.size(); i += tile32.size()) {*/
    for (int i = 0; i < item_ct1.get_group().get_local_linear_range();
         i += item_ct1.get_sub_group().get_local_linear_range()) {
      beta += tmp[i];
    }
/* DPCT_ORIG     outputVec[blockIdx.x] = beta;*/
    outputVec[item_ct1.get_group(2)] = beta;
  }
}

/* DPCT_ORIG __global__ void reduceFinal(double *inputVec, double *result,
                            size_t inputSize) {*/
void reduceFinal(double *inputVec, double *result, size_t inputSize,
                 const sycl::nd_item<3> &item_ct1, double *tmp) {
/* DPCT_ORIG   __shared__ double tmp[THREADS_PER_BLOCK];*/

/* DPCT_ORIG   cg::thread_block cta = cg::this_thread_block();*/
  auto cta = item_ct1.get_group();
/* DPCT_ORIG   size_t globaltid = blockIdx.x * blockDim.x + threadIdx.x;*/
  size_t globaltid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2);

  double temp_sum = 0.0;
/* DPCT_ORIG   for (int i = globaltid; i < inputSize; i += gridDim.x *
 * blockDim.x) {*/
  for (int i = globaltid; i < inputSize;
       i += item_ct1.get_group_range(2) * item_ct1.get_local_range(2)) {
    temp_sum += (double)inputVec[i];
  }
/* DPCT_ORIG   tmp[cta.thread_rank()] = temp_sum;*/
  tmp[item_ct1.get_local_linear_id()] = temp_sum;

/* DPCT_ORIG   cg::sync(cta);*/
  /*
  DPCT1065:3: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

/* DPCT_ORIG   cg::thread_block_tile<32> tile32 =
 * cg::tiled_partition<32>(cta);*/
  sycl::sub_group tile32 = item_ct1.get_sub_group();

  // do reduction in shared mem
/* DPCT_ORIG   if ((blockDim.x >= 512) && (cta.thread_rank() < 256)) {*/
  if ((item_ct1.get_local_range(2) >= 512) &&
      (item_ct1.get_local_linear_id() < 256)) {
/* DPCT_ORIG     tmp[cta.thread_rank()] = temp_sum = temp_sum +
 * tmp[cta.thread_rank() + 256];*/
    tmp[item_ct1.get_local_linear_id()] = temp_sum =
        temp_sum + tmp[item_ct1.get_local_linear_id() + 256];
  }

/* DPCT_ORIG   cg::sync(cta);*/
  /*
  DPCT1065:4: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

/* DPCT_ORIG   if ((blockDim.x >= 256) && (cta.thread_rank() < 128)) {*/
  if ((item_ct1.get_local_range(2) >= 256) &&
      (item_ct1.get_local_linear_id() < 128)) {
/* DPCT_ORIG     tmp[cta.thread_rank()] = temp_sum = temp_sum +
 * tmp[cta.thread_rank() + 128];*/
    tmp[item_ct1.get_local_linear_id()] = temp_sum =
        temp_sum + tmp[item_ct1.get_local_linear_id() + 128];
  }

/* DPCT_ORIG   cg::sync(cta);*/
  /*
  DPCT1065:5: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

/* DPCT_ORIG   if ((blockDim.x >= 128) && (cta.thread_rank() < 64)) {*/
  if ((item_ct1.get_local_range(2) >= 128) &&
      (item_ct1.get_local_linear_id() < 64)) {
/* DPCT_ORIG     tmp[cta.thread_rank()] = temp_sum = temp_sum +
 * tmp[cta.thread_rank() + 64];*/
    tmp[item_ct1.get_local_linear_id()] = temp_sum =
        temp_sum + tmp[item_ct1.get_local_linear_id() + 64];
  }

/* DPCT_ORIG   cg::sync(cta);*/
  /*
  DPCT1065:6: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

/* DPCT_ORIG   if (cta.thread_rank() < 32) {*/
  if (item_ct1.get_local_linear_id() < 32) {
    // Fetch final intermediate sum from 2nd warp
/* DPCT_ORIG     if (blockDim.x >= 64) temp_sum += tmp[cta.thread_rank() +
 * 32];*/
    if (item_ct1.get_local_range(2) >= 64) temp_sum +=
        tmp[item_ct1.get_local_linear_id() + 32];
    // Reduce final warp using shuffle
/* DPCT_ORIG     for (int offset = tile32.size() / 2; offset > 0; offset /= 2)
 * {*/
    for (int offset = item_ct1.get_sub_group().get_local_linear_range() / 2;
         offset > 0; offset /= 2) {
/* DPCT_ORIG       temp_sum += tile32.shfl_down(temp_sum, offset);*/
      temp_sum +=
          sycl::shift_group_left(item_ct1.get_sub_group(), temp_sum, offset);
    }
  }
  // write result for this block to global mem
/* DPCT_ORIG   if (cta.thread_rank() == 0) result[0] = temp_sum;*/
  if (item_ct1.get_local_linear_id() == 0) result[0] = temp_sum;
}

void init_input(float *a, size_t size) {
  for (size_t i = 0; i < size; i++) a[i] = (rand() & 0xFF) / (float)RAND_MAX;
}

/* DPCT_ORIG void CUDART_CB myHostNodeCallback(void *data) {*/
void myHostNodeCallback(void *data) {
  // Check status of GPU after stream operations are done
  callBackData_t *tmp = (callBackData_t *)(data);
  // checkCudaErrors(tmp->status);

  double *result = (double *)(tmp->data);
  char *function = (char *)(tmp->fn_name);
  printf("[%s] Host callback final reduced sum = %lf\n", function, *result);
  *result = 0.0;  // reset the result
}

void syclGraphManual(float *inputVec_h, float *inputVec_d,
                                  double *outputVec_d, double *result_d,
                                  size_t inputSize, size_t numOfBlocks) {
                                      
  namespace sycl_ext = sycl::ext::oneapi::experimental;
  double result_h = 0.0;
  sycl::queue q = sycl::queue{sycl::gpu_selector_v}; //use default sycl queue, which is out of order
  if(!q.get_device().has(sycl::aspect::fp64)){
      printf("Double precision isn't supported : %s \nExit\n",
        q.get_device().get_info<sycl::info::device::name>().c_str());
      exit(0);
  }
  
  sycl_ext::command_graph graph(q.get_context(), q.get_device());  
  auto nodecpy = graph.add([&](sycl::handler& h){
      h.memcpy(inputVec_d, inputVec_h, sizeof(float) * inputSize);
  }); 
  
  auto nodememset1 = graph.add([&](sycl::handler& h){
      //h.memset(outputVec_d, 0, sizeof(double) * numOfBlocks);
      h.fill(outputVec_d, 0, numOfBlocks);
  });

  auto nodememset2 = graph.add([&](sycl::handler& h){
      //h.memset(result_d, 0, sizeof(double));
      h.fill(result_d, 0, 1);
  });

  auto nodek1 = graph.add([&](sycl::handler &cgh) {
    sycl::local_accessor<double, 1> tmp_acc_ct1(
      sycl::range<1>(THREADS_PER_BLOCK), cgh);

    cgh.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, numOfBlocks) *
                            sycl::range<3>(1, 1, THREADS_PER_BLOCK),
                        sycl::range<3>(1, 1, THREADS_PER_BLOCK)),
      [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
        reduce(inputVec_d, outputVec_d, inputSize, numOfBlocks, item_ct1,
               tmp_acc_ct1.get_pointer());
      });
  },  sycl_ext::property::node::depends_on(nodecpy, nodememset1));
   
  
  auto nodek2 = graph.add([&](sycl::handler &cgh) {
    sycl::local_accessor<double, 1> tmp_acc_ct1(
      sycl::range<1>(THREADS_PER_BLOCK), cgh);

    cgh.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, THREADS_PER_BLOCK),
                        sycl::range<3>(1, 1, THREADS_PER_BLOCK)),
      [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
        reduceFinal(outputVec_d, result_d, numOfBlocks, item_ct1,
                    tmp_acc_ct1.get_pointer());
      });
  }, sycl_ext::property::node::depends_on(nodek1, nodememset2));
  auto nodecpy1 = graph.add([&](sycl::handler &cgh) {
      cgh.memcpy(&result_h, result_d, sizeof(double));  
  }, sycl_ext::property::node::depends_on(nodek2));
  auto exec_graph = graph.finalize();
  
  sycl::queue qexec = sycl::queue{sycl::gpu_selector_v, 
      {sycl::ext::intel::property::queue::no_immediate_command_list()}};
  if(qexec.get_device().get_info<sycl_ext::info::device::graph_support>()
      == sycl_ext::info::graph_support_level::unsupported){
      printf("sycl graph not supported : %s\nExit\n", 
        qexec.get_device().get_info<sycl::info::device::name>().c_str());
      exit(0);
  }
  if(!qexec.get_device().has(sycl::aspect::fp64)){
     printf("Double precision isn't supported : %s \nExit\n",
        qexec.get_device().get_info<sycl::info::device::name>().c_str());
      exit(0);
  }
  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
    qexec.submit([&](sycl::handler& cgh) {
      cgh.ext_oneapi_graph(exec_graph);
    }).wait(); 
    printf("Final reduced sum = %lf\n", result_h);
  }
  
}

void syclGraphCaptureQueue(float *inputVec_h, float *inputVec_d,
                                  double *outputVec_d, double *result_d,
                                  size_t inputSize, size_t numOfBlocks) {
                                      
  namespace sycl_ext = sycl::ext::oneapi::experimental;
  double result_h = 0.0;
  sycl::queue q = sycl::queue{sycl::gpu_selector_v}; //use default sycl queue, which is out of order
  if(!q.get_device().has(sycl::aspect::fp64)){
      printf("Double precision isn't supported : %s \nExit\n",
        q.get_device().get_info<sycl::info::device::name>().c_str());
      exit(0);
  }

  sycl_ext::command_graph graph(q.get_context(), q.get_device());
  graph.begin_recording(q);
  
  sycl::event ememcpy = q.memcpy(inputVec_d, inputVec_h, sizeof(float) * inputSize);
  //sycl::event ememset = q.memset(outputVec_d, 0, sizeof(double) * numOfBlocks);
  sycl::event ememset = q.fill(outputVec_d, 0, numOfBlocks);
  //sycl::event ememset1 = q.memset(result_d, 0, sizeof(double));
  sycl::event ememset1 = q.fill(result_d, 0, 1);
  
  sycl::event ek1 = q.submit([&](sycl::handler &cgh) {
    cgh.depends_on({ememcpy, ememset});
    sycl::local_accessor<double, 1> tmp_acc_ct1(
      sycl::range<1>(THREADS_PER_BLOCK), cgh);

    cgh.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, numOfBlocks) *
                            sycl::range<3>(1, 1, THREADS_PER_BLOCK),
                        sycl::range<3>(1, 1, THREADS_PER_BLOCK)),
      [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
        reduce(inputVec_d, outputVec_d, inputSize, numOfBlocks, item_ct1,
               tmp_acc_ct1.get_pointer());
      });
  });
  
  sycl::event ek2 = q.submit([&](sycl::handler &cgh) {
    cgh.depends_on({ek1, ememset1});
    sycl::local_accessor<double, 1> tmp_acc_ct1(
      sycl::range<1>(THREADS_PER_BLOCK), cgh);

    cgh.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, THREADS_PER_BLOCK),
                        sycl::range<3>(1, 1, THREADS_PER_BLOCK)),
      [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
        reduceFinal(outputVec_d, result_d, numOfBlocks, item_ct1,
                    tmp_acc_ct1.get_pointer());
      });
  });
  
  q.submit([&](sycl::handler &cgh) {
      cgh.depends_on(ek2);
      cgh.memcpy(&result_h, result_d, sizeof(double));  
  });
  graph.end_recording();
  auto exec_graph = graph.finalize();
  
  
  sycl::queue qexec = sycl::queue{sycl::gpu_selector_v, 
      {sycl::ext::intel::property::queue::no_immediate_command_list()}};
  if(qexec.get_device().get_info<sycl_ext::info::device::graph_support>()
      == sycl_ext::info::graph_support_level::unsupported){
      printf("sycl graph not supported : %s\nExit\n", 
        qexec.get_device().get_info<sycl::info::device::name>().c_str());
      exit(0);
  }
  if(!qexec.get_device().has(sycl::aspect::fp64)){
      printf("Double precision isn't supported : %s \nExit\n",
        qexec.get_device().get_info<sycl::info::device::name>().c_str());
      exit(0);
  }
  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
    qexec.submit([&](sycl::handler& cgh) {
      cgh.ext_oneapi_graph(exec_graph);
    }).wait(); 
    printf("Final reduced sum = %lf\n", result_h);
  }
  
}

int main(int argc, char **argv) {
  size_t size = 1 << 24;  // number of elements to reduce
  size_t maxBlocks = 512;

  printf("%zu elements\n", size);
  printf("threads per block  = %d\n", THREADS_PER_BLOCK);
  printf("Graph Launch iterations = %d\n", GRAPH_LAUNCH_ITERATIONS);

  float *inputVec_d = NULL, *inputVec_h = NULL;
  double *outputVec_d = NULL, *result_d;  

/* DPCT_ORIG   checkCudaErrors(cudaMallocHost(&inputVec_h, sizeof(float) *
 * size));*/
  inputVec_h = sycl::malloc_host<float>(size, dpct::get_default_queue());
/* DPCT_ORIG   checkCudaErrors(cudaMalloc(&inputVec_d, sizeof(float) * size));*/
  inputVec_d = sycl::malloc_device<float>(size, dpct::get_default_queue());
/* DPCT_ORIG   checkCudaErrors(cudaMalloc(&outputVec_d, sizeof(double) *
 * maxBlocks));*/
  outputVec_d = sycl::malloc_device<double>(maxBlocks, dpct::get_default_queue());
/* DPCT_ORIG   checkCudaErrors(cudaMalloc(&result_d, sizeof(double)));*/
  result_d = sycl::malloc_device<double>(1, dpct::get_default_queue());

  init_input(inputVec_h, size);

  double tmp;
  for(size_t i=1;i<size;i++)
      tmp += inputVec_h[i];
  printf("CPU sum = %lf\n", tmp);
  
  printf("Using manually constructed SYCL graph ... \n");
  syclGraphManual(inputVec_h, inputVec_d, outputVec_d, result_d, size, maxBlocks);
  printf("Using SYCL queue capture on single queue ... \n");
  syclGraphCaptureQueue(inputVec_h, inputVec_d, outputVec_d, result_d, size, maxBlocks);
  
  
/* DPCT_ORIG   checkCudaErrors(cudaFree(inputVec_d));*/
  sycl::free(inputVec_d, dpct::get_default_queue());
/* DPCT_ORIG   checkCudaErrors(cudaFree(outputVec_d));*/
  sycl::free(outputVec_d, dpct::get_default_queue());
/* DPCT_ORIG   checkCudaErrors(cudaFree(result_d));*/
  sycl::free(result_d, dpct::get_default_queue());
/* DPCT_ORIG   checkCudaErrors(cudaFreeHost(inputVec_h));*/
  sycl::free(inputVec_h, dpct::get_default_queue());
  return EXIT_SUCCESS;
}
