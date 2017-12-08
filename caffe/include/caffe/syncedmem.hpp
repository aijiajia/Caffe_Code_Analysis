#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>

#include "caffe/common.hpp"

namespace caffe {
// 如果实在ＧＰＵ模式，且ＣＵＤＡ使能，那么主机内存会以页锁定内存方式分配（使用cudaMallocHost()函数。对于单ＧＰＵ的性能提升不明显，但多ＧＰＵ会非常明显。

// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.
inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda) {
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaMallocHost(ptr, size));
    *use_cuda = true;
    return;
  }
#endif
  *ptr = malloc(size);
  *use_cuda = false;
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

inline void CaffeFreeHost(void* ptr, bool use_cuda) {
#ifndef CPU_ONLY
  if (use_cuda) {
    CUDA_CHECK(cudaFreeHost(ptr));
    return;
  }
#endif
  free(ptr);
}


/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *
 * TODO(dox): more thorough description.
 */
// 该类负责存储分配及主机和设备间同步。
class SyncedMemory {
 public:
 //构造函数
  SyncedMemory()
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {}
// 显示构造函数
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {}
//析构函数
  ~SyncedMemory();
  const void* cpu_data(); //只读取cpu data
  void set_cpu_data(void* data); // 设置cpu data
  const void* gpu_data(); //只读取gpu data
  void set_gpu_data(void* data); //设置gpu data
  void* mutable_cpu_data(); //读写获取cpu data
  void* mutable_gpu_data(); //读写获取gpu data
//  状态机变量，表示4种状态：未初始化、cpu有数据、gpu有效、已同步。
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
//  获得当前状态机变量。
  SyncedHead head() { return head_; }
//  获得当前内存空间尺寸
  size_t size() { return size_; }

#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif

 private:
  void to_cpu(); //数据由显存同步到cpu
  void to_gpu(); //数据由内存同步到显存
  void* cpu_ptr_;//内存指针
  void* gpu_ptr_;//显存指针
  size_t size_;  //数据大小
  SyncedHead head_;//当前数据状态，UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED
  bool own_cpu_data_;
  bool cpu_malloc_use_cuda_;
  bool own_gpu_data_;
  int gpu_device_;

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
