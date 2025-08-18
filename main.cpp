#include <onnxruntime_cxx_api.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cassert>
#include <string>
#include <chrono>
#include <thread>

void thread_worker(Ort::Env &env, int tot_iterations)
{
    cudaSetDevice(3);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    Ort::SessionOptions session_options;
    session_options.SetLogSeverityLevel(1);
    // session_options.DisableCpuMemArena();
    session_options.SetIntraOpNumThreads(1);
    session_options.SetInterOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 配置 CUDA Provider
    // OrtCUDAProviderOptionsV2 cuda_options;
    const OrtApi &api = Ort::GetApi();
    OrtCUDAProviderOptionsV2 *cuda_opts = nullptr;
    Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda_opts));
    const char *keys[] = {
        "device_id",
        "cudnn_conv_algo_search",
        "do_copy_in_default_stream",
        "gpu_mem_limit",
        "arena_extend_strategy",
    };
    const char *values[] = {
        "3",       // device 0
        "DEFAULT", // 或 HEURISTIC / EXHAUSTIVE
        "0",       // 在 default stream 做拷贝
        "0",       // 0 表示不限制（按需分配）
        "kNextPowerOfTwo",
    };

    Ort::ThrowOnError(api.UpdateCUDAProviderOptions(cuda_opts, keys, values,
                                                    static_cast<int>(std::size(keys))));
    Ort::ThrowOnError(api.UpdateCUDAProviderOptionsWithValue(cuda_opts, "user_compute_stream", stream));

    // api.UpdateCUDAProviderOptionsWithValue(cuda_opts, "disable_fallback ", "1");

    Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_CUDA_V2(session_options, cuda_opts));
    // session_options.AppendExecutionProvider_CUDA_V2();

    // 创建 Session
    const char *model_path = "/root/projects/onnx-infer/models/model.onnx";
    Ort::Session session(env, model_path, session_options);

    float *feature_origin;
    int64_t *length_origin;
    float *output;
    cudaHostAlloc((void **)&feature_origin, sizeof(float) * 256 * 200 * 61, cudaHostAllocDefault);
    cudaHostAlloc((void **)&length_origin, sizeof(int64_t) * 256, cudaHostAllocDefault);
    cudaHostAlloc((void **)&output, sizeof(float) * 256 * 200 * 5, cudaHostAllocDefault);
    std::fill(feature_origin, feature_origin + (256 * 200 * 61), 1.0f);
    std::fill(length_origin, length_origin + 256, 200);

    const int64_t feature_shape[] = {256, 200, 61};
    const int64_t length_shape[] = {256};
    const int64_t output_shape[] = {256, 200, 5};

    Ort::MemoryInfo mem_info_cpu_pinned("CudaPinned", OrtDeviceAllocator, 0, OrtMemTypeCPUOutput);

    Ort::IoBinding binding(session);
    Ort::Value feature_tensor = Ort::Value::CreateTensor<float>(
        mem_info_cpu_pinned,
        feature_origin,
        256 * 200 * 61,
        feature_shape,
        3);

    Ort::Value length_tensor = Ort::Value::CreateTensor<int64_t>(
        mem_info_cpu_pinned,
        length_origin,
        256,
        length_shape,
        1);

    Ort::Value probs_tensor = Ort::Value::CreateTensor<float>(
        mem_info_cpu_pinned,
        output,
        256 * 200 * 5,
        output_shape,
        3);
    binding.BindInput("feature", feature_tensor);
    binding.BindInput("length", length_tensor);
    binding.BindOutput("probs", probs_tensor);
    Ort::RunOptions run_options;
    run_options.AddConfigEntry("disable_synchronize_execution_providers", "1");
    float result = 0;
    // warm up
    for (int i = 0; i < 10; i++)
    {
        session.Run(run_options, binding);
        cudaStreamSynchronize(stream);
    }

    result = 0;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < tot_iterations; i++)
    {
        session.Run(run_options, binding);
        cudaStreamSynchronize(stream);
        result += output[0];
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "thread_worker: tot_sum: " << result << " elapsed: " << elapsed.count() << std::endl;
}

// bind 的是 GPU 的内存
void thread_worker_v2(Ort::Env &env, int tot_iterations)
{
    int device_id = 3;

    cudaSetDevice(device_id);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    Ort::SessionOptions session_options;
    session_options.SetLogSeverityLevel(1);
    // session_options.DisableCpuMemArena();
    // session_options.SetExecutionMode(ORT_SEQUENTIAL);
    session_options.DisableCpuMemArena();
    session_options.SetIntraOpNumThreads(1);
    session_options.SetInterOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 配置 CUDA Provider
    // OrtCUDAProviderOptionsV2 cuda_options;
    const OrtApi &api = Ort::GetApi();
    OrtCUDAProviderOptionsV2 *cuda_opts = nullptr;
    Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda_opts));
    const char *keys[] = {
        "device_id",
        "cudnn_conv_algo_search",
        "do_copy_in_default_stream",
        "gpu_mem_limit",
        "arena_extend_strategy",
    };
    const char *values[] = {
        "3",       // device 0
        "DEFAULT", // 或 HEURISTIC / EXHAUSTIVE
        "0",       // 在 default stream 做拷贝
        "0",       // 0 表示不限制（按需分配）
        "kNextPowerOfTwo",
    };

    Ort::ThrowOnError(api.UpdateCUDAProviderOptions(cuda_opts, keys, values,
                                                    static_cast<int>(std::size(keys))));
    Ort::ThrowOnError(api.UpdateCUDAProviderOptionsWithValue(cuda_opts, "user_compute_stream", stream));

    // api.UpdateCUDAProviderOptionsWithValue(cuda_opts, "disable_fallback ", "1");

    Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_CUDA_V2(session_options, cuda_opts));
    // session_options.AppendExecutionProvider_CUDA_V2();

    session_options.DisableMemPattern();
    // session_options.DisablePerSessionThreads();
    // 创建 Session
    const char *model_path = "/root/projects/onnx-infer/models/model.onnx";
    Ort::Session session(env, model_path, session_options);

    float *feature_origin;
    int64_t *length_origin;
    float *output;
    int batch = 256;
    int timestep = 200;
    int feat_size = 61;
    int num_classes = 5;
    cudaHostAlloc((void **)&feature_origin, sizeof(float) * batch * timestep * feat_size, cudaHostAllocDefault);
    cudaHostAlloc((void **)&length_origin, sizeof(int64_t) * batch, cudaHostAllocDefault);
    cudaHostAlloc((void **)&output, sizeof(float) * batch * timestep * num_classes, cudaHostAllocDefault);
    std::fill(feature_origin, feature_origin + (256 * 200 * 61), 1.0f);
    std::fill(length_origin, length_origin + 256, 200);

    const int64_t feature_shape[] = {256, 200, 61};
    const int64_t length_shape[] = {256};
    const int64_t output_shape[] = {256, 200, 5};

    float *feature_cuda;
    int64_t *length_cuda;
    float *output_cuda;

    cudaMalloc(&feature_cuda, sizeof(float) * batch * timestep * feat_size);
    cudaMalloc(&length_cuda, sizeof(int64_t) * batch);
    cudaMalloc(&output_cuda, sizeof(float) * batch * timestep * num_classes);

    Ort::IoBinding binding(session);

    Ort::MemoryInfo memory_info_cuda("Cuda", OrtArenaAllocator, device_id,
                                     OrtMemTypeDefault);

    Ort::Value feature_tensor = Ort::Value::CreateTensor<float>(
        memory_info_cuda,
        feature_cuda,
        256 * 200 * 61,
        feature_shape,
        3);

    Ort::Value length_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info_cuda,
        length_cuda,
        256,
        length_shape,
        1);

    Ort::Value probs_tensor = Ort::Value::CreateTensor<float>(
        memory_info_cuda,
        output_cuda,
        256 * 200 * 5,
        output_shape,
        3);
    binding.BindInput("feature", feature_tensor);
    binding.BindInput("length", length_tensor);
    binding.BindOutput("probs", probs_tensor);
    Ort::RunOptions run_options;
    run_options.AddConfigEntry("disable_synchronize_execution_providers", "1");
    float result = 0;
    // warm up
    for (int i = 0; i < 10; i++)
    {
        cudaMemcpyAsync(feature_cuda, feature_origin, sizeof(float) * batch * timestep * feat_size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(length_cuda, length_origin, sizeof(int64_t) * batch, cudaMemcpyHostToDevice, stream);
        session.Run(run_options, binding);
        cudaMemcpyAsync(output, output_cuda, sizeof(float) * batch * timestep * num_classes, cudaMemcpyDeviceToHost, stream);

        cudaStreamSynchronize(stream);
    }

    result = 0;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < tot_iterations; i++)
    {
        cudaMemcpyAsync(feature_cuda, feature_origin, sizeof(float) * batch * timestep * feat_size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(length_cuda, length_origin, sizeof(int64_t) * batch, cudaMemcpyHostToDevice, stream);
        session.Run(run_options, binding);
        cudaMemcpyAsync(output, output_cuda, sizeof(float) * batch * timestep * num_classes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        result += output[0];
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "thread_worker: tot_sum: " << result << " elapsed: " << elapsed.count() << std::endl;
}

void multi_steam_multi_thread(Ort::Env &env, int tot_iter, int num_threads)
{

    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; i++)
    {
        threads.push_back(std::thread(thread_worker_v2, std::ref(env), tot_iter / num_threads));
    }

    for (int i = 0; i < num_threads; i++)
    {
        threads[i].join();
    }
}

int main()
{
    // 创建 ONNX Runtime 环境
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

    // Ort::GetApi().CreateEnvWithGlobalThreadPools()

    int tot_iterations = 300;
    int num_threads = 2;
    multi_steam_multi_thread(env, tot_iterations, num_threads);
}
