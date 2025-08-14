#include <onnxruntime_cxx_api.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cassert>
#include <string>

int main()
{
    // 创建 ONNX Runtime 环境
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

    // 创建 SessionOptions 并启用 CUDA EP
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetInterOpNumThreads(1);

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
        "arena_extend_strategy"};
    const char *values[] = {
        "0",       // device 0
        "DEFAULT", // 或 HEURISTIC / EXHAUSTIVE
        "1",       // 在 default stream 做拷贝
        "0",       // 0 表示不限制（按需分配）
        "kNextPowerOfTwo"};

    Ort::ThrowOnError(api.UpdateCUDAProviderOptions(cuda_opts, keys, values,
                                                    static_cast<int>(std::size(keys))));
    Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_CUDA_V2(session_options, cuda_opts));
    // session_options.AppendExecutionProvider_CUDA_V2();

    // 创建 Session
    const char *model_path = "/root/projects/onnx-infer/models/model.onnx";
    Ort::Session session(env, model_path, session_options);

    // 验证输入输出个数
    size_t input_count = session.GetInputCount();
    size_t output_count = session.GetOutputCount();
    assert(input_count == 2);
    assert(output_count == 1);

    // 准备输入数据
    std::vector<float> feature_origin(256 * 200 * 61, 1.0f);
    std::vector<int64_t> length_origin(256, 200);

    const int64_t feature_shape[] = {256, 200, 61};
    const int64_t length_shape[] = {256};

    // 创建 pinned 内存（如果想要 CUDA 异步可改成 "CudaPinned"）
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    // 创建输入 tensor
    Ort::Value feature_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        feature_origin.data(),
        feature_origin.size(),
        feature_shape,
        3);

    Ort::Value length_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info,
        length_origin.data(),
        length_origin.size(),
        length_shape,
        1);

    assert(feature_tensor.IsTensor());
    assert(length_tensor.IsTensor());

    // 获取输入输出名称
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_0 = session.GetInputNameAllocated(0, allocator);
    auto input_name_1 = session.GetInputNameAllocated(1, allocator);
    auto output_name_0 = session.GetOutputNameAllocated(0, allocator);
    
    std::cout<< std::string(input_name_0.get()) << std::endl;
    std::cout<< std::string(input_name_1.get()) << std::endl;
    std::cout<< std::string(output_name_0.get()) << std::endl;

    // 运行推理
    std::array<const char *, 2> input_names = {input_name_0.get(), input_name_1.get()};
    std::array<const char *, 1> output_names = {output_name_0.get()};

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        std::array<Ort::Value, 2>{std::move(feature_tensor), std::move(length_tensor)}.data(),
        input_names.size(),
        output_names.data(),
        output_names.size());

    // 访问输出数据
    assert(output_tensors.size() == 1 && output_tensors[0].IsTensor());
    float *output_data = output_tensors[0].GetTensorMutableData<float>();

    std::cout << "Output[0]: " << output_data[0] << std::endl;

    return 0;
}
