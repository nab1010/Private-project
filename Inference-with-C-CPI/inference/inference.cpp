#include <iostream>
#include <fstream>
#include <NvInfer.h>
#include <memory>
#include <NvOnnxParser.h>
#include <vector>
#include <cuda_runtime_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <algorithm>
#include <numeric>
#include <opencv2/opencv.hpp>

class Logger : public nvinfer1::ILogger           
 {
     void log(Severity severity, const char* msg) noexcept override
     {
         // suppress info-level messages
         if (severity != Severity::kINFO)
             std::cout << msg << std::endl;
     }
 } gLogger;

struct TRTDestroy
{
    template <class T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};


template <class T>
using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;

// calculate size of tensor
size_t getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}

std::vector<std::string> getClassNames(const std::string& imagenet_classes)
{
    std::ifstream classes_file(imagenet_classes);
    std::vector<std::string> classes;
    if (!classes_file.good())
    {
        std::cerr << "ERROR: can't read file with classes names.\n";
        return classes;
    }
    std::string class_name;
    while (std::getline(classes_file, class_name))
    {
        classes.push_back(class_name);
    }
    return classes;
}

void preprocessImage(const cv::Mat frame, float* gpu_input, const nvinfer1::Dims& dims)
{
    // read input image
    // cv::Mat frame = cv::imread(image_path);
    if (frame.empty())
    {
        // std::cerr << "Input image " << image_path << " load failed\n";
        return;
    }
    cv::cuda::GpuMat gpu_frame;
    // upload image to GPU
    gpu_frame.upload(frame);

    auto input_width = dims.d[2];
    auto input_height = dims.d[1];
    auto channels = dims.d[0];
    auto input_size = cv::Size(input_width, input_height);
    // resize
    cv::cuda::GpuMat resized;
    cv::cuda::resize(gpu_frame, resized, input_size, 0, 0, cv::INTER_NEAREST);
    // normalize
    cv::cuda::GpuMat flt_image;
    resized.convertTo(flt_image, CV_32FC3, 1.f / 255.f);
    cv::cuda::subtract(flt_image, cv::Scalar(0.485f, 0.456f, 0.406f), flt_image, cv::noArray(), -1);
    cv::cuda::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image, 1, -1);
    // to tensor
    std::vector<cv::cuda::GpuMat> chw;
    for (size_t i = 0; i < channels; ++i)
    {
        chw.emplace_back(cv::cuda::GpuMat(input_size, CV_32FC1, gpu_input + i * input_width * input_height));
    }
    cv::cuda::split(flt_image, chw);
}

std::string postprocessResults(float *gpu_output, const nvinfer1::Dims &dims, int batch_size)
{
    // get class names
    auto classes = getClassNames("imagenet_classes.txt");

    // copy results from GPU to CPU
    std::vector<float> cpu_output(getSizeByDim(dims) * batch_size);
    cudaMemcpy(cpu_output.data(), gpu_output, cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // calculate softmax
    std::transform(cpu_output.begin(), cpu_output.end(), cpu_output.begin(), [](float val) {return std::exp(val);});
    auto sum = std::accumulate(cpu_output.begin(), cpu_output.end(), 0.0);
    // find top classes predicted by the model
    std::vector<int> indices(getSizeByDim(dims) * batch_size);
    std::iota(indices.begin(), indices.end(), 0); // generate sequence 0, 1, 2, 3, ..., 999
    std::sort(indices.begin(), indices.end(), [&cpu_output](int i1, int i2) {return cpu_output[i1] > cpu_output[i2];});
    // print results
    int i = 0;
    while (cpu_output[indices[i]] / sum > 0.005)
    {
        if (classes.size() > indices[i])
        {
            std::cout << "class: " << classes[indices[i]] << " | ";
        }
        std::cout << "confidence: " << 100 * cpu_output[indices[i]] / sum << "% | index: " << indices[i] << "\n";
        ++i;
        return classes[indices[i]];
    }
}

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        std::cerr << "usage: " << argv[0] << " model.onnx image.jpg\n";
        return -1;
    }
    std::string model_path(argv[1]);
    std::string image_path(argv[2]);
    int batch_size = 1;

    // initialize TensorRT engine and parse ONNX model
    // TRTUniquePtr<nvinfer1::ICudaEngine> engine{nullptr};
    // TRTUniquePtr<nvinfer1::IExecutionContext> context{nullptr};
    // parseOnnxModel(model_path, engine, context);
    
    // TRTUniquePtr<nvinfer1::ICudaEngine> engine{nullptr};
    // TRTUniquePtr<nvinfer1::IExecutionContext> context{nullptr};


    //Deserialize Engine
    std::ifstream planFile("resnet50.engine");
    std::stringstream planBuffer;
    planBuffer << planFile.rdbuf();
    std::string plan = planBuffer.str();

    TRTUniquePtr<nvinfer1::IRuntime> runtime {nullptr};
    TRTUniquePtr<nvinfer1::ICudaEngine> engine{nullptr};
    TRTUniquePtr<nvinfer1::IExecutionContext> context{nullptr};

    runtime.reset(nvinfer1::createInferRuntime(gLogger));
    engine.reset(runtime->deserializeCudaEngine((void*) plan.data(), plan.size(), nullptr));
    context.reset(engine->createExecutionContext());

    // get sizes of input and output and allocate memory required for input data and for output data
    std::vector<nvinfer1::Dims> input_dims; // we expect only one input
    std::vector<nvinfer1::Dims> output_dims; // and one output
    std::vector<void*> buffers(engine->getNbBindings()); // buffers for input and output data
    for (size_t i = 0; i < engine->getNbBindings(); ++i)
    {
        auto binding_size = getSizeByDim(engine->getBindingDimensions(i)) * batch_size * sizeof(float);
        cudaMalloc(&buffers[i], binding_size);
        if (engine->bindingIsInput(i))
        {
            input_dims.emplace_back(engine->getBindingDimensions(i));
        }
        else
        {
            output_dims.emplace_back(engine->getBindingDimensions(i));
        }
    }
    if (input_dims.empty() || output_dims.empty())
    {
        std::cerr << "Expect at least one input and one output for network\n";
        return -1;
    }
    cv::VideoCapture cap(0);

    cv::namedWindow("My first result", cv::WINDOW_NORMAL);
    while (cap.isOpened())
    {
        cv::Mat frame;
        bool bSuccess = cap.read(frame);
          if (bSuccess == false) 
        {
        std::cout << "Found the end of the video" << std::endl;
        break;
        }
            // preprocess input data
        preprocessImage(frame, (float *) buffers[0], input_dims[0]);
        // inference
        context->enqueue(batch_size, buffers.data(), 0, nullptr);
        // postprocess results
        // postprocessResults((float *) buffers[1], output_dims[0], batch_size);

        cv::putText(frame,postprocessResults((float *) buffers[1], output_dims[0], batch_size),cv::Point(50,50),cv::FONT_HERSHEY_DUPLEX,1,cv::Scalar(0,255,0),2,false);
        cv::imshow("My first result", frame);
        if (cv::waitKey(10) == 27)
        {
        std::cout << "Esc key is pressed by user. Stoppig the video" << std::endl;
        break;
        }

    }
    for (void* buf : buffers)
    {
        cudaFree(buf);
    }

    return 0;
}