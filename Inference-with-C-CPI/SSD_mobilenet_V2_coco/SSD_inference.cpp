#include <iostream>
#include <chrono>
#include <fstream>
#include <NvInfer.h>
#include <memory>
#include <NvOnnxParser.h>
#include <vector>
#include <cuda_runtime_api.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <algorithm>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <NvInferPlugin.h>

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

void postprocessResults(cv::Mat &frame,float *gpu_output, const nvinfer1::Dims &dims, int batch_size, int output_layout=7, int conf_th=0.9)
{
    // get class names
    auto classes = getClassNames("/home/thienpn/Desktop/project/SSD_mobilenet_V2_coco/label.txt");

    // copy results from GPU to CPU
    std::vector<float> cpu_output(getSizeByDim(dims) * batch_size);
    cudaMemcpy(cpu_output.data(), gpu_output, cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    int img_w = frame.size().width;
    int img_h = frame.size().width;

    for (size_t i = 0; i < cpu_output.size(); i+=output_layout)
    {
        float conf = float(cpu_output[i+2]);
        if (conf < conf_th) 
            continue;
        int x1 = int(cpu_output[i+3] * img_w);
        int y1 = int(cpu_output[i+4] * img_h);
        int x2 = int(cpu_output[i+5] * img_w);
        int y2 = int(cpu_output[i+6] * img_h);
        int cls = int(cpu_output[i+1]);
        if ((x1 <=0) || (y1<=0) || (x2<=0) || (y2<=0) || (cls<=0) )
            continue;
        // std::cout << " class :"<< classes[cls] << std::endl;
        cv::rectangle(frame,cv::Point(x1,y1),cv::Point(x2,y2),cv::Scalar(0,255,0),3);
        cv::putText(frame,classes[cls],cv::Point(x1,y1),cv::FONT_HERSHEY_DUPLEX,1,cv::Scalar(255,255,0),1,false);
    }
}

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        std::cerr << "usage: " << argv[0] << " model.onnx video.mp4\n";
        return -1;
    }
    std::string model_path(argv[1]);
    std::string video_path(argv[2]);
    int batch_size = 1;
    //Deserialize Engine
    initLibNvInferPlugins(&gLogger, "");

    std::ifstream planFile(model_path);
    std::stringstream planBuffer;
    planBuffer << planFile.rdbuf();
    std::string plan = planBuffer.str();

    assert(plan.length() != 0);


    TRTUniquePtr<nvinfer1::IRuntime> runtime {nullptr};
    TRTUniquePtr<nvinfer1::ICudaEngine> engine{nullptr};
    TRTUniquePtr<nvinfer1::IExecutionContext> context{nullptr};

    runtime.reset(nvinfer1::createInferRuntime(gLogger));
    engine.reset(runtime->deserializeCudaEngine((void*) plan.data(), plan.size(), nullptr));
    assert(engine != nullptr); 
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
    cv::VideoCapture cap(video_path);

    cv::VideoWriter writer;
    int codec = cv::VideoWriter::fourcc('a', 'v', 'c', '1');  
    double fps = 24.0;
    cv::Size sizeFrame(1280,720);
    writer.open("demo.mp4", codec, fps, sizeFrame,true);
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
        auto startTime = std::chrono::high_resolution_clock::now();
        context->enqueue(batch_size, buffers.data(), 0, nullptr);
        auto endTime = std::chrono::high_resolution_clock::now();
        float totalTime = std::chrono::duration<float, std::milli>(endTime - startTime).count()/1000;
        // postprocess results
        postprocessResults(frame,(float *) buffers[1], output_dims[0], batch_size);
        cv::putText(frame,"Processing Time :"+std::to_string(totalTime)+"s",cv::Point(11,20),cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(0,255,0),1,cv::LINE_AA,false);
        cv::imshow("My first result", frame);
        writer.write(frame);
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

    cap.release();
    writer.release();
    return 0;
}