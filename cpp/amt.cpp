#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <cmath>
#include <chrono>
#include <vector>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <string>
#include <vector>
#include <algorithm>
#include <eigen3/Eigen/Core>
#include <boost/filesystem.hpp> // c++17 std support
#include "MidiFile.h"
#include "cnpy.h"
#include "librosa.h"
#include<opencv2/opencv.hpp>
using namespace nvinfer1;

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

#define DEVICE 0  // GPU id

class AMT{
private:
    // stuff we know about the network and the input/output blobs
    static const int INPUT_W;
    static const int INPUT_H;
    static const int OUTPUT_W;
    static const int OUTPUT_H;

    // mel 
    static const int SR;
    static const int N_HOP;

    static const std::string INPUT_BLOB_NAME;
    static const std::vector<std::string> OUTPUT_BLOB_NAMES;
    static Logger gLogger;

    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;
    cudaStream_t stream;
    void* buffers[6];

    typedef std::vector<std::vector<float>> OUTPUT;
    typedef unsigned int uint8_t;
    typedef Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor> Vectorf;
    typedef Eigen::Matrix<uint8_t, 1, Eigen::Dynamic, Eigen::RowMajor> Vectoruint8;
    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrixf;
    

public:
    AMT(const std::string& engine_file_path, int batch_size){
        std::ifstream file(engine_file_path, std::ios::binary);
        size_t size=0;
        char* trtModelStream = nullptr;
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
        else{
            std::cout << "file is not good" << std::endl;
            exit(1);
        }

        runtime = createInferRuntime(gLogger);
        assert(runtime != nullptr);
        // 反序列化构建推理的引擎
        engine = runtime->deserializeCudaEngine(trtModelStream, size);
        assert(engine != nullptr); 
        delete[] trtModelStream;

        context = engine->createExecutionContext();
        assert(context != nullptr);

        // 分配显存
        assert(engine->getNbBindings() == 6);

        const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME.c_str());
        context->setBindingDimensions(inputIndex, Dims3(batch_size, INPUT_H, INPUT_W));
        assert(context->allInputDimensionsSpecified());

        assert(engine->getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
        CHECK(cudaMalloc(&buffers[inputIndex], batch_size*INPUT_H*INPUT_W*sizeof(float)));

        for(auto OUTPUT_BLOB_NAME: OUTPUT_BLOB_NAMES){
            const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME.c_str());
            assert(engine->getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);
            
            CHECK(cudaMalloc(&buffers[outputIndex], batch_size*OUTPUT_H*OUTPUT_W*sizeof(float)));
        }

        CHECK(cudaStreamCreate(&stream));
        
    }
    ~AMT(){
        cudaStreamDestroy(stream);
        for(size_t Index=0; Index<1+OUTPUT_BLOB_NAMES.size(); ++Index){
            CHECK(cudaFree(buffers[Index]));
        }
        // context->destroy(); // 同下
        // engine->destroy(); // 同下
        // runtime->destroy(); // 指针内容被管理 在10.0中会被移除，来避免出现双次释放内存错误
    }

    void doInference(const float* input, float** const outputs, int infer_size) {
        // int mBatchSize = engine.getMaxBatchSize();        
        // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host        
        const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME.c_str());

        CHECK(cudaMemcpyAsync(buffers[inputIndex], input, infer_size*INPUT_H*INPUT_W*sizeof(float), cudaMemcpyHostToDevice, stream));
        // enqueue是异步推断 execute是同步推断 第一个参数是batch_size

        context->enqueue(infer_size, buffers, stream, nullptr);

        // 将数据从显存拷贝到内存
        int i=0;
        for(auto OUTPUT_BLOB_NAME: OUTPUT_BLOB_NAMES){
            const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME.c_str());
            CHECK(cudaMemcpyAsync(outputs[i++], buffers[outputIndex], infer_size*OUTPUT_H*OUTPUT_W*sizeof(float), cudaMemcpyDeviceToHost, stream));
        }
        cudaStreamSynchronize(stream);
    }

    void transcribe(const std::string& path, int sequence_length=512){
        auto mels = melspec(path); // Tx229
        int inner_batch_size = 16;
        int length = mels.size(), n_mels = mels[0].size();

        int batches = std::ceil(float(length)/float(sequence_length));
        int paddings = batches*sequence_length - length;
        for(int i=0;i<paddings;++i) mels.push_back(std::vector<float>(n_mels, 0));
        int loop = std::ceil(float(batches) / float(inner_batch_size));

        int output_nodes = OUTPUT_BLOB_NAMES.size();
        std::vector<OUTPUT>results(output_nodes, OUTPUT(length+paddings, std::vector<float>(88)));

        int mem_size = batches;
        if(loop>1) mem_size = inner_batch_size;

        float* input = new float[mem_size*INPUT_H*INPUT_W];
        float** outputs = new float*[output_nodes];
        for(int i=0;i<output_nodes;++i) {
            outputs[i] = new float[mem_size*OUTPUT_H*OUTPUT_W];
        }

        for(int i=0; i<loop; ++i){
            int left_batches = batches - i*inner_batch_size;
            left_batches = std::min(left_batches, inner_batch_size);
            for(int j = i*inner_batch_size*sequence_length; j < (i*inner_batch_size + left_batches)*sequence_length; ++j){
                const auto & sample = mels[j];
                int offset = j - i*inner_batch_size*sequence_length;
                std::copy(sample.begin(), sample.end(), input+offset*n_mels);
            }
            doInference(input, outputs, left_batches);
            for(int k=0;k<output_nodes;++k){
                for(int j = i*inner_batch_size*sequence_length; j < (i*inner_batch_size + left_batches)*sequence_length; ++j){
                    auto & result = results[k][j];
                    int offset = j - i*inner_batch_size*sequence_length;
                    std::copy(outputs[k]+offset*88, outputs[k]+(offset+1)*88, result.begin());
                }
            }
        }

        delete []input;
        for(size_t i=0; i<OUTPUT_BLOB_NAMES.size(); ++i) delete []outputs[i];
        delete []outputs;

        // check_probs(results);

        std::vector<int>p_est;
        std::vector<int>v_est;
        std::vector<std::pair<float, float>>i_est;

        extrace_note(results, i_est, p_est, v_est);
        // show_notes(i_est, p_est, v_est);

        // 结果保存为midi
        // boost::filesystem::path midipath(path);
        // midipath.replace_extension(".mid");
        // std::cout << i_est.size() << " " << p_est.size() << " " << v_est.size()<< std::endl;
        std::string midipath(path);
        midipath.replace(midipath.find(".wav"), 4, ".mid");
        savemidi(midipath, i_est, p_est, v_est);
        
        // 结果保存为图片
        // boost::filesystem::path imagepath = path;
        // imagepath.replace_extension(".png");
        // saveimage(imagepath.string(), results[0], results[3]);
        std::string imagepath(path);
        imagepath.replace(imagepath.find(".wav"), 4, ".png"); 
        saveimage(imagepath, results[0], results[3]);
    }

    void saveimage(const std::string&imgpath,
                   const OUTPUT&onset,
                   const OUTPUT&frame,
                   float onset_threshold=0.5,
                   float frame_threshold=0.5){
        // 概率矩阵变成eigen
        int length = onset.size();
        Eigen::MatrixXf onsets(length, 88);
        Eigen::MatrixXf frames(length, 88);
        for(int i=0; i<length; ++i) onsets.row(i) = Eigen::VectorXf::Map(&onset[i][0], 88);
        for(int i=0; i<length; ++i) frames.row(i) = Eigen::VectorXf::Map(&frame[i][0], 88);
        onsets.transposeInPlace(); // 88xT
        frames.transposeInPlace();
        auto onsets_color = 1 - (onsets.array()>onset_threshold).cast<uint8_t>();
        auto frames_color = 1 - (frames.array()>frame_threshold).cast<uint8_t>();
        auto both_color = 1- (1-onsets_color)*(1-frames_color);

        std::vector<int>sizes{88, length};
        cv::Mat image(sizes, CV_8UC3);
        for(int i=0; i<88; ++i){
            for(int j=0; j<length; ++j){
                image.at<cv::Vec3b>(87-i, j)[0] = onsets_color(i, j)*255;
                image.at<cv::Vec3b>(87-i, j)[1] = frames_color(i, j)*255;
                image.at<cv::Vec3b>(87-i, j)[2] = both_color(i, j)*255;
            }
        }
        //  保存image
        cv::Mat resized_image;// = cv::Mat::zeros(image.size(), image.type());;
        cv::resize(image, resized_image, cv::Size(0, 0), 8, 8);
        cv::imwrite(imgpath, resized_image);
    }

    void savemidi(const std::string&midipath,
                  const std::vector<std::pair<float, float>>&i_est,
                  const std::vector<int>&p_est,
                  const std::vector<int>&v_est){

        int tpq = 120;
        double defaultTempo = 120.0;
        double secondsPerTick = 60.0 / (defaultTempo * tpq);
        int starttick, endtick, key, velocity;

        smf::MidiFile midifile;
        int track   = 0;
        int channel = 0;
        int instr   = 0;
        midifile.setTPQ(tpq);
        midifile.addTimbre(track, 0, channel, instr);
        for(size_t index=0; index<i_est.size(); ++index){
            starttick = int(i_est[index].first/secondsPerTick);
            endtick   = int(i_est[index].second/secondsPerTick);
            key       = p_est[index];
            velocity  = v_est[index];
            midifile.addNoteOn (track, starttick, channel, key, velocity);
            midifile.addNoteOff(track, endtick,   channel, key);
        }
        midifile.sortTracks();
        midifile.write(midipath);
    }

    void show_notes(const std::vector<std::pair<float, float>>&i_est,
                    const std::vector<int>&p_est,
                    const std::vector<int>&v_est){
        std::cout << i_est.size() << std::endl;
        std::vector<size_t>indexs(i_est.size());
        for(size_t i=0;i<i_est.size();++i) indexs[i]=i;
        auto cmp = [i_est](size_t x, size_t y){
            return i_est[x].first < i_est[y].first;
        };
        std::sort(indexs.begin(), indexs.end(), cmp);
        std::cout << "onset\toffset\tmidi\tvelocity" << std::endl;
        for(size_t i=0;i<indexs.size();++i){
            int index = indexs[i];
            std::cout << i_est[index].first << "\t" << i_est[index].second << "\t"
                      << p_est[index] << "\t" << v_est[index] << std::endl << std::endl;
        }
    }

    void check_probs(const std::vector<OUTPUT>&results){
        float maxval1 = 0, sum1=0, tmp1;
        float maxval2 = 0, sum2=0, tmp2;
        float maxval3 = 0, sum3=0, tmp3;
        float maxval4 = 0, sum4=0, tmp4;
        float maxval5 = 0, sum5=0, tmp5;
        std::cout << "the result shape: " << results[0].size() << std::endl;

        for(size_t i=0;i<results[0].size();++i){
            tmp1 = *std::max_element(results[0][i].begin(), results[0][i].end());
            maxval1 = std::max(maxval1, tmp1);
            sum1 += tmp1;

            tmp2 = *std::max_element(results[1][i].begin(), results[1][i].end());
            maxval2 = std::max(maxval2, tmp2);
            sum2 += tmp2;

            tmp3 = *std::max_element(results[2][i].begin(), results[2][i].end());
            maxval3 = std::max(maxval3, tmp3);
            sum3 += tmp3;

            tmp4 = *std::max_element(results[3][i].begin(), results[3][i].end());
            maxval4 = std::max(maxval4, tmp4);
            sum4 += tmp4;

            tmp5 = *std::max_element(results[4][i].begin(), results[4][i].end());
            maxval5 = std::max(maxval5, tmp5);
            sum5 += tmp5;
        }
        std::cout << "max onset prob: " << maxval1 << std::endl;
        std::cout << "max offset prob: " << maxval2 << std::endl;
        std::cout << "max activation prob: " << maxval3 << std::endl;
        std::cout << "max frame prob: " << maxval4 << std::endl;
        std::cout << "max velocity prob: " << maxval5 << std::endl;

        std::cout << "max onset sum: " << sum1/results[0].size() << std::endl;
        std::cout << "max offset sum: " << sum2/results[0].size() << std::endl;
        std::cout << "max activation sum: " << sum3/results[0].size() << std::endl;
        std::cout << "max frame sum: " << sum4/results[0].size() << std::endl;
        std::cout << "max velocity sum: " << sum5/results[0].size() << std::endl;

        std::vector<float>t;
        for(size_t i=0; i<5; ++i){
            for(size_t j=0;j<results[i].size();++j){
                for(size_t k=0;k<results[i][j].size();++k){
                    t.push_back(results[i][j][k]);
                }
            }
        }
        cnpy::npy_save("cpp_prob.npy",&t[0],{5,results[0].size(),88},"w");
    }

    void extrace_note(std::vector<OUTPUT>& results, std::vector<std::pair<float, float>>& i_est, std::vector<int>& p_est, 
                      std::vector<int>& v_est, float onset_threshold=0.5, float frame_threshold=0.5){
        // results分别代表 "onset_pred", "offset_pred", "activation_pred", "frame_pred", "velocity_pred"
        int length = results[0].size();
        Eigen::MatrixXf onsets(length, 88);
        Eigen::MatrixXf frames(length, 88);
        Eigen::MatrixXf velocity(length, 88);
        
        for(int i=0; i<length; ++i) onsets.row(i) = Eigen::VectorXf::Map(&results[0][i][0], 88);
        for(int i=0; i<length; ++i) frames.row(i) = Eigen::VectorXf::Map(&results[3][i][0], 88);
        for(int i=0; i<length; ++i) velocity.row(i) = Eigen::VectorXf::Map(&results[4][i][0], 88);

        auto onsets_value = (onsets.array()>onset_threshold).cast<uint8_t>();
        auto frames_value = (frames.array()>frame_threshold).cast<uint8_t>();
        
        Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>onsets_diff_value(length, 88);
        onsets_diff_value << onsets_value.row(0),
                             onsets_value.bottomRows(length-1) - onsets_value.topRows(length-1);
        auto onsets_diff = (onsets_diff_value.array() == 1); // 不能省略
        
        int onset, offset;
        for(int pitch=0; pitch<onsets_diff.cols(); ++pitch){
            for(int frame=0; frame<onsets_diff.rows(); ++frame){
                if(!onsets_diff(frame, pitch)) continue;
                onset = offset = frame;
                std::vector<float>v_tmp;
                
                while(offset<length && (onsets_value(offset, pitch) || frames_value(offset, pitch)) ){
                    if(onsets_value(offset, pitch)) v_tmp.push_back(velocity(offset, pitch));
                    ++offset;
                }

                if(offset>onset){
                    p_est.push_back(pitch+21);
                    i_est.push_back({float(onset*N_HOP)/float(SR), float(offset*N_HOP)/float(SR)});
                    float vel = 0;
                    if(!v_tmp.empty()){
                        vel = std::accumulate(v_tmp.begin(), v_tmp.end(), 0.f)/v_tmp.size();
                    }
                    v_est.push_back(std::min(int(vel*127), 127));
                }
            }
        }
    }

    std::vector<std::vector<float>> melspec(const std::string& path, int n_fft = 2048, int n_mel = 229, 
                                            int fmin = 30, int fmax = 8000, float power=1.f){
        std::vector<float> x = librosa::load(path, SR);
        std::vector<std::vector<float>> mels = librosa::Feature::melspectrogram(x, SR, n_fft, N_HOP, "hann", true, "reflect", power, n_mel, fmin, fmax, true);
        auto trans = [] (float v) { return logf(std::max(v, 1e-5f)); };
        for(size_t i=0; i<mels.size(); ++i){
            std::transform(mels[i].begin(), mels[i].end(), mels[i].begin(), trans);
        }
        return mels;
    }
};

const int AMT::INPUT_H = 512;
const int AMT::INPUT_W = 229;
const int AMT::OUTPUT_H = 512;
const int AMT::OUTPUT_W = 88;
const int AMT::SR = 16000;
const int AMT::N_HOP = 512;
const std::string AMT::INPUT_BLOB_NAME = "melspec";
const std::vector<std::string> AMT::OUTPUT_BLOB_NAMES {"onset_pred", "offset_pred", "activation_pred", "frame_pred", "velocity_pred"};
Logger AMT::gLogger;

void parse_args(int argc, char* argv[], std::string &engine_file_path, std::string &wavpath){
    // create a model using the API directly and serialize it to a stream
    if (argc == 3) {
        engine_file_path = argv[1];
        wavpath = argv[2];
    } else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "run 'python src/py/trt.py' to serialize model first!" << std::endl;
        std::cerr << "Then use the following command:" << std::endl;
        std::cerr << "./amt ../../../model.engine wav_path // deserialize file and run inference" << std::endl;
        exit(1);
    }
}

int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    std::string engine_file_path;
    std::string wavpath;
    parse_args(argc, argv, engine_file_path, wavpath);

    AMT amt(engine_file_path, 16);
    amt.transcribe(wavpath);
    return 0;
}