// Path for c_predict_api
// #include <dmlc/base.h>
// #include <dmlc/memory_io.h>
// #include </mxnet/include/mxnet/symbolic.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <time.h>
#include <opencv2/opencv.hpp>

#include </mxnet/include/mxnet/c_predict_api.h>
#include </mxnet/include/mxnet/c_api.h>

const mx_float DEFAULT_MEAN = 0.0;
const std::string CATEGORY_NAMES[2] = {"BG", "Person"};

// Read file to buffer
class BufferFile {
 public :
    std::string file_path_;
    int length_;
    char* buffer_;

    explicit BufferFile(std::string file_path)
    :file_path_(file_path) {

        std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
        if (!ifs) {
            std::cerr << "Can't open the file. Please check " << file_path << ". \n";
            length_ = 0;
            buffer_ = NULL;
            return;
        }

        ifs.seekg(0, std::ios::end);
        length_ = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        std::cout << file_path.c_str() << " ... "<< length_ << " bytes\n";

        buffer_ = new char[sizeof(char) * length_];
        ifs.read(buffer_, length_);
        ifs.close();
    }

    int GetLength() {
        return length_;
    }
    char* GetBuffer() {
        return buffer_;
    }

    ~BufferFile() {
        if (buffer_) {
          delete[] buffer_;
          buffer_ = NULL;
        }
    }
};

void GetImageFile(const std::string image_file,
                  mx_float* image_data, const int channels,
                  const cv::Size resize_size, const mx_float* mean_data = NULL) {
    // Read all kinds of file into a BGR color 3 channels image
    cv::Mat im_ori = cv::imread(image_file, cv::IMREAD_COLOR);

    if (im_ori.empty()) {
        std::cerr << "Can't open the image. Please check " << image_file << ". \n";
        assert(false);
    }

    cv::Mat im;

    resize(im_ori, im, resize_size);

    int size = im.rows * im.cols * channels;

    mx_float* ptr_image_r = image_data;
    mx_float* ptr_image_g = image_data + size / 3;
    mx_float* ptr_image_b = image_data + size / 3 * 2;

    float mean_b, mean_g, mean_r;
    mean_b = mean_g = mean_r = DEFAULT_MEAN;

    for (int i = 0; i < im.rows; i++) {
        uchar* data = im.ptr<uchar>(i);

        for (int j = 0; j < im.cols; j++) {
            if (mean_data) {
                mean_r = *mean_data;
                if (channels > 1) {
                    mean_g = *(mean_data + size / 3);
                    mean_b = *(mean_data + size / 3 * 2);
                }
               mean_data++;
            }
            if (channels > 1) {
                *ptr_image_b++ = static_cast<mx_float>(*data++) - mean_b;
                *ptr_image_g++ = static_cast<mx_float>(*data++) - mean_g;
            }

            *ptr_image_r++ = static_cast<mx_float>(*data++) - mean_r;;
        }
    }
}

// LoadSynsets
// Code from : https://github.com/pertusa/mxnet_predict_cc/blob/master/mxnet_predict.cc


void PrintOutputResult(const std::vector<float>& data) {

    float best_accuracy = 0.0;
    int best_idx = 0;
    int index = 0;
    float score = 0;
    int category_id = -1;

    std::cout << "get "<< int(data.size())/6 << " objects\n";

    // data format [cateogry, score, min-x, min-y, max-x, max-y]
    // category in [0, 1, 2].
    //             0: background
    //             1:  bike
    //             2:  e-bike
    // score in [0~1]
    while(index < static_cast<int>(data.size())) {
        category_id = data[index]+1;
        score = data[index+1];

        if (category_id > 0 && score > 0.55) {
            std::cout << "index[" << index << "] category id: " << category_id << ", Category Name: " << CATEGORY_NAMES[category_id] << ", score: " << score << ". BBox: [min-x: " << data[index+2] << ", min-y: " << data[index+3] << ", max-x: " << data[index+4] << ", max-y: " << data[index+5] << "]" << std::endl;
        }

        index = index + 6;
        if(index > 100)
            break;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "No test image here." << std::endl
        << "Usage: ./image-classification-predict apple.jpg" << std::endl;
        return 0;
    }

    clock_t base_start_time, final_end_time;
    clock_t startTime,endTime;

    base_start_time = clock();

    startTime = clock();
    std::string test_file;
    test_file = std::string(argv[1]);

    // Models path for your model, you have to modify it
    std::string json_file = "/mnt/models/train-inception-v5/deploy_ssd-symbol.json";
    std::string param_file = "/mnt/models/train-inception-v5/deploy_ssd-0512.params";

    // std::string json_file = "/mnt/jobs/job2/deploy_ssd-1-1-symbol.json";
    // std::string param_file = "/mnt/jobs/job2/deploy_ssd-1-1-0150.params";

    BufferFile json_data(json_file);
    BufferFile param_data(param_file);

    endTime = clock();
    double read_model_costs = (double)(endTime - startTime) / CLOCKS_PER_SEC * 1000;
    std::cout << "load model file to memory costs : " << read_model_costs << "ms" << std::endl;

    startTime = clock();
    // Parameters
    int dev_type = 2;  // 1: cpu, 2: gpu
    int dev_id = 1;  // arbitrary.
    mx_uint num_input_nodes = 1;  // 1 for feedforward
    const char* input_key[1] = {"data"};
    const char** input_keys = input_key;

    // Image size and channels
    int width = 360;
    int height = 360;
    int channels = 3;

    const mx_uint input_shape_indptr[2] = { 0, 4 };
    const mx_uint input_shape_data[4] = { 1,
                                        static_cast<mx_uint>(channels),
                                        static_cast<mx_uint>(height),
                                        static_cast<mx_uint>(width)};
    PredictorHandle pred_hnd = 0;

    if (json_data.GetLength() == 0 ||
        param_data.GetLength() == 0) {
        return -1;
    }

    // Create Predictor
    assert(0==MXPredCreate((const char*)json_data.GetBuffer(),
        (const char*)param_data.GetBuffer(),
        static_cast<size_t>(param_data.GetLength()),
        dev_type,
        dev_id,
        num_input_nodes,
        input_keys,
        input_shape_indptr,
        input_shape_data,
        &pred_hnd));
    assert(pred_hnd);

    endTime = clock();
    double init_model_costs = (double)(endTime - startTime) / CLOCKS_PER_SEC * 1000;
    std::cout << "init model costs : " << init_model_costs << "ms" << std::endl;

    startTime = clock();
    int image_size = width * height * channels;

    // Read Mean Data
    const mx_float* nd_data = NULL;

    // Read Image Data
    std::vector<mx_float> image_data = std::vector<mx_float>(image_size);

    GetImageFile(test_file, image_data.data(),
                 channels, cv::Size(width, height));

    // Set Input Image
    MXPredSetInput(pred_hnd, "data", image_data.data(), image_size);

    endTime = clock();
    double setup_inputs_costs = (double)(endTime - startTime) / CLOCKS_PER_SEC * 1000;
    std::cout << "bind data costs : " << setup_inputs_costs << "ms" << std::endl;

    startTime = clock();
    // Do Predict Forward
    MXPredForward(pred_hnd);
    endTime = clock();
    double forward_costs = (double)(endTime - startTime) / CLOCKS_PER_SEC * 1000;
    std::cout << "model forward costs : " << forward_costs << "ms" << std::endl;

    startTime = clock();
    mx_uint output_index = 0;
    mx_uint *shape = 0;
    mx_uint shape_len;
    // Get Output Result
    MXPredGetOutputShape(pred_hnd, output_index, &shape, &shape_len);
    size_t size = 1;
    for (mx_uint i = 0; i < shape_len; ++i) size *= shape[i];
    std::vector<float> data(size);
    std::cout << "shape: " << *shape << " shape_len: " << shape_len << " size: " << size << std::endl;

    startTime = clock();
    MXNDArrayWaitToRead(&(data[0]));
    endTime = clock();
    double wait_to_read_costs = (double)(endTime - startTime) / CLOCKS_PER_SEC * 1000;
    std::cout << "wait_to_read costs : " << wait_to_read_costs << "ms" << std::endl;

    startTime = clock();
    MXPredGetOutput(pred_hnd, output_index, &(data[0]), size);
    endTime = clock();
    double filter_output_costs = (double)(endTime - startTime) / CLOCKS_PER_SEC * 1000;
    std::cout << "filter output costs : " << filter_output_costs << "ms" << std::endl;

    // Print Output Data
    PrintOutputResult(data);

    final_end_time = clock();
    double total_time = (double)(final_end_time - base_start_time) / CLOCKS_PER_SEC * 1000;
    std::cout << "total_time : " << total_time << "ms" << std::endl;

    // Release Predictor
    MXPredFree(pred_hnd);
    std::cout << "free models" << std::endl;

    return 0;
}
