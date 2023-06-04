#include <iostream>
#include <map>
#include <string>
#include <random>
#include <chrono>
#include <omp.h>

#include "TaskManager.h"
#include "FloatParameter.h"
#include "OpenCV_Parameter.h"

//#define MATRIX_MULT_VECTOR

#define OPENCV_TASK

#define M 16000
#define N 80000

//#define DEBUG
void ThreadSafeLog(const std::string& out)
{
#ifdef DEBUG
    std::cout << out << std::endl;
#endif // DEBUG
}

bool Test(const std::vector<float>& simpleRes, const std::vector<float>& taskManagerRes,
    const std::vector<float>& ompRes)
{
    uint32_t size = simpleRes.size();
    for (size_t i = 0; i < size; ++i)
    {
        if (std::fabs(simpleRes[i] - taskManagerRes[i]) > 0.1 && std::fabs(ompRes[i] - taskManagerRes[i]) > 0.1 &&
            std::fabs(ompRes[i] - simpleRes[i]) > 0.1)
        {
            return false;
        }
    }

    return true;
}

std::vector<float> GetRandomVector(uint32_t size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(1, 10);

    std::vector<float> vec(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = dist(gen);
    }

    return vec;
}

const double pi = std::acos(-1);

std::vector<float> MatrixMultVector(const std::vector<std::vector<float>>& matrix, const std::vector<float>& vector)
{
    std::vector<float> res(matrix.size(), 0.0);
    for (size_t i = 0; i < matrix.size(); ++i)
    {
        for (size_t j = 0; j < vector.size(); ++j)
        {
            res[i] += vector[j] * matrix[i][j];
        }
    }

    return res;
}

std::vector<float> MatrixMultVectorOMP(const std::vector<std::vector<float>>& matrix, const std::vector<float>& vector)
{
    std::vector<float> res(matrix.size(), 0.0);

#pragma omp parallel for
    for (int i = 0; i < matrix.size(); ++i)
    {
        for (int j = 0; j < vector.size(); ++j)
        {
            res[i] += vector[j] * matrix[i][j];
        }
    }
    return res;
}

std::map<std::string, FloatParameter> VectorMultVector(
    const std::map<std::string,
    FloatParameter>& input_parameters, int32_t id)
{
    float res = 0;
    const std::vector<std::vector<float>>* matrix =
        std::move((input_parameters.at("matrix")).GetPointerFMatrixData());
    const std::vector<float>* vector =
        std::move((input_parameters.at("vector")).GetPointerFVectorData());
    for (size_t i = 0; i < vector->size(); ++i)
    {
        res += matrix->at(id).at(i) * vector->at(i);
    }

    FloatParameter parameter(res);
    std::map<std::string, FloatParameter> output_parameters;
    output_parameters.insert(std::pair<std::string, FloatParameter>(std::to_string(id), parameter));
    return output_parameters;
}

std::map<std::string, FloatParameter> LineMultVector(
    const std::map<std::string,
    FloatParameter>& input_parameters, int32_t id)
{
    const std::vector<std::vector<float>>* line =
        std::move((input_parameters.at(std::to_string(id))).GetPointerFMatrixData());
    const std::vector<float>* vector =
        std::move((input_parameters.at("vector")).GetPointerFVectorData());
    const int lineSize =
        (input_parameters.at("lineSize")).GetIntData();

    std::vector<float> res(lineSize, 0);
  
    for (size_t lineIndex = 0; lineIndex < lineSize; ++lineIndex)
    {
        for (size_t i = 0; i < vector->size(); ++i)
        {
            res[lineIndex] += line->at(lineIndex).at(i) * vector->at(i);
        }
    }

    FloatParameter parameter(res);
    std::map<std::string, FloatParameter> output_parameters;
    output_parameters.insert(std::pair<std::string, FloatParameter>("res_" + std::to_string(id), parameter));
    return output_parameters;
}

std::map<std::string, FloatParameter> CollectLinesIntoVector(
    const std::map<std::string, FloatParameter>& input_parameters,
    int32_t id)
{
    int size =
        (input_parameters.at("size")).GetIntData();
    int lineSize =
        (input_parameters.at("lineSize")).GetIntData();

    int linesCount = size / lineSize;

    std::vector<float> res(size);
    std::vector<float> currentLine(lineSize);
    for (size_t i = 0; i < linesCount; ++i)
    {
        currentLine = std::move(input_parameters.at("res_" + std::to_string(i)).GetFVectorData());
        for (size_t j = 0; j < lineSize; ++j)
        {
            res[j + i * lineSize] = currentLine[j];
        }
    }

    FloatParameter parameter(res);
    std::map<std::string, FloatParameter> output_parameters;
    output_parameters.insert(std::make_pair("result", parameter));
    return output_parameters;
}

std::map<std::string, FloatParameter> CollectValuesIntoVector(
    const std::map<std::string, FloatParameter>& input_parameters,
    int32_t id)
{
    int size =
        (input_parameters.at("size")).GetIntData();
    std::vector<float> res(size);
    for (size_t i = 0; i < size; ++i)
    {
        res[i] = input_parameters.at(std::to_string(i)).GeFloatData();
    }

    FloatParameter parameter(res);
    std::map<std::string, FloatParameter> output_parameters;
    output_parameters.insert(std::make_pair("result", parameter));
    return output_parameters;
}

void processing_image(std::string input_dir, std::string output_dir, int index)
{
    // формирование имени файла
    std::string filename = input_dir + '(' + std::to_string(index) + ')' + ".jpg";

    // загрузка изображения
    cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cout << "Could not open or find the image: " << filename << std::endl;
    }

    // преобразование в оттенки серого
    cv::Mat gray_image;
    cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

    // размытие изображения
    cv::Mat blurred_image;
    blur(gray_image, blurred_image, cv::Size(6, 6));

    // формирование имени файла для сохранения
    std::string output_filename = output_dir + std::to_string(index) + ".jpg";

    // сохранение обработанного изображения
    imwrite(output_filename, blurred_image);
}

std::map<std::string, OpenCV_Parameter> loadImages(
    const std::map<std::string, OpenCV_Parameter>& input_parameters,
    int32_t id)
{
    int imagesBlockSize = (input_parameters.at("images_block_size")).GetIntData();

    int startIndex = imagesBlockSize * id;

    std::string input_dir = (input_parameters.at("input_dir")).GetStringData();
    std::vector<cv::Mat> images;

    int vectorIndex = 0;
    for (int i = startIndex; i < startIndex + imagesBlockSize; ++i)
    {
        std::string filename = input_dir + '(' + std::to_string(i) + ')' + ".jpg";
        images.push_back(cv::imread(filename, cv::IMREAD_COLOR));
        ++vectorIndex;
    }

    OpenCV_Parameter parameter(images);
    std::map<std::string, OpenCV_Parameter> output_parameters;
    output_parameters.insert(std::make_pair("images_" + std::to_string(id), parameter));
    std::vector<cv::Mat> new_images3 = (output_parameters.at("images_" + std::to_string(id))).GetVectorMatData();
    return output_parameters;
}

std::map<std::string, OpenCV_Parameter> toGrayColor(
    const std::map<std::string, OpenCV_Parameter>& input_parameters,
    int32_t id)
{
    std::vector<cv::Mat> images = std::move((input_parameters.at("images_" + std::to_string(id))).GetVectorMatData());

    std::vector<cv::Mat> gray_images(images.size());

    for (size_t i = 0; i < images.size(); ++i)
    {
        if (images[i].empty())
        {
            std::cout << "toGrayColor image is empty. id: " << id << " index: " << std::endl;
            break;
        }
        cvtColor(images[i], gray_images[i], cv::COLOR_BGR2GRAY);
    }

    OpenCV_Parameter parameter(gray_images);
    std::map<std::string, OpenCV_Parameter> output_parameters;
    output_parameters.insert(std::make_pair("gray_images_" + std::to_string(id), parameter));
    return output_parameters;
}

std::map<std::string, OpenCV_Parameter> toBlur(
    const std::map<std::string, OpenCV_Parameter>& input_parameters,
    int32_t id)
{
    std::vector<cv::Mat> gray_images = std::move((input_parameters.at("gray_images_" + std::to_string(id))).GetVectorMatData());

    std::vector<cv::Mat> blur_images(gray_images.size());

    for (size_t i = 0; i < gray_images.size(); ++i)
    {
        blur(gray_images[i], blur_images[i], cv::Size(6, 6));
    }

    OpenCV_Parameter parameter(blur_images);
    std::map<std::string, OpenCV_Parameter> output_parameters;
    output_parameters.insert(std::make_pair("blur_images_" + std::to_string(id), parameter));
    return output_parameters;
}

std::map<std::string, OpenCV_Parameter> saveImages(
    const std::map<std::string, OpenCV_Parameter>& input_parameters,
    int32_t id)
{
    std::vector<cv::Mat> blur_images = std::move((input_parameters.at("blur_images_" + std::to_string(id))).GetVectorMatData());

    int startIndex = blur_images.size() * id;

    std::string output_dir = (input_parameters.at("output_dir")).GetStringData();
    std::vector<cv::Mat> res_images(blur_images.size());

    int vectorIndex = 0;
    for (int i = startIndex; i < startIndex + blur_images.size(); ++i)
    {
        std::string filename = output_dir + std::to_string(i) + ".jpg";
        cv::imwrite(filename, blur_images[vectorIndex]);
        ++vectorIndex;
    }

    OpenCV_Parameter parameter(true);
    std::map<std::string, OpenCV_Parameter> output_parameters;
    output_parameters.insert(std::make_pair("res_" + std::to_string(id), parameter));
    return output_parameters;
}

std::map<std::string, OpenCV_Parameter> checkingExecution(
    const std::map<std::string, OpenCV_Parameter>& input_parameters,
    int32_t id)
{
    OpenCV_Parameter parameter(true);
    std::map<std::string, OpenCV_Parameter> output_parameters;
    output_parameters.insert(std::make_pair("result", parameter));
    return output_parameters;
}

namespace boost {
    namespace serialization {

        template<class Archive>
        void save(Archive& ar, const cv::Mat& mat, const unsigned int version)
        {
            // Сохраняем размер матрицы
            int rows = mat.rows;
            int cols = mat.cols;
            int type = mat.type();
            ar& BOOST_SERIALIZATION_NVP(rows);
            ar& BOOST_SERIALIZATION_NVP(cols);
            ar& BOOST_SERIALIZATION_NVP(type);

            // Сохраняем данные матрицы
            int size = rows * cols * mat.elemSize();
            const char* data = reinterpret_cast<const char*>(mat.data);
            ar& boost::serialization::make_array(data, size);
        }

        template<class Archive>
        void load(Archive& ar, cv::Mat& mat, const unsigned int version)
        {
            // Загружаем размер матрицы
            int rows, cols, type;
            ar& BOOST_SERIALIZATION_NVP(rows);
            ar& BOOST_SERIALIZATION_NVP(cols);
            ar& BOOST_SERIALIZATION_NVP(type);

            // Создаем матрицу
            mat.create(rows, cols, type);

            // Загружаем данные матрицы
            int size = rows * cols * mat.elemSize();
            char* data = reinterpret_cast<char*>(mat.data);
            ar& boost::serialization::make_array(data, size);
        }

        template<class Archive>
        void serialize(Archive& ar, cv::Mat& mat, const unsigned int version)
        {
            boost::serialization::split_free(ar, mat, version);
        }

    } // namespace serialization
} // namespace boost

int main(int argc, char* argv[])
{

#ifdef OPENCV_TASK

    double start;
    double end;

    int num_images = 1000; // количество изображений
    std::string input_dir = "input_images/"; // директория с исходными изображениями
    std::string output_dir = "output_images/"; // директория для сохранения обработанных изображений

    start = omp_get_wtime();
    // загрузка и обработка изображений
    for (int i = 1; i <= num_images; i++)
    {
        processing_image(input_dir, output_dir, i);
    }
    end = omp_get_wtime();
    double simpleSeconds = end - start;

    start = omp_get_wtime();
    // загрузка и обработка изображений
#pragma omp parallel for
    for (int i = 1; i <= num_images; i++)
    {
        processing_image(input_dir, output_dir, i);
    }
    end = omp_get_wtime();
    double ompSeconds = end - start;

    boost::mpi::environment env;
    boost::mpi::communicator world;

    std::vector<Task<OpenCV_Parameter>> tasks;
    auto loadImagesFunction = loadImages;
    auto toGrayColorFunction = toGrayColor;
    auto toBlurFunction = toBlur;
    auto saveImagesFunction = saveImages;
    auto checkingExecutionFunction = checkingExecution;

    int images_block_size = 100;
    int blocks_count = num_images / images_block_size;

    for (size_t i = 0; i < blocks_count; ++i)
    {
        std::vector<std::string> loadImagesParameters = { "input_dir", "images_block_size"};
        Task<OpenCV_Parameter> loadImagesTask(loadImagesFunction, loadImagesParameters, i);
        tasks.push_back(loadImagesTask);

        std::vector<std::string> toGrayColorParameters = { "images_" + std::to_string(i)};
        Task<OpenCV_Parameter> toGrayColorTask(toGrayColorFunction, toGrayColorParameters, i);
        tasks.push_back(toGrayColorTask);

        std::vector<std::string> toBlurParameters = { "gray_images_" + std::to_string(i)};
        Task<OpenCV_Parameter> toBlurTask(toBlurFunction, toBlurParameters, i);
        tasks.push_back(toBlurTask);

        std::vector<std::string> saveImagesParameters = { "blur_images_" + std::to_string(i), "output_dir"};
        Task<OpenCV_Parameter> saveImagesTask(saveImagesFunction, saveImagesParameters, i);
        tasks.push_back(saveImagesTask);
    }

    std::vector<std::string> parametersForCheckingExecution(blocks_count);
    for (size_t i = 0; i < blocks_count; ++i)
    {
        parametersForCheckingExecution[i] = "res_" + std::to_string(i);
    }
    Task<OpenCV_Parameter> checkingExecutionTask(checkingExecutionFunction, parametersForCheckingExecution, 0);
    tasks.push_back(checkingExecutionTask);

    std::vector<std::pair<std::string, OpenCV_Parameter>> parameters;

    OpenCV_Parameter inputDirParameter(input_dir);
    parameters.push_back(std::make_pair("input_dir", inputDirParameter));

    output_dir = "output_images_taskManager/"; // директория для сохранения обработанных изображений
    OpenCV_Parameter outputDirParameter(output_dir);
    parameters.push_back(std::make_pair("output_dir", outputDirParameter));
    OpenCV_Parameter imagesBlockSizeParameter(images_block_size);
    parameters.push_back(std::make_pair("images_block_size", imagesBlockSizeParameter));

    TaskManager<OpenCV_Parameter> taskManager(parameters, tasks);
    start = omp_get_wtime();
    taskManager.Run(world.rank());
    end = omp_get_wtime();
    if (world.rank() == 0)
    {
        OpenCV_Parameter resultTaskManager = std::move(taskManager.GetResult());
        double taskManagerSeconds = end - start;

        std::cout << "taskManagerSeconds: " << taskManagerSeconds <<
            ' ' << "simpleSeconds: " << simpleSeconds <<
            ' ' << "omp: " << ompSeconds;
        std::cout << std::endl;
    }
#endif // OPENCV_TASK

#ifdef MATRIX_MULT_VECTOR
    double start;
    double end;

    boost::mpi::environment env;
    boost::mpi::communicator world;

    std::vector<Task<FloatParameter>> tasks;
    auto lineMultVectorFunction = LineMultVector;

    int processesCount = 8;
    int lineSize = M / processesCount;

    for (size_t i = 0; i < processesCount; ++i)
    {
        std::vector<std::string> lineMultVectorParameters = { "lineSize", "vector", std::to_string(i) };
        Task<FloatParameter> lineMultVectorTask(lineMultVectorFunction, lineMultVectorParameters, i);

        tasks.push_back(lineMultVectorTask);
    }

    auto collectLinesIntoVectorFunction = CollectLinesIntoVector;
    std::vector<std::string> collectLinesIntoVectorParameters;
    for (size_t i = 0; i < processesCount; ++i)
    {
        collectLinesIntoVectorParameters.push_back("res_" + std::to_string(i));
    }
    collectLinesIntoVectorParameters.push_back("size");
    collectLinesIntoVectorParameters.push_back("lineSize");

    Task<FloatParameter> collectValuesTask(collectLinesIntoVectorFunction, collectLinesIntoVectorParameters, 0);
    tasks.push_back(collectValuesTask);

    std::vector<std::pair<std::string, FloatParameter>> parameters;

    std::vector<std::vector<float>> matrix;
    for (size_t i = 0; i < M; ++i)
    {
        matrix.push_back(std::move(GetRandomVector(N)));
    }

    std::vector<float> vector = std::move(GetRandomVector(N));

    start = omp_get_wtime();
    std::vector<float> resultOMP = std::move(MatrixMultVectorOMP(matrix, vector));
    end = omp_get_wtime();
    double ompSeconds = end - start;
    start = omp_get_wtime();
    std::vector<float> resultSimple = std::move(MatrixMultVector(matrix, vector));
    end = omp_get_wtime();
    double simpleSeconds = end - start;

    std::vector<std::vector<float>> line(lineSize);
    for (size_t i = 0; i < processesCount; ++i)
    {
        for (size_t j = 0; j < lineSize; ++j)
        {
            line[j] = std::move(matrix[j + lineSize * i]);
        }

        FloatParameter lineParameter(&line);
        parameters.push_back(std::make_pair(std::to_string(i), lineParameter));
    }

    FloatParameter vectorParameter(&vector);
    parameters.push_back(std::make_pair("vector", vectorParameter));
    FloatParameter sizeParameter(M);
    parameters.push_back(std::make_pair("size", sizeParameter));
    FloatParameter lineSizeParameter(lineSize);
    parameters.push_back(std::make_pair("lineSize", lineSizeParameter));

    TaskManager<FloatParameter> taskManager(parameters, tasks);

    start = omp_get_wtime();
    taskManager.Run(world.rank());
    if (world.rank() == 0)
    {
        FloatParameter resultTaskManager = std::move(taskManager.GetResult());
        end = omp_get_wtime();
        double taskManagerSeconds = end - start;

        if (world.size() == 1)
            resultTaskManager = resultSimple;

        std::cout << "taskManagerSeconds: " << taskManagerSeconds <<
            ' ' << "simpleSeconds: " << simpleSeconds <<
            ' ' << "omp: " << ompSeconds;
        std::cout << std::endl;
        std::cout << "Correct? : " << Test(resultSimple, resultTaskManager.GetFVectorData(), resultOMP);
    }
#endif // MATRIX_MULT_VECTOR

    return 0;
}