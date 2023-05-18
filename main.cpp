#include <iostream>
#include <map>
#include <string>
#include <random>
#include <chrono>
#include <omp.h>

#include "TaskManager.h"
#include "FloatParameter.h"

#define M 1000
#define N 1000

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
    float res = 0;
    const std::vector<std::vector<float>>* line =
        std::move((input_parameters.at(std::to_string(id))).GetPointerFMatrixData());
    const std::vector<float>* vector =
        std::move((input_parameters.at("vector")).GetPointerFVectorData());
    const int lineSize =
        (input_parameters.at("lineSize")).GetIntData();


    for (size_t lineIndex = 0; lineIndex < lineSize; ++lineIndex)
    {
        for (size_t i = 0; i < vector->size(); ++i)
        {
            res += line->at(id).at(i) * vector->at(i);
        }
    }

    FloatParameter parameter(res);
    std::map<std::string, FloatParameter> output_parameters;
    output_parameters.insert(std::pair<std::string, FloatParameter>(std::to_string(id), parameter));
    return output_parameters;
}

std::map<std::string, FloatParameter> CollectValuesIntoVector(
    const std::map<std::string, FloatParameter>& input_parameters,
    int32_t id)
{
    std::vector<float> res;
    const std::vector<float>* vector =
        (input_parameters.at("vector")).GetPointerFVectorData();
    int size = vector->size();
    for (size_t i = 0; i < size; ++i)
    {
        res.push_back((input_parameters.at(std::to_string(i))).GeFloatData());
    }

    FloatParameter parameter(res);
    std::map<std::string, FloatParameter> output_parameters;
    output_parameters.insert(std::make_pair("result", parameter));
    return output_parameters;
}

int main(int argc, char* argv[])
{
    double start;
    double end;

    std::vector<Task<FloatParameter>> tasks;
    for (size_t i = 0; i < M; ++i)
    {
        auto vectorMultVectorFunction = VectorMultVector;
        std::vector<std::string> vectorMultVectorParameters = { "matrix", "vector" };
        Task<FloatParameter> vectorMultVectorTask(vectorMultVectorFunction, vectorMultVectorParameters, i);
        tasks.push_back(vectorMultVectorTask);
    }

    auto collectValuesIntoVectorFunction = CollectValuesIntoVector;
    std::vector<std::string> collectValuesIntoVectorParameters;
    for (size_t i = 0; i < M; ++i)
    {
        collectValuesIntoVectorParameters.push_back(std::to_string(i));
    }
    Task<FloatParameter> vectorMultVectorTask(collectValuesIntoVectorFunction, collectValuesIntoVectorParameters, 0);
    tasks.push_back(vectorMultVectorTask);

    std::vector<std::pair<std::string, FloatParameter>> parameters;

    std::vector<std::vector<float>> matrix;
    for (size_t i = 0; i < M; ++i)
    {
        matrix.push_back(std::move(GetRandomVector(N)));
    }

    std::vector<float> vector = std::move(GetRandomVector(N));

    FloatParameter matrixParameter(&matrix);
    parameters.push_back(std::make_pair("matrix", matrixParameter));
    FloatParameter vectorParameter(&vector);
    parameters.push_back(std::make_pair("vector", vectorParameter));

    start = omp_get_wtime();
    TaskManager<FloatParameter> taskManager(parameters, tasks);
    taskManager.Run();
    FloatParameter resultTaskManager = std::move(taskManager.GetResult());
    end = omp_get_wtime();
    double taskManagerSeconds = end - start;

    start = omp_get_wtime();
    std::vector<float> resultOMP = std::move(MatrixMultVectorOMP(matrix, vector));
    end = omp_get_wtime();
    double ompSeconds = end - start;

    start = omp_get_wtime();
    std::vector<float> resultSimple = std::move(MatrixMultVector(matrix, vector));
    end = omp_get_wtime();
    double simpleSeconds = end - start;

    std::cout << "taskManagerSeconds: " << taskManagerSeconds << 
        ' ' << "simpleSeconds: " << simpleSeconds <<
        ' ' << "omp: " << ompSeconds;
    std::cout << std::endl;
    std::cout << "Correct? : " << Test(resultSimple, resultSimple, resultTaskManager.GetFVectorData());

    return 0;
}