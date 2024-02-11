#include "FloatParameter.h"

FloatParameter::FloatParameter()
{

}

FloatParameter::FloatParameter(std::vector<float>* pointerVectorData)
{
	m_pointerVectorData = pointerVectorData;
}

FloatParameter::FloatParameter(std::vector<std::vector<float>>* pointerMatrixData)
{
	m_pointerMatrixData = pointerMatrixData;
}

FloatParameter::FloatParameter(float data)
{
	m_singleData = data;
}

FloatParameter::FloatParameter(const std::vector<float>& data)
{
	m_vectorData = data;
}

FloatParameter::FloatParameter(int data)
{
	m_intData = data;
}

float FloatParameter::GeFloatData() const
{
	return m_singleData;
}

std::vector<float> FloatParameter::GetFVectorData() const
{
	return m_vectorData;
}

std::vector<float>* FloatParameter::GetPointerFVectorData() const
{
	return m_pointerVectorData;
}

std::vector<std::vector<float>>* FloatParameter::GetPointerFMatrixData() const
{
	return m_pointerMatrixData;
}

int FloatParameter::GetIntData() const
{
	return m_intData;
}

void FloatParameter::print() const
{
	std::cout << "FloatParameter" << std::endl;
}