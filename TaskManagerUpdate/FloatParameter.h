#pragma once

#include <variant>
#include <string>
#include <cassert>
#include <iostream>
#include <vector>

#include <boost/serialization/access.hpp>
#include <boost/serialization/map.hpp>

class FloatParameter
{
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar& m_singleData;
		ar& m_vectorData;
		ar& m_pointerVectorData;
		ar& m_pointerMatrixData;
		ar& m_intData;
	}

private:
	float m_singleData = 0.0;
	std::vector<float> m_vectorData = {};
	std::vector<float>* m_pointerVectorData = {};
	std::vector<std::vector<float>>* m_pointerMatrixData = {};
	int m_intData = 0;
public:
	FloatParameter();
	FloatParameter(std::vector<float>* pointerVectorData);
	FloatParameter(std::vector<std::vector<float>>* pointerMatrixData);
	FloatParameter(float data);
	FloatParameter(const std::vector<float> &data);
	FloatParameter(int data);

	float GeFloatData() const;
	std::vector<float> GetFVectorData() const;
	std::vector<float>* GetPointerFVectorData() const;
	std::vector<std::vector<float>>* GetPointerFMatrixData() const;
	int GetIntData() const;

	void print() const;
};

