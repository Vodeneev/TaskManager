#pragma once

#include <variant>
#include <string>
#include <cassert>
#include <iostream>
#include <vector>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>

#include <opencv2/opencv.hpp>

class OpenCV_Parameter
{
	friend class boost::serialization::access;

	template<class Archive>
	void serialize(Archive& ar, cv::Mat& mat, const unsigned int version)
	{
		int cols = mat.cols;
		int rows = mat.rows;
		int type = mat.type();

		ar& cols;
		ar& rows;
		ar& type;

		if (mat.isContinuous()) {
			int dataSize = cols * rows * mat.elemSize();
			ar& boost::serialization::make_array(mat.ptr(), dataSize);
		}
		else {
			int rowSize = cols * mat.elemSize();
			for (int i = 0; i < rows; i++) {
				ar& boost::serialization::make_array(mat.ptr(i), rowSize);
			}
		}
	}

	template<class Archive>
	void serialize(Archive& ar, std::vector<cv::Mat>& vec, const unsigned int version)
	{
		size_t size = vec.size();
		ar& size;

		for (size_t i = 0; i < size; i++) {
			ar& vec[i];
		}
	}

	template<class Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar& m_stringData;
		ar& m_boolData;
		ar& m_vectorMatData;
		ar& m_intData;
	}

private:
	std::string m_stringData;
	bool m_boolData;
	std::vector<cv::Mat> m_vectorMatData;
	int m_intData;
public:
	OpenCV_Parameter();
	OpenCV_Parameter(bool data);
	OpenCV_Parameter(const std::vector<cv::Mat>& data);
	OpenCV_Parameter(int data);
	OpenCV_Parameter(const std::string& data);

	bool GetBoolData() const;
	std::vector<cv::Mat> GetVectorMatData() const;
	int GetIntData() const;
	std::string GetStringData() const;
};

