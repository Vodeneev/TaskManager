#include "OpenCV_Parameter.h"

OpenCV_Parameter::OpenCV_Parameter()
{

}

OpenCV_Parameter::OpenCV_Parameter(bool data)
{
	m_boolData = data;
}

OpenCV_Parameter::OpenCV_Parameter(const std::vector<cv::Mat>& data)
{
	m_vectorMatData = data;
}

OpenCV_Parameter::OpenCV_Parameter(int data)
{
	m_intData = data;
}


OpenCV_Parameter::OpenCV_Parameter(const std::string &data)
{
	m_stringData = data;
}


bool OpenCV_Parameter::GetBoolData() const
{
	return m_boolData;
}

std::vector<cv::Mat> OpenCV_Parameter::GetVectorMatData() const
{
	return m_vectorMatData;
}

int OpenCV_Parameter::GetIntData() const
{
	return m_intData;
}

std::string OpenCV_Parameter::GetStringData() const
{
	return m_stringData;
}
