#pragma once

#include <functional>
#include <map>

template<typename Type>
class Task
{
private:
	std::function<std::map<std::string, Type>
		(const std::map<std::string, Type>&, int32_t)> m_function;
	std::vector<std::string> m_namesOfInputParameters;
	std::map<std::string, Type> m_outputParameters;
	int32_t m_id = 0;

public:
	Task()
	{

	}

	Task(const 	std::function<std::map<std::string, Type>
		(const std::map<std::string, Type>&, int32_t)>& function,
		const std::vector<std::string>& namesOfInputParameters, int32_t id)
	{
		m_function = function;
		m_namesOfInputParameters = namesOfInputParameters;
		m_id = id;
	}

	bool HaveAllParameters(const std::map<std::string, Type>& parameters)
	{
		for (auto nameOfParameter : m_namesOfInputParameters)
		{
			if (parameters.find(nameOfParameter) == parameters.end())
			{
				return false;
			}
		}
		return true;
	}

	void Execute(const std::map<std::string, Type>& parameters)
	{
		m_outputParameters = m_function(parameters, m_id);
	}

	std::map<std::string, Type> GetOtputParameters()
	{
		return m_outputParameters;
	}
};