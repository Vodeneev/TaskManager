#pragma once

#include <map>
#include<vector>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <array>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>

#include <boost/serialization/access.hpp>
#include <boost/serialization/map.hpp>

#include <boost/serialization/vector.hpp>
#include <boost/mpi.hpp>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include "Task.h"
#include "FloatParameter.h"

using namespace std::chrono_literals;

//#define DEBUG_LOG

template<typename Type>
class TaskManagerForThreads
{
private:
	std::vector<Task<Type>> m_tasks;
	uint32_t m_tasksCount;
	std::map<std::string, Type> m_inputParameters;
	std::vector<std::pair<std::string, Type>> m_newParameters;

	int m_countThreads = 1;
	std::vector<std::thread> m_workThreads;
	uint32_t m_threadsCount;
	std::condition_variable m_conditionVariable;
	std::mutex m_mutexForParameters;
	std::mutex m_mutexForTasks;
	std::mutex m_mutexForLog;
	std::atomic_bool m_done;

	bool m_allTasksCompleted;
	bool m_haveResult = false;

	Type m_result;

public:

	TaskManagerForThreads()
	{
		m_tasksCount = 0;
	}
	TaskManagerForThreads(const std::map<std::string, Type>& parameters,
		const std::vector<Task<Type>>& tasks, int countThreads)
	{
		m_inputParameters = parameters;
		m_tasks = tasks;
		m_tasksCount = tasks.size();
		m_threadsCount = tasks.size() > countThreads ? countThreads : tasks.size();
		m_done = false;
	}
	void Run()
	{
		try
		{
			ThreadSafeLog("Run ");
			m_newParameters.clear();
			m_workThreads.clear();
			for (size_t i = 0; i < m_threadsCount; ++i)
			{
				m_workThreads.push_back(std::thread(&TaskManagerForThreads::TaskCompletion, this));
			}

			for (size_t i = 0; i < m_threadsCount; ++i)
			{
				m_workThreads[i].join();
			}

			if (m_allTasksCompleted)
			{
				auto result = m_inputParameters.find("result");

				if (result != m_inputParameters.end())
				{
					m_haveResult = true;
					m_result = result->second;
				}
			}
		}
		catch (...)
		{
			m_done = true;
			std::cout << "catch " << std::this_thread::get_id() << std::endl;
			throw;
		}
	}
	bool HaveResult() { return m_haveResult; }
	bool AllTaksCompleted() { return m_allTasksCompleted; }
	Type GetResult() { return m_result; }
	bool HaveNewParameters() { return m_newParameters.size() > 0; }
	std::vector<std::pair<std::string, Type>> GetNewParameters() { return m_newParameters; }
	int GetNewParametersSize() { return m_newParameters.size(); }
	void SetInputParameters(const std::map<std::string, Type>& parameters) { m_inputParameters = parameters; }
	void AddNewParameters(const std::map<std::string, Type>& parameters)
	{
		ThreadSafeLog("AddNewParameters ");
		std::unique_lock<std::mutex> lck(m_mutexForParameters);
		for (auto parameter : parameters)
		{
			m_inputParameters.insert(parameter);
			//std::cout << "parameter name: " << parameter.first << std::endl;
			m_newParameters.push_back(parameter);
		}
	}
	void AddNewParameters(const std::vector<std::pair<std::string, Type>>& parameters)
	{
		for (size_t i = 0; i < parameters.size(); ++i)
		{
			m_inputParameters[parameters[i].first] = parameters[i].second;
			m_newParameters.push_back(parameters[i]);
		}
	}
	~TaskManagerForThreads() { m_done = true; }
private:

	bool GetTask(size_t index, Task<Type>& task)
	{
		if (index > m_tasks.size())
		{
			ThreadSafeLog("GetTask not found ");
			return false;
		}
		else
		{
			ThreadSafeLog("GetTask found ");
			task = m_tasks[index];
			DeleteTask(index);
			return true;
		}
	}

	void DeleteTask(size_t index)
	{
		ThreadSafeLog("DeleteTask ");
		m_tasks.erase(m_tasks.begin() + index);
	}

	void TaskCompletion()
	{
		ThreadSafeLog("TaskCompletion ");
		Task<Type> task;
		bool needToExecute = false;
		m_done = false;
		while (!m_done)
		{
			if (m_tasksCount > 0)
			{
				{
					std::unique_lock<std::mutex> lck(m_mutexForTasks);
					for (size_t i = 0; i < m_tasks.size(); ++i)
					{
						if (m_tasks[i].HaveAllParameters(m_inputParameters))
						{
							GetTask(i, task);
							--m_tasksCount;
							needToExecute = true;
							break;
						}

						if (i == m_tasks.size() - 1)
						{
							m_allTasksCompleted = false;
							m_done = true;
						}
					}
				}
				if (needToExecute)
				{
					task.Execute(m_inputParameters);
					AddNewParameters(task.GetOtputParameters());
				}
				needToExecute = false;
			}
			else
			{
				m_allTasksCompleted = true;
				m_done = true;
			}
		}
	}

	void ThreadSafeLog(const std::string& out)
	{
#ifdef DEBUG_LOG
		std::unique_lock<std::mutex> lck(m_mutexForLog);
		std::cout << out << std::this_thread::get_id() << std::endl;
#endif // DEBUG_LOG
	}
};
