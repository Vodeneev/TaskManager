#include "TaskManagerForThreads.h"
#include <boost/mpi/collectives.hpp>

template<typename Type>
class TaskManager
{
private:
    std::vector<std::pair<std::string, Type>> m_inputParameters;
    std::vector<Task<Type>> m_tasks;
    int m_countThreadsForProceses = 1;
    Type m_result;

public:
    TaskManager(const std::vector<std::pair<std::string, Type>>& parameters,
        const std::vector<Task<Type>>& tasks, int countThreadsForProceses)
    {
        std::copy(parameters.begin(), parameters.end(), std::back_inserter(m_inputParameters));
        std::copy(tasks.begin(), tasks.end(), std::back_inserter(m_tasks));
        m_countThreadsForProceses = countThreadsForProceses;
    }

	void Run(int rank)
	{
        boost::mpi::communicator world;
        boost::mpi::environment env;

        int delta = m_tasks.size() / (world.size() - 1 == 0 ? 1 : world.size() - 1);
        if (rank == 0)
        {
            if (world.size() > 1)
            {
                std::cout << "world size: " << world.size() << " tasks size: " << m_tasks.size() << " delta: " << delta << std::endl;
                int worldSize = world.size();
                std::vector<int> taskForProcess;
                if (delta > 0)
                {
                    for (int i = 0; i < worldSize - 1; ++i)
                    {
                        for (int j = 0; i + j * (worldSize - 1) < m_tasks.size(); ++j)
                        {
                            taskForProcess.push_back(i + j * (worldSize - 1));
                        }

                        world.send(i + 1, 0, taskForProcess);
                        taskForProcess.clear();
                    }
                }
                else
                {
                    for (int i = 0; i < m_tasks.size(); ++i)
                    {
                        taskForProcess.push_back(i);
                        world.send(i + 1, 0, taskForProcess);
                        taskForProcess.clear();
                    }
                }

                std::vector<std::pair<int, bool>> processesId;
                for (size_t i = 0; i < worldSize - 1; ++i)
                {
                    processesId.push_back(std::make_pair(i + 1, false));
                }

                int countProcessesDone = 0;
                std::vector<std::pair<std::string, Type>> newParameters;
                while (countProcessesDone != (worldSize - 1))
                {
                    for (size_t i = 0; i < processesId.size(); ++i)
                    {
                        if (!processesId[i].second)
                        {
                            std::vector<std::pair<std::string, Type>> newParametersFromProcess;

                            world.recv(processesId[i].first, 0, processesId[i].second);

                            world.recv(processesId[i].first, 0, newParametersFromProcess);

                            if (processesId[i].second)
                            {
                                ++countProcessesDone;
                            }

                            if (newParametersFromProcess.size() > 0)
                            {
                                for (size_t j = 0; j < newParametersFromProcess.size(); ++j)
                                {
                                    newParameters.push_back(newParametersFromProcess[j]);

                                    if (newParametersFromProcess[j].first == "result")
                                    {
                                        m_result = newParametersFromProcess[j].second;
                                    }
                                }
                            }

                            newParametersFromProcess.clear();
                        }
                    }

                    for (size_t i = 0; i < processesId.size(); ++i)
                    {
                        if (!processesId[i].second)
                        {
                            world.send(processesId[i].first, 0, newParameters);
                        }
                    }

                    newParameters.clear();
                }
            }
            else
            {
                std::map<std::string, Type> mapParameters;
                for (size_t i = 0; i < m_inputParameters.size(); ++i)
                {
                    mapParameters.insert(std::make_pair(m_inputParameters[i].first, m_inputParameters[i].second));
                }
                TaskManagerForThreads<Type> taskManager(mapParameters, m_tasks, m_countThreadsForProceses);
                taskManager.Run();
                std::vector<std::pair<std::string, Type>> newParameters = taskManager.GetNewParameters();
                for (size_t i = 0; i < newParameters.size(); ++i)
                {
                    if (newParameters[i].first == "result")
                    {
                        m_result = newParameters[i].second;
                    }
                }
            }
        }
        else
        {
            if (rank <= m_tasks.size())
            {
                std::vector<int> tasksIndexes;
                world.recv(0, 0, tasksIndexes);

                std::map<std::string, Type> localParameters;
                for (size_t i = 0; i < m_inputParameters.size(); ++i)
                {
                    localParameters.insert(std::make_pair(m_inputParameters[i].first, m_inputParameters[i].second));
                }

                std::vector<Task<Type>> tasks(tasksIndexes.size());
                for (size_t i = 0; i < tasks.size(); ++i)
                {
                    tasks[i] = m_tasks[tasksIndexes[i]];
                }
                TaskManagerForThreads<Type> taskManager(localParameters, tasks, m_countThreadsForProceses);

                while (!taskManager.AllTaksCompleted())
                {
                    taskManager.Run();

                    world.send(0, 0, taskManager.AllTaksCompleted());

                    std::vector<std::pair<std::string, Type>> newTypeParameters = taskManager.GetNewParameters();

                    world.send(0, 0, newTypeParameters);
                    newTypeParameters.clear();

                    if (!taskManager.AllTaksCompleted())
                    {
                        world.recv(0, 0, newTypeParameters);

                        taskManager.AddNewParameters(newTypeParameters);

                        newTypeParameters.clear();
                    }
                }
            }
        }
	}

    Type GetResult() { return m_result; }
};