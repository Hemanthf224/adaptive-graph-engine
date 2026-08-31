#pragma once

#include <string>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <thread>
#include <mutex>

namespace graph_engine {
namespace core {

struct ProfileResult {
    std::string name;
    long long start, end;
    uint32_t threadID;
};

class Profiler {
private:
    std::string m_SessionName;
    std::ofstream m_OutputStream;
    int m_ProfileCount;
    std::mutex m_Mutex;

public:
    Profiler() : m_ProfileCount(0) {}

    static Profiler& Get() {
        static Profiler instance;
        return instance;
    }

    void BeginSession(const std::string& name, const std::string& filepath = "trace.json") {
        std::lock_guard<std::mutex> lock(m_Mutex);
        m_OutputStream.open(filepath);
        WriteHeader();
        m_SessionName = name;
        m_ProfileCount = 0;
    }

    void EndSession() {
        std::lock_guard<std::mutex> lock(m_Mutex);
        if (m_OutputStream.is_open()) {
            WriteFooter();
            m_OutputStream.close();
            m_ProfileCount = 0;
        }
    }

    void WriteProfile(const ProfileResult& result) {
        std::lock_guard<std::mutex> lock(m_Mutex);
        if (!m_OutputStream.is_open()) return;

        if (m_ProfileCount++ > 0) m_OutputStream << ",";

        std::string name = result.name;
        std::replace(name.begin(), name.end(), '"', '\'');

        m_OutputStream << "{";
        m_OutputStream << "\"cat\":\"function\",";
        m_OutputStream << "\"dur\":" << (result.end - result.start) << ",";
        m_OutputStream << "\"name\":\"" << name << "\",";
        m_OutputStream << "\"ph\":\"X\",";
        m_OutputStream << "\"pid\":0,";
        m_OutputStream << "\"tid\":" << result.threadID << ",";
        m_OutputStream << "\"ts\":" << result.start;
        m_OutputStream << "}";
        m_OutputStream.flush();
    }

private:
    void WriteHeader() {
        m_OutputStream << "{\"otherData\": {},\"traceEvents\":[";
        m_OutputStream.flush();
    }

    void WriteFooter() {
        m_OutputStream << "]}";
        m_OutputStream.flush();
    }
};

class InstrumentationTimer {
private:
    std::string m_Name;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTimepoint;
    bool m_Stopped;

public:
    InstrumentationTimer(const std::string& name)
        : m_Name(name), m_Stopped(false) {
        m_StartTimepoint = std::chrono::high_resolution_clock::now();
    }

    ~InstrumentationTimer() {
        if (!m_Stopped) Stop();
    }

    void Stop() {
        auto endTimepoint = std::chrono::high_resolution_clock::now();

        long long start = std::chrono::time_point_cast<std::chrono::microseconds>(m_StartTimepoint).time_since_epoch().count();
        long long end = std::chrono::time_point_cast<std::chrono::microseconds>(endTimepoint).time_since_epoch().count();

        uint32_t threadID = std::hash<std::thread::id>{}(std::this_thread::get_id());

        Profiler::Get().WriteProfile({m_Name, start, end, threadID});
        m_Stopped = true;
    }
};

} // namespace core
} // namespace graph_engine

#define PROFILING 1
#if PROFILING
    #define PROFILE_SCOPE(name) graph_engine::core::InstrumentationTimer timer##__LINE__(name)
    #define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCTION__)
#else
    #define PROFILE_SCOPE(name)
    #define PROFILE_FUNCTION()
#endif
