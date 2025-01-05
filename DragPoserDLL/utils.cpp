#include "utils.h"

void add_venv_libs()
{
    py::module sys = py::module::import("sys");
    // get python executable path
    std::filesystem::path env_path = std::filesystem::path(PYTHON3_EXECUTABLE);
    // remove last 2 directories
    env_path = env_path.parent_path().parent_path();
    std::filesystem::path site_packages = env_path;
    // find children (recursively) directory with "site-packages"
    bool found = false;
    for (auto& p : std::filesystem::recursive_directory_iterator(env_path)) {
        if (p.path().filename() == "site-packages") {
            site_packages = p.path();
            found = true;
            break;
        }
    }
    if (!found) 
    {
		std::cout << "site-packages not found" << std::endl;
		return;
	}
    // add it to python sys path
    sys.attr("path").attr("insert")(0, site_packages.string());
    // search for the src directory
    std::filesystem::path project_path = env_path.parent_path();
    found = false;
    for (auto& p : std::filesystem::recursive_directory_iterator(project_path)) {
        if (p.path().filename() == "run_drag.py") {
			project_path = p.path();
            found = true;
			break;
		}
	}
    if (!found) 
    {
        std::cout << "run_drag.py not found" << std::endl;
        return;
    }
    project_path = project_path.parent_path();
	sys.attr("path").attr("insert")(0, project_path.string());
}

py::array_t<float> create_numpy_array_quaternion(quaternion* buffer, int length)
{
    // Calculate the strides
    py::ssize_t stride[] = { sizeof(quaternion), sizeof(float) };
    py::ssize_t shape[] = { length, 4 };

    // Create a 2D numpy array view without copying the data
    return py::array_t<float>(shape, stride, reinterpret_cast<float*>(buffer), py::none());
}

py::array_t<float> create_numpy_array_float3(float3* buffer, int length)
{
    // Calculate the strides
    py::ssize_t stride[] = { sizeof(float3), sizeof(float) };
    py::ssize_t shape[] = { length, 3 };

    // Create a 2D numpy array view without copying the data
    return py::array_t<float>(shape, stride, reinterpret_cast<float*>(buffer), py::none());
}

py::array_t<float> create_numpy_array_float2(float2* buffer, int length) {
    // Calculate the strides
    py::ssize_t stride[] = { sizeof(float2), sizeof(float) };
    py::ssize_t shape[] = { length, 2 };

    // Create a 2D numpy array view without copying the data
    return py::array_t<float>(shape, stride, reinterpret_cast<float*>(buffer), py::none());
}

py::array_t<float> create_numpy_array_float(float* buffer, int length) {
    // Calculate the strides
    py::ssize_t stride[] = { sizeof(float) };
    py::ssize_t shape[] = { length };

    // Create a 1D numpy array view without copying the data
    return py::array_t<float>(shape, stride, buffer, py::none());
}

void logMessage(const std::string& message) {
    std::ofstream logFile;

    // Open the file in append mode
    logFile.open("C:\\Users\\user\\Desktop\\Projects\\DragPoser\\python\\cpp_log.txt", std::ios_base::app);

    if (logFile.is_open()) {
        // Write the message to the file
        logFile << message << std::endl;

        // Close the file
        logFile.close();
    }
    else {
        std::cerr << "Unable to open log file." << std::endl;
    }
}