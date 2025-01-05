#pragma once

#if _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT  __attribute__((dllexport))
#endif

#include "utils.h"
#include <string>

class DragPoser
{
public:
    DragPoser()
    {
        // Initialize the interpreter on first creation of DragPoser
        if (!interpreter_guard) {
            interpreter_guard = std::make_unique<py::scoped_interpreter>();
        }
        add_venv_libs();
    }

    void create_run_drag_python_object(std::string moduleName)
    {
        RunDragInstance = py::module::import(moduleName.c_str()).attr("RunDrag")(); // Calls RunDrag.__init__()
    }

    py::object get_run_drag_python_object()
    {
		return RunDragInstance;
	}

    void set_num_joints(int numJoints)
    {
		NumJoints = numJoints;
	}

    int get_num_joints()
    {
        return NumJoints;
    }

    void set_num_endeffectors(int numEndEffectors)
    {
        NumEndEffectors = numEndEffectors;
    }

    int get_num_endeffectors()
    {
		return NumEndEffectors;
	}

private:
    static std::unique_ptr<py::scoped_interpreter> interpreter_guard;
    py::object RunDragInstance;
    int NumJoints;
    int NumEndEffectors;
};

extern "C" EXPORT DragPoser* init_drag_poser();
extern "C" EXPORT void set_reference_skeleton(DragPoser* dragPoser, char* bvhPath);
extern "C" EXPORT void load_models(DragPoser* dragPoser, char* modelPath);
extern "C" EXPORT void set_mask_and_weights(DragPoser* dragPoser, float* mask, float2* weights);
extern "C" EXPORT void init_drag_model(DragPoser* dragPoser, float3 initialGlobalPos, quaternion initialGlobalRot);
extern "C" EXPORT void set_optim_params(DragPoser* dragPoser, float stopEpsPos, float stopEpsRot, int maxIter, float lr);
extern "C" EXPORT void set_lambdas(DragPoser* dragPoser, float lambdaRot, float lambdaTemporal, int temporalFutureWindow);
extern "C" EXPORT void set_global_pos(DragPoser* dragPoser, float3 globalPos);
extern "C" EXPORT void drag_pose(DragPoser* dragPoser, int nEndEffectors, float3* targetEEPos, quaternion* targetEERot, quaternion* resultPose, float3* resultGlobalPos);
extern "C" EXPORT void destroy_drag_poser(DragPoser* dragPoser);