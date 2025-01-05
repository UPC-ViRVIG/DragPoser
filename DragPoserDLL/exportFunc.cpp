#include "exportFunc.h"

std::unique_ptr<py::scoped_interpreter> DragPoser::interpreter_guard = nullptr;

EXPORT DragPoser* init_drag_poser()
{
	try
	{
		DragPoser* dragPoser = new DragPoser();
		dragPoser->create_run_drag_python_object("run_drag");
		return dragPoser;
	}
	catch (const py::error_already_set& e)
	{
		logMessage("Python error.");
		logMessage(e.what());
		std::cout << "Python error." << std::endl;
		std::cout << e.what() << std::endl;
	}
}

EXPORT void set_reference_skeleton(DragPoser* dragPoser, char* bvhPath)
{
	py::object instance = dragPoser->get_run_drag_python_object();
	py::object result = instance.attr("set_reference_skeleton")(std::string(bvhPath));
	int nJoints = result.cast<int>();
	dragPoser->set_num_joints(nJoints);
}

EXPORT void load_models(DragPoser* dragPoser, char* modelPath)
{
	py::object instance = dragPoser->get_run_drag_python_object();
	instance.attr("load_models")(std::string(modelPath));
}

EXPORT void set_mask_and_weights(DragPoser* dragPoser, float* mask, float2* weights)
{
	py::object instance = dragPoser->get_run_drag_python_object();
	int nJoints = dragPoser->get_num_joints();
	py::array_t<float> maskNumpy = create_numpy_array_float(mask, nJoints);
	py::array_t<float> weightsNumpy = create_numpy_array_float2(weights, nJoints);
	py::object result = instance.attr("set_mask_and_weights")(maskNumpy, weightsNumpy);
	int nEndEffectors = result.cast<int>();
	dragPoser->set_num_endeffectors(nEndEffectors);
}

EXPORT void init_drag_model(DragPoser* dragPoser, float3 initialGlobalPos, quaternion initialGlobalRot)
{
	py::object instance = dragPoser->get_run_drag_python_object();
	py::array_t<float> initialGlobalPosNumpy = create_numpy_array_float3(&initialGlobalPos, 1);
	py::array_t<float> initialGlobalRotNumpy = create_numpy_array_quaternion(&initialGlobalRot, 1);
	instance.attr("init_drag_pose")(initialGlobalPosNumpy, initialGlobalRotNumpy);
}

EXPORT void set_optim_params(DragPoser* dragPoser, float stopEpsPos, float stopEpsRot, int maxIter, float lr)
{
	py::object instance = dragPoser->get_run_drag_python_object();
	instance.attr("set_optim_params")(stopEpsPos, stopEpsRot, maxIter, lr);
}

EXPORT void set_lambdas(DragPoser* dragPoser, float lambdaRot, float lambdaTemporal, int temporalFutureWindow)
{
	py::object instance = dragPoser->get_run_drag_python_object();
	instance.attr("set_lambdas")(lambdaRot, lambdaTemporal, temporalFutureWindow);
}

EXPORT void set_global_pos(DragPoser* dragPoser, float3 globalPos)
{
	py::object instance = dragPoser->get_run_drag_python_object();
	py::array_t<float> globalPosNumpy = create_numpy_array_float3(&globalPos, 1);
	instance.attr("set_global_pos")(globalPosNumpy);
}

EXPORT void drag_pose(DragPoser* dragPoser, int nEndEffectors, float3* targetEEPos, quaternion* targetEERot, quaternion* resultPose, float3* resultGlobalPos)
{
	assert(nEndEffectors == dragPoser->get_num_endeffectors());

	py::object instance = dragPoser->get_run_drag_python_object();
	py::array_t<float> targetEEPosNumpy = create_numpy_array_float3(targetEEPos, nEndEffectors);
	py::array_t<float> targetEERotNumpy = create_numpy_array_quaternion(targetEERot, nEndEffectors);
	py::array_t<float> resultPoseNumpy = create_numpy_array_quaternion(resultPose, dragPoser->get_num_joints());
	py::array_t<float> resultGlobalPosNumpy = create_numpy_array_float3(resultGlobalPos, 1);

	try
	{
		instance.attr("drag_pose")(targetEEPosNumpy, targetEERotNumpy, resultPoseNumpy, resultGlobalPosNumpy);
	}
	catch (const py::error_already_set& e)
	{
		logMessage("Python error.");
		logMessage(e.what());
		std::cout << "Python error." << std::endl;
		std::cout << e.what() << std::endl;
	}
}

EXPORT void destroy_drag_poser(DragPoser* dragPoser)
{
	delete dragPoser;
}
