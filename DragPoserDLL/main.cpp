#include "utils.h"
#include "exportFunc.h"

namespace py = pybind11;

int main() {

    try
    {
        for (int i = 0; i < 3; ++i)
        {
            DragPoser* dragPoser = init_drag_poser();

            set_reference_skeleton(dragPoser, "C:\\Users\\user\\Desktop\\Projects\\DragPoser\\DragPoserUnity\\Assets\\Data\\S01_A02.txt");
            load_models(dragPoser, "C:\\Users\\user\\Desktop\\Projects\\DragPoser\\python\\models\\model_consecutive_xsens\\");

            float mask[] = { 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1 };
            float2 weights[] = { float2(10, 10), float2(1, 0.01), float2(1, 0.01), float2(5, 0.01), float2(1, 0.01), float2(1, 0.01), float2(1, 0.01), float2(5, 0.01),
                                 float2(1, 0.01), float2(1, 0.01), float2(1, 0.01), float2(1, 0.01), float2(1, 0.01), float2(5, 0.01), float2(1, 0.01), float2(1, 0.01),
                                 float2(1, 0.01), float2(5, 0.01), float2(1, 0.01), float2(1, 0.01), float2(1, 0.01), float2(5, 0.01) };
            set_mask_and_weights(dragPoser, mask, weights);

            float3 initialGlobalPos = float3(-2.6648, 0.9977, 3.7518);
            quaternion initialGlobalRot = quaternion(0.6381, 0.0078, -0.7698, 0.0110);

            init_drag_model(dragPoser, initialGlobalPos, initialGlobalRot);

            set_optim_params(dragPoser, 0.01 * 0.01, 0.01, 10, 0.01);
            set_lambdas(dragPoser, 1, 0.02, 60);

            float3 targetEEPos[] = { float3(0, 0, 0), float3(0.0953, -0.8287, 0.0786), float3(-0.0835, -0.8810, 0.0721), float3(-0.0032, 0.6362, 0.0252), float3(-0.1830, -0.0721, 0.1874), float3(-0.0642, -0.0306, -0.2561) };
            quaternion targetEERot[] = { quaternion(-0.6381, -0.0078, 0.7698, -0.0110), quaternion(0.7759, 0.2940, -0.5198, 0.2034), quaternion(-0.3807, -0.0448, 0.9235, -0.0174), quaternion(-0.5807, 0.0365, 0.8109, 0.0632), quaternion(-0.2944, -0.4933, 0.6859, 0.4468), quaternion(0.6386, -0.5011, -0.3526, 0.4655) };

            quaternion resultPose[22];
            float3 resultGlobalPos[1];

            const int nEndEffectors = 6;
            drag_pose(dragPoser, nEndEffectors, targetEEPos, targetEERot, resultPose, resultGlobalPos);

            destroy_drag_poser(dragPoser);
        }
    }
    catch (const py::error_already_set& e)
    {
        std::cout << "Python error." << std::endl;
		std::cout << e.what() << std::endl;
    }

    return 0;
}
