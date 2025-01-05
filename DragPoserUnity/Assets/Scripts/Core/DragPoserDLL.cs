using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Unity.Mathematics;
using UnityEngine;

public class DragPoserDLL : IDisposable
{
    [DllImport("DragPoserDLL")]
    private static extern IntPtr init_drag_poser();
    [DllImport("DragPoserDLL")]
    private static extern void set_reference_skeleton(IntPtr dragPoser, string bvhPath);
    [DllImport("DragPoserDLL")]
    private static extern void load_models(IntPtr dragPoser, string modelPath);
    [DllImport("DragPoserDLL")]
    private static extern void set_mask_and_weights(IntPtr dragPoser, IntPtr mask, IntPtr weights);
    [DllImport("DragPoserDLL")]
    private static extern void init_drag_model(IntPtr dragPoser, float3 initialGlobalPos, quaternion initialGlobalRot);
    [DllImport("DragPoserDLL")]
    private static extern void set_optim_params(IntPtr dragPoser, float stopEpsPos, float stopEpsRot, int maxIter, float lr);
    [DllImport("DragPoserDLL")]
    private static extern void set_lambdas(IntPtr dragPoser, float lambdaRot, float lambdaTemporal, int temporalFutureWindow);
    [DllImport("DragPoserDLL")]
    private static extern void set_global_pos(IntPtr intPtr, float3 globalPos);
    [DllImport("DragPoserDLL")]
    private static extern float drag_pose(IntPtr dragPoser, int nEndEffectors, IntPtr targetEEPos, IntPtr targetEERot, IntPtr resultPose, IntPtr resultGlobalPos);
    [DllImport("DragPoserDLL")]
    private static extern void destroy_drag_poser(IntPtr dragPoser);


    private IntPtr DragPoserPtr;

    private IntPtr MaskPtr;
    private GCHandle MaskHandler;
    private IntPtr WeightsPtr;
    private GCHandle WeightsHandler;
    private IntPtr TargetEEPosPtr;
    private GCHandle TargetEEPosHandler;
    private IntPtr TargetEERotPtr;
    private GCHandle TargetEERotHandler;
    private IntPtr ResultPosePtr;
    private GCHandle ResultPoseHandler;
    private IntPtr ResultGlobalPosPtr;
    private GCHandle ResultGlobalPosHandler;

    private int NumEndEffectors;
    private bool disposed = false;

    public DragPoserDLL()
    {
        DragPoserPtr = init_drag_poser();
    }

    public void DragPose()
    {
        Debug.Assert(TargetEEPosHandler.IsAllocated, "Target EE Pos must be allocated");
        Debug.Assert(TargetEERotHandler.IsAllocated, "Target EE Rot must be allocated");
        Debug.Assert(ResultPoseHandler.IsAllocated, "Result Pose must be allocated");
        Debug.Assert(ResultGlobalPosHandler.IsAllocated, "Result Global Pos must be allocated");

        drag_pose(DragPoserPtr, NumEndEffectors, TargetEEPosPtr, TargetEERotPtr, ResultPosePtr, ResultGlobalPosPtr);
    }

    public void SetGlobalPosition(float3 globalPos)
    {
        set_global_pos(DragPoserPtr, globalPos);
    }

    public void SetReferenceSkeleton(string bvhPath)
    {
        set_reference_skeleton(DragPoserPtr, bvhPath);
    }

    public void LoadModels(string modelPath)
    {
        load_models(DragPoserPtr, modelPath);
    }

    public void InitDragModel(float3 initialGlobalPos, quaternion initialGlobalRot)
    {
        init_drag_model(DragPoserPtr, initialGlobalPos, initialGlobalRot);
    }

    public void SetOptimParams(float stopEpsPos, float stopEpsRot, int maxIter, float lr)
    {
        set_optim_params(DragPoserPtr, stopEpsPos, stopEpsRot, maxIter, lr);
    }

    public void SetLambdas(float lambdaRot, float lambdaTemporal, int temporalFutureWindow)
    {
        set_lambdas(DragPoserPtr, lambdaRot, lambdaTemporal, temporalFutureWindow);
    }

    public void SetMaskAndWeightsBuffers(float[] mask, float2[] weights)
    {
        Debug.Assert(mask.Length == weights.Length, "Mask and weights must be the same length");

        if (MaskHandler.IsAllocated)
        {
            MaskHandler.Free();
        }
        if (WeightsHandler.IsAllocated)
        {
            WeightsHandler.Free();
        }

        MaskHandler = GCHandle.Alloc(mask, GCHandleType.Pinned);
        MaskPtr = MaskHandler.AddrOfPinnedObject();
        WeightsHandler = GCHandle.Alloc(weights, GCHandleType.Pinned);
        WeightsPtr = WeightsHandler.AddrOfPinnedObject();
    }

    public void UpdateMaskAndWeights()
    {
        set_mask_and_weights(DragPoserPtr, MaskPtr, WeightsPtr);
    }

    public void SetTargetEEBuffers(float3[] targetEEPos, quaternion[] targetEERot)
    {
        Debug.Assert(targetEEPos.Length == targetEERot.Length, "Target EE Pos and Rot must be the same length");

        if (TargetEEPosHandler.IsAllocated)
        {
            TargetEEPosHandler.Free();
        }
        if (TargetEERotHandler.IsAllocated)
        {
            TargetEERotHandler.Free();
        }

        TargetEEPosHandler = GCHandle.Alloc(targetEEPos, GCHandleType.Pinned);
        TargetEEPosPtr = TargetEEPosHandler.AddrOfPinnedObject();
        TargetEERotHandler = GCHandle.Alloc(targetEERot, GCHandleType.Pinned);
        TargetEERotPtr = TargetEERotHandler.AddrOfPinnedObject();

        NumEndEffectors = targetEEPos.Length;
    }

    public void SetResultBuffers(quaternion[] resultPose, float3[] resultGlobalPos)
    {
        Debug.Assert(resultGlobalPos.Length == 1, "Result Global Pos must be length 1");

        if (ResultPoseHandler.IsAllocated)
        {
            ResultPoseHandler.Free();
        }
        if (ResultGlobalPosHandler.IsAllocated)
        {
            ResultGlobalPosHandler.Free();
        }

        ResultPoseHandler = GCHandle.Alloc(resultPose, GCHandleType.Pinned);
        ResultPosePtr = ResultPoseHandler.AddrOfPinnedObject();
        ResultGlobalPosHandler = GCHandle.Alloc(resultGlobalPos, GCHandleType.Pinned);
        ResultGlobalPosPtr = ResultGlobalPosHandler.AddrOfPinnedObject();
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!disposed)
        {
            if (disposing)
            {
                // Dispose managed resources
            }

            if (MaskHandler.IsAllocated)
            {
                MaskHandler.Free();
            }
            if (WeightsHandler.IsAllocated)
            {
                WeightsHandler.Free();
            }
            if (TargetEEPosHandler.IsAllocated)
            {
                TargetEEPosHandler.Free();
            }
            if (TargetEERotHandler.IsAllocated)
            {
                TargetEERotHandler.Free();
            }
            if (ResultPoseHandler.IsAllocated)
            {
                ResultPoseHandler.Free();
            }
            if (ResultGlobalPosHandler.IsAllocated)
            {
                ResultGlobalPosHandler.Free();
            }

            destroy_drag_poser(DragPoserPtr);

            disposed = true;
        }
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    ~DragPoserDLL()
    {
        Dispose(false);
    }
}
