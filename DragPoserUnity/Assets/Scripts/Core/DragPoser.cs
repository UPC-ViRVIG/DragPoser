using BVH;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.Mathematics;
using UnityEngine;

using Debug = UnityEngine.Debug;

public class DragPoser : MonoBehaviour
{
    [Header("References")]
    public TrackerRetargeter TrackerRetargeter;
    public SkeletonAvatar SkeletonAvatar;

    [Header("Settings")]
    public float RotationSmooth = 10;
    [Tooltip("Drag the avatar towards the adjustment joint to minimize global position errors")]
    public bool DoAdjustment = true;
    public int AdjustmentJoint = 0;
    [Range(0.0f, 2.0f)] public float AdjustmentWeightHalflife = 0.1f; // Time needed to move half of the distance

    [Header("Mask and Weights")]
    public float[] Mask;
    public float2[] Weights;

    [Header("Paths")]
    public string ReferenceSkeletonPath;
    public string ModelsPath;

    [Header("Parameters")]
    public float StopEpsPos = 0.01f * 0.01f;
    public float StopEpsRot = 0.01f;
    public int MaxIter = 10;
    public float LearningRate = 0.01f;
    public float LambdaRot = 1.0f;
    public float LambdaTemporal = 0.02f;
    public int TemporalFutureWindow = 60;


    public Transform[] SkeletonTransforms { get; private set; }
    public float3 CharacterRoot 
    { 
        get
        {
            return SkeletonAvatar.SkeletonTransforms[0].position;
        } 
    }

    private DragPoserDLL DragPoserDLL;

    private float3[] TargetEEPosBuffer;
    private quaternion[] TargetEERotBuffer;
    private quaternion[] ResultPoseBuffer;
    private float3[] ResultGlobalPosBuffer;

    private Quaternion[] TargetRotations;
    private Vector3 TargetRootPos;

    private quaternion[] PreviousEndEffectorsRotations;

    private void Awake()
    {
        // Init DLL
        DragPoserDLL = new DragPoserDLL();
        DragPoserDLL.SetReferenceSkeleton(ReferenceSkeletonPath);
        DragPoserDLL.LoadModels(ModelsPath);
        DragPoserDLL.SetMaskAndWeightsBuffers(Mask, Weights);
        DragPoserDLL.UpdateMaskAndWeights();
        DragPoserDLL.SetOptimParams(StopEpsPos, StopEpsRot, MaxIter, LearningRate);
        DragPoserDLL.SetLambdas(LambdaRot, LambdaTemporal, TemporalFutureWindow);

        // Import T-Pose BVH
        BVHImporter importer = new BVHImporter();
        BVHAnimation tpose = importer.Import(TrackerRetargeter.RetargetTPose, 1.0f, true);
        int nJoints = tpose.Skeleton.Joints.Count;

        // Buffers
        TargetEEPosBuffer = new float3[1];
        TargetEERotBuffer = new quaternion[1];
        ResultPoseBuffer = new quaternion[nJoints];
        ResultGlobalPosBuffer = new float3[1];
        PreviousEndEffectorsRotations = new quaternion[nJoints];
        for (int i = 0; i < PreviousEndEffectorsRotations.Length; ++i) PreviousEndEffectorsRotations[i] = quaternion.identity;
        DragPoserDLL.SetResultBuffers(ResultPoseBuffer, ResultGlobalPosBuffer);

        // Create Skeleton
        SkeletonTransforms = new Transform[tpose.Skeleton.Joints.Count];
        for (int j = 0; j < tpose.Skeleton.Joints.Count; j++)
        {
            // Joints
            Skeleton.Joint joint = tpose.Skeleton.Joints[j];
            Transform t = (new GameObject()).transform;
            t.name = joint.Name;
            t.SetParent(j == 0 ? transform : SkeletonTransforms[joint.ParentIndex], false);
            t.localPosition = joint.LocalOffset;
            tpose.GetWorldPositionAndRotation(joint, 0, out quaternion worldRot, out _);
            t.rotation = worldRot;
            SkeletonTransforms[j] = t;
        }
    }

    private void Start()
    {
        TargetRotations = new Quaternion[SkeletonTransforms.Length];
        for (int j = 0; j < SkeletonTransforms.Length; j++)
        {
            TargetRotations[j] = SkeletonTransforms[j].localRotation;
        }
    }

    private void OnEnable()
    {
        UpdateManager.Instance.OnDragPoser += OnDragPoser;
        UpdateManager.Instance.OnAfterRetargetTrackers += AfterRetargetTrackers;
    }

    private void OnDisable()
    {
        UpdateManager.Instance.OnDragPoser -= OnDragPoser;
    }

    private void AfterRetargetTrackers()
    {
        // Only execute once at the beginning
        UpdateManager.Instance.OnAfterRetargetTrackers -= AfterRetargetTrackers;

        TrackerRetargeter.GetRetarget(HumanBodyBones.Hips, out float3 retPos, out quaternion retRot);
        TargetRootPos = UnityToPython(retPos);

        SkeletonAvatar.SetRootPosition(TargetRootPos);

        // Init Drag Model
        DragPoserDLL.InitDragModel(TargetRootPos, UnityToPython(retRot));
    }

    private void OnDragPoser()
    {
        CheckAndUpdateBuffers();
        FillBuffers();
        ForwardDragPoser();
        UpdatePose();
        if (DoAdjustment) AdjustJoint();
        // Update Global Position
        DragPoserDLL.SetGlobalPosition(UnityToPython(SkeletonTransforms[0].position));
    }

    private void CheckAndUpdateBuffers()
    {
        int prevNumEndEffectors = TargetEEPosBuffer.Length;
        int currentNumEndEffectors = 0;
        for (int i = 0; i < Mask.Length; ++i)
        {
            if (Mask[i] > 0.1f)
            {
                ++currentNumEndEffectors;
                Mask[i] = 1.0f;
            }
            else
            {
                Mask[i] = 0.0f;
            }
        }
        if (prevNumEndEffectors != currentNumEndEffectors)
        {
            TargetEEPosBuffer = new float3[currentNumEndEffectors];
            TargetEERotBuffer = new quaternion[currentNumEndEffectors];
            DragPoserDLL.SetTargetEEBuffers(TargetEEPosBuffer, TargetEERotBuffer);
        }
        DragPoserDLL.UpdateMaskAndWeights();
        DragPoserDLL.SetOptimParams(StopEpsPos, StopEpsRot, MaxIter, LearningRate);
        DragPoserDLL.SetLambdas(LambdaRot, LambdaTemporal, TemporalFutureWindow);
    }

    private void FillBuffers()
    {
        // Fill Target EE Buffers
        int currentEEIndex = 0;
        for (int i = 0; i < Mask.Length; ++i)
        {
            if (Mask[i] > 0.1f)
            {
                TrackerRetargeter.GetRetarget(PythonSkeletonToUnity[i], out float3 retPos, out quaternion retRot);
                // Position
                TargetEEPosBuffer[currentEEIndex] = UnityToPython(retPos - CharacterRoot);
                // Rotation
                quaternion rot = UnityToPython(retRot);
                quaternion prevRot = PreviousEndEffectorsRotations[i];
                rot = EnsureContinuity(prevRot, rot);
                PreviousEndEffectorsRotations[i] = rot;
                TargetEERotBuffer[currentEEIndex] = rot;
                ++currentEEIndex;
            }
        }
    }

    private void ForwardDragPoser()
    {
        DragPoserDLL.DragPose();
    }

    private void AdjustJoint()
    {
        TrackerRetargeter.GetRetarget(PythonSkeletonToUnity[AdjustmentJoint], out float3 endEffector, out _);
        float3 joint = SkeletonTransforms[AdjustmentJoint].position;
        float3 difference = endEffector - joint;
        // Damp the difference using the adjustment halflife and dt
        float3 adjustment = MathExtensions.DampAdjustmentImplicit(difference, AdjustmentWeightHalflife, Time.deltaTime);
        // Move the character root towards the adjustment joint
        SkeletonTransforms[0].position += (Vector3)adjustment;
    }

    private void UpdatePose()
    {
        // From ResultBuffers to Targets
        for (int i = 0; i < SkeletonTransforms.Length; ++i)
        {
            quaternion rot = PythonToUnity(ResultPoseBuffer[i]);
            quaternion currentRot = SkeletonTransforms[i].localRotation;
            rot = EnsureContinuity(currentRot, rot);
            TargetRotations[i] = rot;
        }
        TargetRootPos = PythonToUnity(ResultGlobalPosBuffer[0]);

        // Update Pose and Root
        for (int i = 0; i < SkeletonTransforms.Length; ++i)
        {
            SkeletonTransforms[i].localRotation = Quaternion.Slerp(SkeletonTransforms[i].localRotation, TargetRotations[i], Time.deltaTime * RotationSmooth);
        }
        SkeletonTransforms[0].position = TargetRootPos;
    }

    private float3 UnityToPython(float3 unityPos)
    {
        // BVH's z+ axis is Unity's (z-) (Unity is left-handed BVH is right-handed)
        return new float3(unityPos.x, unityPos.y, -unityPos.z);
    }

    private float3 PythonToUnity(float3 pythonPos)
    {
        // BVH's z+ axis is Unity's (z-) (Unity is left-handed BVH is right-handed)
        return new float3(pythonPos.x, pythonPos.y, -pythonPos.z);
    }

    private quaternion UnityToPython(quaternion unityRot)
    {
        // right to left-handed negate imaginary part (x,y,z), negate z again because BVH's z+ is Unity's z-
        unityRot.value = new float4(-unityRot.value.x, -unityRot.value.y, unityRot.value.z, unityRot.value.w);
        unityRot = math.normalizesafe(unityRot);
        // Unity uses (x,y,z,w) and Python uses (w,x,y,z)
        quaternion pythonRot = new quaternion(unityRot.value.w, unityRot.value.x, unityRot.value.y, unityRot.value.z);
        return pythonRot;
    }

    private quaternion PythonToUnity(quaternion pythonRot)
    {
        // Unity uses (x,y,z,w) and Python uses (w,x,y,z)
        quaternion unityRot = new quaternion(pythonRot.value[1], pythonRot.value[2], pythonRot.value[3], pythonRot.value[0]);
        // right to left-handed negate imaginary part (x,y,z), negate z again because BVH's z+ is Unity's z-
        unityRot.value = new float4(-unityRot.value.x, -unityRot.value.y, unityRot.value.z, unityRot.value.w);
        unityRot = math.normalizesafe(unityRot);
        return unityRot;
    }

    private quaternion EnsureContinuity(quaternion currentRot, quaternion nextRot)
    {
        // if the distance (or, equivalently, maximal dot product) between the previous rotation
        // and the flipped current quaternion is smaller than
        // the distance between the previous rotation and the current quaternion, then flip the quaternion
        if (math.dot(currentRot, -nextRot.value) > math.dot(currentRot, nextRot.value))
        {
            nextRot.value = -nextRot.value;
        }
        return nextRot;
    }

    public static HumanBodyBones[] PythonSkeletonToUnity =
    {
        HumanBodyBones.Hips,
        HumanBodyBones.LeftUpperLeg,
        HumanBodyBones.LeftLowerLeg,
        HumanBodyBones.LeftFoot,
        HumanBodyBones.LeftToes,
        HumanBodyBones.RightUpperLeg,
        HumanBodyBones.RightLowerLeg,
        HumanBodyBones.RightFoot,
        HumanBodyBones.RightToes,
        HumanBodyBones.Spine,
        HumanBodyBones.Chest,
        HumanBodyBones.UpperChest,
        HumanBodyBones.Neck,
        HumanBodyBones.Head,
        HumanBodyBones.LeftShoulder,
        HumanBodyBones.LeftUpperArm,
        HumanBodyBones.LeftLowerArm,
        HumanBodyBones.LeftHand,
        HumanBodyBones.RightShoulder,
        HumanBodyBones.RightUpperArm,
        HumanBodyBones.RightLowerArm,
        HumanBodyBones.RightHand,
    };

    private void OnDestroy()
    {
        DragPoserDLL.Dispose();
    }

    private void OnApplicationQuit()
    {
        DragPoserDLL.Dispose();
    }

#if UNITY_EDITOR
    private void OnDrawGizmosSelected()
    {
        // Skeleton
        if (SkeletonTransforms == null) return;

        Gizmos.color = Color.red;
        for (int i = 1; i < SkeletonTransforms.Length; i++)
        {
            Transform t = SkeletonTransforms[i];
            GizmosExtensions.DrawLine(t.parent.position, t.position, 3);
        }
    }

    private void OnValidate()
    {
        if (TrackerRetargeter.RetargetTPose != null && (Mask == null || Mask.Length == 0))
        {
            // Import T-Pose BVH
            BVHImporter importer = new BVHImporter();
            BVHAnimation tpose = importer.Import(TrackerRetargeter.RetargetTPose, 1.0f, true);
            int nJoints = tpose.Skeleton.Joints.Count;

            // Buffers
            Mask = new float[nJoints];
            Weights = new float2[nJoints];
        }
    }
#endif
}
