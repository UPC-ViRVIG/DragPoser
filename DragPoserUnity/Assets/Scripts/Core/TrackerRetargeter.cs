using BVH;
using System.Collections;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

public class TrackerRetargeter : MonoBehaviour
{
    [Header("Settings")]
    public bool ResetOrientation = false;
    public bool DebugDraw = false;

    [Header("References")]
    public GameObject TrackerPrefab;
    public TextAsset RetargetTPose;

    [Header("Skeleton Mapping")]
    public BoneMap[] BoneMapping;

    [Header("BVH Reference Vectors")]
    // Target vectors are forward, up and right as defined in Unity
    // Here the user defines the reference vectors in the BVH file
    public float3 BVHForwardLocalVector = new float3(0, 0, 1);
    public float3 BVHUpLocalVector = new float3(0, 1, 0);

    // Private ---------------------------------------------------------
    public quaternion RootAlign { get; private set; }
    private quaternion InverseRootAlign;
    private quaternion[] SourceTPose;
    private quaternion[] InverseTargetTPose;
    private Transform[] Trackers;
    private float3[] RetargetedPositions;
    private quaternion[] RetargetedRotations;

    private void Awake()
    {
        for (int i = 0; i < BodyJoints.Length; i++)
        {
            BodyJointsToIndex[BodyJoints[i]] = i;
        }

        Calibrate();
    }

    private void OnEnable()
    {
        UpdateManager.Instance.OnRetargetTrackers += OnRetargetTrackers;
    }

    private void OnDisable()
    {
        UpdateManager.Instance.OnRetargetTrackers -= OnRetargetTrackers;
    }

    public Transform GetTracker(HumanBodyBones bone)
    {
        int index = BodyJointsToIndex[bone];
        return GetTracker(index);
    }

    public Transform GetTracker(int boneIndex)
    {
        return Trackers[boneIndex];
    }

    public void GetRetarget(HumanBodyBones bone, out float3 retPos, out quaternion retRot)
    {
        int index = BodyJointsToIndex[bone];
        GetRetarget(index, out retPos, out retRot);
    }

    public void GetRetarget(int boneIndex, out float3 retPos, out quaternion retRot)
    {
        retPos = RetargetedPositions[boneIndex];
        retRot = RetargetedRotations[boneIndex];
    }

    // Given a position and rotation in world space, retarget to the retarget reference pose
    private void OnRetargetTrackers()
    {
        float3 rootPos = Trackers[0].position;
        for (int i = 0; i < Trackers.Length; ++i)
        {
            Transform tracker = Trackers[i];
            float3 pos = tracker.position;
            quaternion rot = tracker.rotation;
            // Retarget Position
            float3 retPos = math.mul(RootAlign, pos - rootPos) + rootPos;
            // Retarget Rotations -> [source local] to [source world]
            // [source local] -> [source world] -> [source tpose world] -> [target tpose world] -> [target world] -> [source world]
            quaternion retRot = math.mul(RootAlign, math.mul(math.mul(InverseTargetTPose[i], rot), math.mul(InverseRootAlign, SourceTPose[i])));
            // Apply Retarget
            RetargetedPositions[i] = retPos;
            RetargetedRotations[i] = retRot;
        }
    }

    private void ComputeRootAlign(quaternion worldRootRot)
    {
        float3 targetWorldForward = math.forward();
        float3 targetWorldUp = math.up();
        float3 sourceWorldForward = math.mul(worldRootRot, BVHForwardLocalVector);
        float3 sourceWorldUp = math.mul(worldRootRot, BVHUpLocalVector);
        quaternion targetLookAt = quaternion.LookRotation(targetWorldForward, targetWorldUp);
        quaternion sourceLookAt = quaternion.LookRotation(sourceWorldForward, sourceWorldUp);
        // RootAlign -> [target tpose world] to [source tpose world]
        RootAlign = math.mul(sourceLookAt, math.inverse(targetLookAt));
        InverseRootAlign = math.inverse(RootAlign);
    }

    private void CreateTrackers(BVHAnimation tpose)
    {
        Skeleton skeleton = tpose.Skeleton;
        Trackers = new Transform[BodyJoints.Length];
        int i = 0;
        foreach (HumanBodyBones bone in BodyJoints)
        {
            Vector3 pos = Vector3.zero;
            Quaternion rot = Quaternion.identity;
            if (UnityToName(bone, out string jointName) && skeleton.Find(jointName, out Skeleton.Joint joint))
            {
                tpose.GetWorldPositionAndRotation(joint, 0, out quaternion worldRot, out float3 worldPos);
                pos = math.mul(RootAlign, worldPos);
                if (!ResetOrientation) rot = math.mul(RootAlign, worldRot);
            }
            else Debug.Assert(false, "Joint not found: " + bone.ToString());
            GameObject tracker = Instantiate(TrackerPrefab, pos, rot, transform);
            tracker.name = bone.ToString();
            Trackers[i] = tracker.transform;
            i += 1;
        }
        SourceTPose = new quaternion[BodyJoints.Length];
        InverseTargetTPose = new quaternion[BodyJoints.Length];
        RetargetedPositions = new float3[BodyJoints.Length];
        RetargetedRotations = new quaternion[BodyJoints.Length];
        UpdateDebugVisual();
    }

    private void UpdateDebugVisual()
    {
        foreach(Transform tracker in Trackers)
        {
            for (int i = 0; i < tracker.childCount; ++i)
            {
                tracker.GetChild(i).gameObject.SetActive(DebugDraw);
            }
        }
    }

    private void ComputeJointAlign(BVHAnimation tpose)
    {
        Skeleton skeleton = tpose.Skeleton;
        int i = 0;
        foreach (HumanBodyBones bone in BodyJoints)
        {
            if (UnityToName(bone, out string jointName) && skeleton.Find(jointName, out Skeleton.Joint joint))
            {
                quaternion worldRot = tpose.GetWorldRotation(joint, 0);

                InverseTargetTPose[i] = math.inverse(Trackers[i].rotation);
                SourceTPose[i] = worldRot;
            }
            else Debug.Assert(false, "Joint not found: " + bone.ToString());

            i += 1;
        }
    }

    // Align current reference pose T-Pose to retarget reference pose T-Pose
    private void Calibrate()
    {
        // Import BVH
        BVHImporter importer = new BVHImporter();
        BVHAnimation tposeAnimation = importer.Import(RetargetTPose, 1.0f, true);

        // Root Align
        UnityToName(HumanBodyBones.Hips, out string hipsJointName);
        tposeAnimation.Skeleton.Find(hipsJointName, out Skeleton.Joint hipsJoint);
        ComputeRootAlign(tposeAnimation.GetWorldRotation(hipsJoint, 0));

        // Create Trackers
        CreateTrackers(tposeAnimation);

        // Compute Joint Alignments
        ComputeJointAlign(tposeAnimation);
    }


    // Used for retargeting. First parent, then children
    private HumanBodyBones[] BodyJoints =
    {
            HumanBodyBones.Hips, // 0

            HumanBodyBones.Spine, // 1
            HumanBodyBones.Chest, // 2
            HumanBodyBones.UpperChest, // 3

            HumanBodyBones.Neck, // 4
            HumanBodyBones.Head, // 5

            HumanBodyBones.LeftShoulder, // 6
            HumanBodyBones.LeftUpperArm, // 7
            HumanBodyBones.LeftLowerArm, // 8
            HumanBodyBones.LeftHand, // 9

            HumanBodyBones.RightShoulder, // 10
            HumanBodyBones.RightUpperArm, // 11
            HumanBodyBones.RightLowerArm, // 12
            HumanBodyBones.RightHand, // 13

            HumanBodyBones.LeftUpperLeg, // 14
            HumanBodyBones.LeftLowerLeg, // 15
            HumanBodyBones.LeftFoot, // 16
            HumanBodyBones.LeftToes, // 17

            HumanBodyBones.RightUpperLeg, // 18
            HumanBodyBones.RightLowerLeg, // 19
            HumanBodyBones.RightFoot, // 20
            HumanBodyBones.RightToes // 21
    };
    private Dictionary<HumanBodyBones, int> BodyJointsToIndex = new Dictionary<HumanBodyBones, int>();

    public bool UnityToName(HumanBodyBones humanBodyBone, out string name)
    {
        name = "";
        for (int i = 0; i < BoneMapping.Length; ++i)
        {
            if (BoneMapping[i].Unity == humanBodyBone)
            {
                name = BoneMapping[i].Name;
                return true;
            }
        }
        Debug.Assert(false, "Not found");
        return false;
    }

    [System.Serializable]
    public struct BoneMap
    {
        public string Name;
        public HumanBodyBones Unity;
    }

#if UNITY_EDITOR
    private void OnValidate()
    {
        if (RetargetTPose != null && (BoneMapping == null || BoneMapping.Length != 22))
        {
            BVHImporter importer = new BVHImporter();
            BVHAnimation tpose = importer.Import(RetargetTPose, 1.0f, true);
            BoneMapping = new BoneMap[tpose.Skeleton.Joints.Count];
            for (int i = 0; i < BoneMapping.Length; ++i)
            {
                BoneMapping[i].Name = tpose.Skeleton.Joints[i].Name;
            }
        }
        if (Trackers != null) UpdateDebugVisual();
    }

    private void OnDrawGizmosSelected()
    {
        if (Trackers != null)
        {
            for (int i = 0; i < RetargetedPositions.Length; ++i)
            {
                Gizmos.color = Color.yellow;
                Gizmos.DrawSphere(RetargetedPositions[i], 0.02f);
                float3 forward = math.mul(RetargetedRotations[i], math.forward());
                float3 up = math.mul(RetargetedRotations[i], math.up());
                float3 right = math.mul(RetargetedRotations[i], math.right());
                Gizmos.color = Color.blue;
                Gizmos.DrawLine(RetargetedPositions[i], RetargetedPositions[i] + forward * 0.1f);
                Gizmos.color = Color.green;
                Gizmos.DrawLine(RetargetedPositions[i], RetargetedPositions[i] + up * 0.1f);
                Gizmos.color = Color.red;
                Gizmos.DrawLine(RetargetedPositions[i], RetargetedPositions[i] + right * 0.1f);

                if (i > 0 && i != 6 && i != 10 && i != 14 && i != 18)
                {
                    float3 parentPos = RetargetedPositions[i - 1];
                    Gizmos.color = Color.yellow;
                    Gizmos.DrawLine(parentPos, RetargetedPositions[i]);
                }
            }
        }
    }
#endif
}
