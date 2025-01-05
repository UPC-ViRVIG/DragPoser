using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using BVH;
using Unity.Mathematics;

public class BVHPlayback : MonoBehaviour
{
    public TrackerRetargeter TrackerRetargeter;

    public int Frame = 0;
    public int TargetFramerate = 60;
    public TextAsset BVH;

    private BVHAnimation Animation;
    private bool IsPaused;

    private void Awake()
    {
        Application.targetFrameRate = TargetFramerate;
    }

    private void Start()
    {
        BVHImporter importer = new BVHImporter();
        Animation = importer.Import(BVH);
    }

    public void UpdateTrackers(Transform[] trackers, HumanBodyBones[] humanBodyBones)
    {
        if (!gameObject.activeSelf) return;

        int animationLength = Animation.Frames.Length;
        Skeleton sk = Animation.Skeleton;

        for (int i = 0; i < trackers.Length; ++i)
        {
            HumanBodyBones bone = humanBodyBones[i];
            if (TrackerRetargeter.UnityToName(bone, out string jointName) &&
                sk.Find(jointName, out Skeleton.Joint joint))
            {
                Animation.GetWorldPositionAndRotation(joint, Frame, out quaternion worldRot, out float3 worldPos);
                trackers[i].SetPositionAndRotation(worldPos, worldRot);
            }
        }

        if (!IsPaused) Frame = (Frame + 1) % animationLength;
    }

    private void OnGUI()
    {
        if (IsPaused)
        {
            if (GUI.Button(new Rect(240, 10, 100, 20), "Play"))
            {
                IsPaused = false;
            }
        }
        else
        {
            if (GUI.Button(new Rect(240, 10, 100, 20), "Pause"))
            {
                IsPaused = true;
            }
        }

        if (GUI.Button(new Rect(240, 40, 100, 20), "Reset"))
        {
            Frame = 0;
        }

        GUI.Label(new Rect(240, 70, 100, 20), "Frame: " + Frame);
    }


    private void OnDrawGizmos()
    {
        if (Animation == null) return;

        int updateFrame = Frame - 1;

        if (updateFrame < 0) return;

        Gizmos.color = Color.blue;
        foreach (Skeleton.Joint joint in Animation.Skeleton.Joints)
        {
            if (joint.Index == 0) continue;
            Animation.GetWorldPositionAndRotation(joint, updateFrame, out quaternion worldRot, out float3 worldPos);
            Skeleton.Joint parent = Animation.Skeleton.GetParent(joint);
            Animation.GetWorldPositionAndRotation(parent, updateFrame, out quaternion parentWorldRot, out float3 parentWorldPos);
            GizmosExtensions.DrawLine(worldPos, parentWorldPos, 3);
        }
    }
}
