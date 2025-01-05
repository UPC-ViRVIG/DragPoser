using BVH;
using System.Collections;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

public class SkeletonAvatar : MonoBehaviour
{
    public TrackerRetargeter TrackerRetargeter;
    public DragPoser DragPoser;
    public Color Color = Color.yellow;
    public float Smoothness = 0.7f;

    public Transform[] SkeletonTransforms { get; private set; }

    private List<Material> Materials = new List<Material>();

    private void OnEnable()
    {
        UpdateManager.Instance.OnCharacterUpdated += OnCharacterUpdated;
    }

    private void OnDisable()
    {
        UpdateManager.Instance.OnCharacterUpdated -= OnCharacterUpdated;
    }

    private void Start()
    {
        InitSkeleton();
    }

    private void InitSkeleton()
    {
        // Create Skeleton
        BVHImporter importer = new BVHImporter();
        BVHAnimation tpose = importer.Import(TrackerRetargeter.RetargetTPose, 1.0f, true);
        SkeletonTransforms = new Transform[tpose.Skeleton.Joints.Count];
        for (int j = 0; j < SkeletonTransforms.Length; j++)
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
            // Visual
            Transform visual = (new GameObject()).transform;
            visual.name = "Visual";
            visual.SetParent(t, false);
            visual.localScale = new Vector3(0.1f, 0.1f, 0.1f);
            visual.localPosition = Vector3.zero;
            visual.localRotation = Quaternion.identity;
            // Sphere
            Transform sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere).transform;
            sphere.name = "Sphere";
            sphere.SetParent(visual, false);
            sphere.localScale = Vector3.one;
            sphere.localPosition = Vector3.zero;
            sphere.localRotation = Quaternion.identity;
            Materials.Add(sphere.GetComponent<MeshRenderer>().material);
            Materials[^1].color = Color;
            Materials[^1].SetFloat("_Glossiness", Smoothness);
            // Capsule
            Transform capsule = GameObject.CreatePrimitive(PrimitiveType.Capsule).transform;
            capsule.name = "Capsule";
            capsule.SetParent(SkeletonTransforms[joint.ParentIndex].Find("Visual"), false);
            float distance = Vector3.Distance(t.position, t.parent.position) * (1.0f / visual.localScale.y) * 0.5f;
            capsule.localScale = new Vector3(0.5f, distance, 0.5f);
            Vector3 up = (t.position - t.parent.position).normalized;
            if (up.magnitude < 0.0001f)
            {
                continue;
            }
            capsule.localPosition = t.parent.InverseTransformDirection(up) * distance;
            capsule.localRotation = Quaternion.Inverse(t.parent.rotation) * Quaternion.LookRotation(new Vector3(-up.y, up.x, 0.0f), up);
            Materials.Add(capsule.GetComponent<MeshRenderer>().material);
            Materials[^1].color = Color;
            Materials[^1].SetFloat("_Glossiness", Smoothness);
        }
    }

    private void OnCharacterUpdated()
    {
        Quaternion hipsCorrection = TrackerRetargeter.RootAlign;
        for (int i = 0; i < SkeletonTransforms.Length; i++)
        {
            SkeletonTransforms[i].localPosition = DragPoser.SkeletonTransforms[i].localPosition;
            SkeletonTransforms[i].rotation = Quaternion.Inverse(hipsCorrection) * DragPoser.SkeletonTransforms[i].rotation;
        }
        SkeletonTransforms[0].position = DragPoser.SkeletonTransforms[0].position;
    }

    public void SetRootPosition(Vector3 pos)
    {
        SkeletonTransforms[0].position = pos;
    }

    private void OnDestroy()
    {
        if (Materials != null)
        {
            foreach (Material material in Materials)
            {
                Destroy(material);
            }
        }
    }
}
