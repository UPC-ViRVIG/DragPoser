using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;

public class FBIK : MonoBehaviour
{
    public GameObject EndEffectorPrefab;
    public DragPoser DragPoser;
    public SkeletonAvatar SkeletonAvatar;
    public TrackerRetargeter Trackers;
    public Color MinColor;
    public Color MaxColor;
    public UnityEvent<Transform[], HumanBodyBones[]> OnAfterEndEffectorsUpdated;

    private bool IsCreated;
    private Transform[] EndEffectors;
    private bool[] Active;
    private float[] Weights;
    private Material[] Materials;


    private void OnEnable()
    {
        UpdateManager.Instance.OnBeforeRetargetTrackers += OnBeforeRetargetTrackers;
        UpdateManager.Instance.OnAfterCharacterUpdated += OnAfterCharacterUpdated;
    }

    private void OnDisable()
    {
        UpdateManager.Instance.OnBeforeRetargetTrackers -= OnBeforeRetargetTrackers;
        UpdateManager.Instance.OnAfterCharacterUpdated -= OnAfterCharacterUpdated;
    }

    private void OnBeforeRetargetTrackers()
    {
        if (!IsCreated) return;

        // Update End Effector Positions
        for (int i = 0; i < EndEffectors.Length; i++)
        {
            Transform ee = EndEffectors[i];
            if (!Active[i])
            {
                if (ee.gameObject.activeSelf) ee.gameObject.SetActive(false);
                Transform t = SkeletonAvatar.SkeletonTransforms[i];
                EndEffectors[i].SetPositionAndRotation(t.position, t.rotation);
            }
            else
            {
                if (!ee.gameObject.activeSelf) ee.gameObject.SetActive(true);
                Transform tracker = Trackers.GetTracker(DragPoser.PythonSkeletonToUnity[i]);
                tracker.SetPositionAndRotation(ee.position, ee.rotation);
            }
        }
        if (OnAfterEndEffectorsUpdated != null)
        {
            OnAfterEndEffectorsUpdated.Invoke(EndEffectors, DragPoser.PythonSkeletonToUnity);
        }

        // Update Active and Weights
        for (int i = 0; i < Active.Length; ++i)
        {
            DragPoser.Mask[i] = Active[i] ? 1 : 0;
        }
        for (int i = 0; i < Weights.Length; ++i)
        {
            DragPoser.Weights[i].x = Weights[i];
        }
    }

    private void OnAfterCharacterUpdated()
    {
        if (!IsCreated)
        {
            Create();
            IsCreated = true;
        }
    }

    private void OnGUI()
    {
        // Dark Background
        GUI.Box(new Rect(0, 0, 225, 500), "");
        // Reset
        if (GUI.Button(new Rect(10, 10, 100, 20), "Reset"))
        {
            ResetMaskAndWeights();
        }
        // Mask and Weights
        for (int i = 0; i < EndEffectors.Length; i++)
        {
            Active[i] = GUI.Toggle(new Rect(10, 40 + i * 20, 100, 20), Active[i], EndEffectorNames[i]);
            Weights[i] = GUI.HorizontalSlider(new Rect(120, 40 + i * 20, 100, 20), Weights[i], 0.0f, 10.0f);
        }
        UpdateVisuals();
    }

    private void Create()
    {
        EndEffectors = new Transform[SkeletonAvatar.SkeletonTransforms.Length];
        Active = new bool[EndEffectors.Length];
        Weights = new float[EndEffectors.Length];
        Materials = new Material[EndEffectors.Length];
        for (int i = 0; i < EndEffectors.Length; i++)
        {
            Transform t = Trackers.GetTracker(DragPoser.PythonSkeletonToUnity[i]);
            GameObject ee = Instantiate(EndEffectorPrefab, t.position, t.rotation, transform);
            ee.name = t.name;
            ee.transform.localScale = Vector3.one;
            ee.transform.position = t.position;
            ee.transform.rotation = t.rotation;
            EndEffectors[i] = ee.transform;
            Materials[i] = ee.transform.GetChild(1).GetComponent<MeshRenderer>().material;
        }
        ResetMaskAndWeights();
    }

    private void ResetMaskAndWeights()
    {
        for (int i = 0; i < EndEffectors.Length; i++)
        {
            Active[i] = false;
        }
        Active[0] = true;
        Active[3] = true;
        Active[7] = true;
        Active[13] = true;
        Active[17] = true;
        Active[21] = true;
        for (int i = 0; i < Weights.Length; i++)
        {
            Weights[i] = 1.0f;
        }
        Weights[0] = 10.0f;
        Weights[3] = 5.0f;
        Weights[7] = 5.0f;
        Weights[13] = 5.0f;
        Weights[17] = 5.0f;
        Weights[21] = 5.0f;
        UpdateVisuals();
    }

    private void UpdateVisuals()
    {
        float maxWeight = 0.0f;
        for (int i = 0; i < EndEffectors.Length; i++)
        {
            if (Weights[i] > maxWeight)
            {
                maxWeight = Weights[i];
            }
        }
        for (int i = 0; i < EndEffectors.Length; i++)
        {
            float weight = Weights[i] / maxWeight;
            Materials[i].color = Color.Lerp(MinColor, MaxColor, weight * weight * weight);
        }
    }

    private string[] EndEffectorNames =
    {
        "Hips", // 0
        "LeftUpLeg",
        "LeftLowerLeg",
        "LeftFoot", // 3
        "LeftToe",
        "RightUpLeg",
        "RightLowerLeg",
        "RightFoot", // 7
        "RightToe",
        "Spine",
        "Chest",
        "UpperChest",
        "Neck",
        "Head", // 13
        "LeftShoulder",
        "LeftUpperArm",
        "LeftLowerArm",
        "LeftHand", // 17
        "RightShoulder",
        "RightUpperArm",
        "RightLowerArm",
        "RightHand", // 21
    };

    private void OnDestroy()
    {
        if (Materials != null)
        {
            for (int i = 0; i < Materials.Length; i++)
            {
                Destroy(Materials[i]);
            }
        }
    }
}
