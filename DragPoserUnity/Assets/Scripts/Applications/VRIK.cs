using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class VRIK : MonoBehaviour
{
    public GameObject EndEffectorPrefab;
    public DragPoser DragPoser;
    public TrackerRetargeter Trackers;
    public Color EnabledColor;
    public Color DisabledColor;

    [Header("Trackers")]
    public GameObject Hips;
    public GameObject LeftFoot;
    public GameObject RightFoot;
    public GameObject Head;
    public GameObject LeftHand;
    public GameObject RightHand;

    [Header("Active")]
    public bool HipsActive = true;
    public bool LeftFootActive = true;
    public bool RightFootActive = true;
    public bool HeadActive = true;
    public bool LeftHandActive = true;
    public bool RightHandActive = true;


    private Transform HipsEE;
    private Transform LeftFootEE;
    private Transform RightFootEE;
    private Transform HeadEE;
    private Transform LeftHandEE;
    private Transform RightHandEE;
    private Material HipsMat;
    private Material LeftFootMat;
    private Material RightFootMat;
    private Material HeadMat;
    private Material LeftHandMat;
    private Material RightHandMat;
    private bool IsCreated;
    private bool IsCalibrated;


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

    private void Update()
    {
        if (!IsCreated) return;
        UpdateVisuals();
    }

    private void OnBeforeRetargetTrackers()
    {
        if (!IsCreated || !IsCalibrated) return;

        // Update End Effector Positions
        if (HipsActive)
        {
            Transform c = Hips.transform.GetChild(0);
            HipsEE.SetPositionAndRotation(c.transform.position, c.transform.rotation);
        }
        if (LeftFootActive)
        {
            Transform c = LeftFoot.transform.GetChild(0);
            LeftFootEE.SetPositionAndRotation(c.transform.position, c.transform.rotation);
        }
        if (RightFootActive)
        {
            Transform c = RightFoot.transform.GetChild(0);
            RightFootEE.SetPositionAndRotation(c.transform.position, c.transform.rotation);
        }
        if (HeadActive)
        {
            Transform c = Head.transform.GetChild(0);
            HeadEE.SetPositionAndRotation(c.transform.position, c.transform.rotation);
        }
        if (LeftHandActive)
        {
            Transform c = LeftHand.transform.GetChild(0);
            LeftHandEE.SetPositionAndRotation(c.transform.position, c.transform.rotation);
        }
        if (RightHandActive)
        {
            Transform c = RightHand.transform.GetChild(0);
            RightHandEE.SetPositionAndRotation(c.transform.position, c.transform.rotation);
        }

        // Update Active and Weights
        DragPoser.Mask[0] = HipsActive ? 1 : 0;
        DragPoser.Mask[3] = LeftFootActive ? 1 : 0;
        DragPoser.Mask[7] = RightFootActive ? 1 : 0;
        DragPoser.Mask[13] = HeadActive ? 1 : 0;
        DragPoser.Mask[17] = LeftHandActive ? 1 : 0;
        DragPoser.Mask[21] = RightHandActive ? 1 : 0;
        DragPoser.Weights[0].x = 10.0f;
        DragPoser.Weights[3].x = 5.0f;
        DragPoser.Weights[7].x = 5.0f;
        DragPoser.Weights[13].x = 5.0f;
        DragPoser.Weights[17].x = 5.0f;
        DragPoser.Weights[21].x = 5.0f;
    }

    private void OnAfterCharacterUpdated()
    {
        if (!IsCreated)
        {
            Create();
            IsCreated = true;
        }
    }

    private void Create()
    {
        {
            HipsEE = Trackers.GetTracker(HumanBodyBones.Hips);
            GameObject ee = Instantiate(EndEffectorPrefab, Vector3.zero, Quaternion.identity);
            ee.name = "Hips";
            ee.transform.SetParent(HipsEE.transform, false);
            HipsMat = ee.transform.GetChild(1).GetComponent<MeshRenderer>().material;
        }
        {
            LeftFootEE = Trackers.GetTracker(HumanBodyBones.LeftFoot);
            GameObject ee = Instantiate(EndEffectorPrefab, Vector3.zero, Quaternion.identity);
            ee.name = "LeftFoot";
            ee.transform.SetParent(LeftFootEE.transform, false);
            LeftFootMat = ee.transform.GetChild(1).GetComponent<MeshRenderer>().material;
        }
        {
            RightFootEE = Trackers.GetTracker(HumanBodyBones.RightFoot);
            GameObject ee = Instantiate(EndEffectorPrefab, Vector3.zero, Quaternion.identity);
            ee.name = "RightFoot";
            ee.transform.SetParent(RightFootEE.transform, false);
            RightFootMat = ee.transform.GetChild(1).GetComponent<MeshRenderer>().material;
        }
        {
            HeadEE = Trackers.GetTracker(HumanBodyBones.Head);
            GameObject ee = Instantiate(EndEffectorPrefab, Vector3.zero, Quaternion.identity);
            ee.name = "Head";
            ee.transform.SetParent(HeadEE.transform, false);
            HeadMat = ee.transform.GetChild(1).GetComponent<MeshRenderer>().material;
        }
        {
            LeftHandEE = Trackers.GetTracker(HumanBodyBones.LeftHand);
            GameObject ee = Instantiate(EndEffectorPrefab, Vector3.zero, Quaternion.identity);
            ee.name = "LeftHand";
            ee.transform.SetParent(LeftHandEE.transform, false);
            LeftHandMat = ee.transform.GetChild(1).GetComponent<MeshRenderer>().material;
        }
        {
            RightHandEE = Trackers.GetTracker(HumanBodyBones.RightHand);
            GameObject ee = Instantiate(EndEffectorPrefab, Vector3.zero, Quaternion.identity);
            ee.name = "RightHand";
            ee.transform.SetParent(RightHandEE.transform, false);
            RightHandMat = ee.transform.GetChild(1).GetComponent<MeshRenderer>().material;
        }

        UpdateVisuals();
    }

    [ContextMenu("Calibrate()")]
    private void Calibrate()
    {
        {
            Transform child = new GameObject("Hips").transform;
            child.SetParent(Hips.transform);
            child.SetLocalPositionAndRotation(Vector3.zero, HipsEE.rotation * Quaternion.Inverse(Hips.transform.rotation));
        }
        {
            Transform child = new GameObject("LeftFoot").transform;
            child.SetParent(LeftFoot.transform);
            child.SetLocalPositionAndRotation(Vector3.zero, LeftFootEE.rotation * Quaternion.Inverse(LeftFoot.transform.rotation));
        }
        {
            Transform child = new GameObject("RightFoot").transform;
            child.SetParent(RightFoot.transform);
            child.SetLocalPositionAndRotation(Vector3.zero, RightFootEE.rotation * Quaternion.Inverse(RightFoot.transform.rotation));
        }
        {
            Transform child = new GameObject("Head").transform;
            child.SetParent(Head.transform);
            child.SetLocalPositionAndRotation(Vector3.zero, HeadEE.rotation * Quaternion.Inverse(Head.transform.rotation));
        }
        {
            Transform child = new GameObject("LeftHand").transform;
            child.SetParent(LeftHand.transform);
            child.SetLocalPositionAndRotation(Vector3.zero, LeftHandEE.rotation * Quaternion.Inverse(LeftHand.transform.rotation));
        }
        {
            Transform child = new GameObject("RightHand").transform;
            child.SetParent(RightHand.transform);
            child.SetLocalPositionAndRotation(Vector3.zero, RightHandEE.rotation * Quaternion.Inverse(RightHand.transform.rotation));
        }
        IsCalibrated = true;
    }

    private void UpdateVisuals()
    {
        HipsMat.color = HipsActive ? EnabledColor : DisabledColor;
        LeftFootMat.color = LeftFootActive ? EnabledColor : DisabledColor;
        RightFootMat.color = RightFootActive ? EnabledColor : DisabledColor;
        HeadMat.color = HeadActive ? EnabledColor : DisabledColor;
        LeftHandMat.color = LeftHandActive ? EnabledColor : DisabledColor;
        RightHandMat.color = RightHandActive ? EnabledColor : DisabledColor;
    }

    private void OnDestroy()
    {
        if (HipsMat != null)
        {
            Destroy(HipsMat);
        }
        if (LeftFootMat != null)
        {
            Destroy(LeftFootMat);
        }
        if (RightFootMat != null)
        {
            Destroy(RightFootMat);
        }
        if (HeadMat != null)
        {
            Destroy(HeadMat);
        }
        if (LeftHandMat != null)
        {
            Destroy(LeftHandMat);
        }
        if (RightHandMat != null)
        {
            Destroy(RightHandMat);
        }
    }
}
