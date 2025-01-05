using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[DefaultExecutionOrder(-100)]
public class UpdateManager : MonoBehaviour
{
    public static UpdateManager Instance;

    public event System.Action OnBeforeRetargetTrackers;
    public event System.Action OnRetargetTrackers;
    public event System.Action OnAfterRetargetTrackers;
    public event System.Action OnDragPoser;
    public event System.Action OnAfterDragPoser;
    public event System.Action OnCharacterUpdated;
    public event System.Action OnAfterCharacterUpdated;

    private void Awake()
    {
        Debug.Assert(Instance == null, "There should only be one UpdateManager in the scene");
        if (Instance == null)
        {
            Instance = this;
        }
    }


    void Update()
    {
        if (OnBeforeRetargetTrackers != null)
        {
            OnBeforeRetargetTrackers();
        }

        if (OnRetargetTrackers != null)
        {
            OnRetargetTrackers();
        }

        if (OnAfterRetargetTrackers != null)
        {
            OnAfterRetargetTrackers();
        }

        if (OnDragPoser != null)
        {
            OnDragPoser();
        }

        if (OnAfterDragPoser != null)
        {
            OnAfterDragPoser();
        }

        if (OnCharacterUpdated != null)
        {
            OnCharacterUpdated();
        }

        if (OnAfterCharacterUpdated != null)
        {
            OnAfterCharacterUpdated();
        }
    }
}
