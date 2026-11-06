import h5py

HDF5_PATH = "data/processed/head_pose.h5"

with h5py.File(HDF5_PATH, "r") as f:

    print("Top-level groups:", list(f.keys()))

    users = f["users"]

    print("Number of users:", len(users))

    for user_id in users:

        sessions = users[user_id]["sessions"]

        for session_id in sessions:

            group = sessions[session_id]

            timestamps = group["timestamps"][:]
            quaternions = group["quaternions"][:]

            print("\nUser:", user_id)
            print("Session:", session_id)
            print("Timestamps shape:", timestamps.shape)
            print("Quaternions shape:", quaternions.shape)
            print("Sampling rate:", group.attrs["sampling_rate_hz"])
            print("Quaternion order:", group.attrs["quaternion_order"])

            break

        break