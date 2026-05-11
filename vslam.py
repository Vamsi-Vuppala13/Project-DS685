import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


class VisualOdometry:
    def __init__(self, K, dist=None):
        """
        K: 3x3 camera intrinsic matrix
        dist: distortion coefficients
        """
        self.K = K
        self.dist = dist if dist is not None else np.zeros(5)
        self.orb = cv2.ORB_create(nfeatures=3000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.prev_frame = None
        self.prev_kp = None
        self.prev_des = None
        self.poses = [np.eye(4)]
        self.landmarks = []

    def detect_and_describe(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        return kp, des

    def match_features(self, des1, des2):
        if des1 is None or des2 is None:
            return []

        matches = self.matcher.knnMatch(des1, des2, k=2)
        good = []

        for pair in matches:
            if len(pair) < 2:
                continue

            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

        return good

    def estimate_pose(self, kp1, kp2, matches):
        if len(matches) < 8:
            return None, None, None

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(
            pts1,
            pts2,
            self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        if E is None:
            return None, None, None

        _, R, t, mask = cv2.recoverPose(
            E,
            pts1,
            pts2,
            self.K,
            mask=mask
        )

        return R, t, mask

    def triangulate(self, kp1, kp2, matches, R, t, mask):
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        if mask is not None:
            inliers = mask.ravel() == 1
            pts1 = pts1[inliers]
            pts2 = pts2[inliers]

        if len(pts1) < 2:
            return np.empty((0, 3))

        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K @ np.hstack([R, t])

        pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pts3d = pts4d[:3] / pts4d[3]

        return pts3d.T

    def process_frame(self, frame):
        kp, des = self.detect_and_describe(frame)

        if self.prev_frame is None or des is None or self.prev_des is None:
            self.prev_frame = frame
            self.prev_kp = kp
            self.prev_des = des
            return self.poses[-1]

        matches = self.match_features(self.prev_des, des)

        if len(matches) < 8:
            self.prev_frame = frame
            self.prev_kp = kp
            self.prev_des = des
            return self.poses[-1]

        R, t, mask = self.estimate_pose(self.prev_kp, kp, matches)

        if R is None:
            self.prev_frame = frame
            self.prev_kp = kp
            self.prev_des = des
            return self.poses[-1]

        prev_pose = self.poses[-1]

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.ravel()

        new_pose = prev_pose @ np.linalg.inv(T)
        self.poses.append(new_pose)

        pts3d = self.triangulate(self.prev_kp, kp, matches, R, t, mask)
        self.landmarks.extend(pts3d.tolist())

        self.prev_frame = frame
        self.prev_kp = kp
        self.prev_des = des

        return new_pose


def run_on_kitti(sequence_path, max_frames=200):
    """Run visual odometry on KITTI dataset sequence"""
    seq_path = Path(sequence_path)
    image_path = seq_path / "image_0"
    calib_path = seq_path / "calib.txt"
    gt_path = seq_path / "poses.txt" if (seq_path / "poses.txt").exists() else None

    if not image_path.exists():
        raise FileNotFoundError(f"Image folder not found: {image_path}")

    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    K = None

    with open(calib_path) as f:
        for line in f:
            if line.startswith("P0:"):
                vals = list(map(float, line.split()[1:]))
                K = np.array(vals).reshape(3, 4)[:3, :3]
                break

    if K is None:
        raise RuntimeError("Could not read camera matrix from calib.txt")

    vo = VisualOdometry(K)
    images = sorted(image_path.glob("*.png"))

    if len(images) == 0:
        raise RuntimeError(f"No PNG images found in {image_path}")

    images = images[:max_frames]

    print(f"Processing {len(images)} KITTI frames...")

    estimated = []

    for i, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))

        if frame is None:
            continue

        pose = vo.process_frame(frame)
        estimated.append(pose[:3, 3])

        if i % 20 == 0:
            print(f"Frame {i}/{len(images)}")

    estimated = np.array(estimated)

    gt = None

    if gt_path and gt_path.exists():
        gt_poses = []

        with open(gt_path) as f:
            for line in f:
                vals = list(map(float, line.split()))
                T = np.array(vals).reshape(3, 4)
                gt_poses.append(T[:3, 3])

        gt = np.array(gt_poses[:len(estimated)])

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(estimated[:, 0], estimated[:, 2], "b-", label="Estimated")

    if gt is not None:
        plt.plot(gt[:, 0], gt[:, 2], "r-", label="Ground Truth")

    plt.legend()
    plt.title("KITTI Camera Trajectory - Top View")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid(True)
    plt.axis("equal")

    plt.subplot(1, 2, 2)
    plt.plot(estimated[:, 1], label="Estimated Y")

    if gt is not None:
        plt.plot(gt[:, 1], label="GT Y")

    plt.legend()
    plt.title("Height Over Time")
    plt.xlabel("Frame")
    plt.ylabel("Y")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("kitti_trajectory.png", dpi=150)
    plt.show()

    print("Trajectory saved to kitti_trajectory.png")

    return estimated, gt


def run_on_webcam():
    """Run visual odometry on webcam feed"""
    K = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ], dtype=np.float64)

    vo = VisualOdometry(K)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    poses = []

    print("Webcam VSLAM started.")
    print("Press Q to stop.")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        pose = vo.process_frame(frame)
        poses.append(pose[:3, 3])

        kp, _ = vo.detect_and_describe(frame)
        frame_kp = cv2.drawKeypoints(
            frame,
            kp,
            None,
            color=(0, 255, 0),
            flags=cv2.DrawMatchesFlags_DEFAULT
        )

        cv2.imshow("Webcam VSLAM - Press Q to stop", frame_kp)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    poses = np.array(poses)

    if len(poses) > 1:
        poses = poses - poses[0]

        plt.figure(figsize=(9, 7))
        plt.plot(poses[:, 0], poses[:, 2], "b-", label="Estimated")
        plt.scatter(poses[0, 0], poses[0, 2], color="green", s=100, label="Start")
        plt.scatter(poses[-1, 0], poses[-1, 2], color="red", s=100, label="End")
        plt.title("Webcam Visual Odometry Trajectory")
        plt.xlabel("X")
        plt.ylabel("Z")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        plt.savefig("webcam_trajectory.png", dpi=150)
        plt.show()

        print("Trajectory saved to webcam_trajectory.png")

    return poses


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2 and sys.argv[1] == "--kitti":
        estimated, gt = run_on_kitti(sys.argv[2])
        print(f"Processed {len(estimated)} KITTI frames")

    elif len(sys.argv) > 1 and sys.argv[1] == "--webcam":
        poses = run_on_webcam()
        print(f"Captured {len(poses)} webcam poses")

    else:
        print("Usage:")
        print("  python3 vslam.py --kitti <kitti_sequence_path>")
        print("  python3 vslam.py --webcam")
