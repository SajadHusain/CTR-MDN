# Probabilistic Forward and Inverse Kinematics of Concentric Tube Robots

Code for the mixture density networks (MDNs) in *A Probabilistic Learning Framework for Forward and Inverse Kinematics of Concentric Tube Robots*.

-FK-MDN: actuator configuration plus a buffer of the M previous configurations → Gaussian mixture over the tip pose (position + unit quaternion).
-IK-MDN: tip pose → Gaussian mixture over the actuator configuration.


# Data
We use the public [CRL-Dataset-CTCR-Pose](https://github.com/ContinuumRoboticsLab/CRL-Dataset-CTCR-Pose) dataset (Grassmann et al., IROS 2022, MIT license). It contains 100,000 measured configurations of a three-tube CTR in 8 sequences of 12,500 samples. For each sample it gives the joint values and the tip poses of the three tubes, measured with an electromagnetic tracker.

