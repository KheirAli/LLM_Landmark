## Landmark Detection

To identify landmarks, follow the steps below:

1. **Labeling Landmarks:**
   First, use contrastive learning methods to label landmarks from both **positive** and **negative** trajectories. This process generates:
   - **PS (Positive States)**: A file containing all identified positive states.
   - **NS (Negative States)**: A file containing all identified negative states.

2. **Applying Landmark Detection:**
   Once you have the `PS` and `NS` files, feed them into the provided graph search algorithm. This algorithm uses the labeled states to find meaningful landmarks within the environment.

The resulting landmarks serve as key points to guide the learning process, enabling more efficient and targeted policy improvement.
