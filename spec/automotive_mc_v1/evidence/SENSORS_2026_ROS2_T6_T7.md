# Sensors (MDPI) 2026, 26(2):463 - ROS 2 autonomous driving architecture
Platform: Jetson Orin Nano 8GB on a 1:10 scale vehicle; framework ROS 2.
Table 6 (per stage, mean / P95 / max):
  object detection   28.853 / 33.243 / 49.458 ms
  decision making     6.675 / 11.082 / 15.869 ms
  (preprocessing, lane detection, obstacle detection and state machine rows also
   reported; transcribed values only, no paraphrase)
Table 7 (pipeline end-to-end, mean / P95 / max) for the lane, object and obstacle
pipelines.
Per-stage CPU/GPU allocation is NOT stated, so t*f -> cycles is not defensible.
