# Image ROS 2 Jazzy (LTS) avec les bibliothèques du dépôt et le paquet robotique_ros.
# À construire depuis la racine du dépôt :
#   docker build -f projets/ros2/Dockerfile -t robotique-ros2 .
#   docker run -it --rm robotique-ros2 ros2 launch robotique_ros demo_navigation.launch.py
FROM ros:jazzy-ros-base

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-pip python3-numpy python3-pytest \
    && rm -rf /var/lib/apt/lists/*

COPY socle /opt/robotique/socle
COPY projets/navigation /opt/robotique/projets/navigation
RUN pip3 install --break-system-packages --no-deps /opt/robotique/socle /opt/robotique/projets/navigation

WORKDIR /ws
COPY projets/ros2/robotique_ros /ws/src/robotique_ros
RUN . /opt/ros/jazzy/setup.sh && colcon build --symlink-install

RUN echo "source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash" >> /root/.bashrc
SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash && exec \"$@\"", "--"]
CMD ["ros2", "launch", "robotique_ros", "demo_navigation.launch.py"]
