from glob import glob

from setuptools import find_packages, setup

package_name = "robotique_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="beladjioo",
    description="Adaptateurs ROS 2 des branches du dépôt robotique.",
    license="TODO",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "unicycle_sim = robotique_ros.unicycle_sim_node:main",
            "path_planner = robotique_ros.path_planner_node:main",
            "pure_pursuit = robotique_ros.pure_pursuit_node:main",
        ],
    },
)
