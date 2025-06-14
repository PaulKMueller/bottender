from setuptools import find_packages, setup

package_name = 'ai_hackathon'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'std_msgs', 'sereact_humanoid_msg'],
    zip_safe=True,
    maintainer='bmeyjohann',
    maintainer_email='benjamin.meyjohann@gmx.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
		'my_node = ai_hackathon.my_node:main',
        'gripper_loop = gripper_loop.gripper_loop_node:main',
        'policy_inference_node = your_package.policy_inference_node:main',
        ],
    },
)
