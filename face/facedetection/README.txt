CMD as admin
winget install usbipd

WSL
# Update and install essential packages
sudo apt update && sudo apt upgrade -y


# Install USBIP tools and video utilities
sudo apt install linux-tools-virtual hwdata -y
sudo apt install usbip -y
sudo update-alternatives --install /usr/local/bin/usbip usbip /usr/lib/linux-tools/*/usbip 20

# Install OpenCV and development tools
sudo apt install libopencv-dev python3-opencv -y
sudo apt install build-essential cmake git pkg-config -y


sudo apt install v4l-utils
ls /dev/video*

In CMD
# List available USB devices
usbipd list

# Bind the camera to make it shareable
usbipd bind --busid 2-7

# Attach to WSL
usbipd attach --wsl --busid 2-7


g++ -o face_detection face_detection.cpp `pkg-config --cflags --libs opencv4`

