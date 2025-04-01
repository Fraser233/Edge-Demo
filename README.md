# Raspberry Pi Setup and YOLO Demo Installation

## 1. Assemble the Raspberry Pi
Follow the manual in the box to assemble the Raspberry Pi.

## 2. Install the Camera
Refer to the video: [Camera Installation Guide](https://youtu.be/GImeVqHQzsE)

> ⚠️ Pay attention to the orientation of the wide wire shown in the video. The metal pins should face the pins inside the socket.

## 3. Connect Peripherals and Power On
- Connect to a screen using the HDMI cable.
- Connect a keyboard and mouse.
- Plug in the power to boot up your Raspberry Pi.
- Follow the on-screen steps to set up the system.

## 4. Transfer YOLO Demo
- Use a USB drive to transfer the `YOLO_Demo` folder to your Raspberry Pi.
- Copy the folder into your `Documents` directory.
- Open Terminal and navigate to the folder:

```bash
cd ~/Documents/YOLO_Demo
```

## 5. Install System Dependencies
Run the following command to install required libraries:

```bash
sudo apt install libcap-dev libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libavfilter-dev libswscale-dev libswresample-dev
```

> When prompted with:
> ```
> Do you want to continue? [Y/n]
> ```
> Type `Y` and press Enter.

## 6. Install Python Build Tools

```bash
pip install --upgrade pip setuptools wheel
pip install --upgrade sip pyqt-builder
```

## 7. Create a Virtual Environment

```bash
python -m venv edge-env
```

## 8. Install Python Dependencies

```bash
pip install -r requirements.txt
```

> ⏳ This step may take a while. Be patient.
```
