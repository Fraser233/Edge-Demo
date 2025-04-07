# Raspberry Pi Setup and YOLO Demo Installation

![Final Setup](final_setup.jpg)
*Figure: Final hardware+sofeware setup for the Raspberry Pi 4 with Picamera.*

## 1. Flash Raspberry Pi OS to SD Card
1. On your computer, download the Raspberry Pi OS imager from the [official site](https://www.raspberrypi.com/software/).  
2. Insert the microSD card (come with the device in the box) into the computer’s card reader.  
3. Use **Raspberry Pi Imager** to select the OS image (64-bit) and the SD card, then click **Flash**.  
4. Wait for the process to complete, then safely eject the card.

## 2. Connect Peripherals and Power On
- Connect a monitor via HDMI.  
- Plug in a USB keyboard and mouse.  
- Insert the power supply to boot. 
- Insert the flashed SD card into the Raspberry Pi’s microSD slot.  
- Complete the on‑screen setup (locale, Wi‑Fi, password, updates).
    - Connect to your own Wi‑Fi or UGuest

## 3. Install the Camera
Refer to the video: [Camera Installation Guide](https://youtu.be/GImeVqHQzsE)

> ⚠️ Pay attention to the orientation of the ribbon cable. The metal contacts should face the metal pins inside the socket.


## 4. Transfer YOLO Demo
1. Download the `YOLO_Demo` folder to a USB drive.
2. Plug the USB drive into your Respberry PI.  
3. Copy the demo folder into your home directory (root of your user)
4. Open the terminal and use the following command to open the `YOLO_Demo` folder:

   ```bash
   cd Demo

## 5. Install System Dependencies
Run the following command to install required libraries:

```bash
sudo apt install -y libcap-dev libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libavfilter-dev libswscale-dev libswresample-dev libcamera-apps
```

## 6. Install Python Build Tools

```bash
pip install --upgrade pip setuptools wheel
pip install --upgrade sip pyqt-builder
```

## 7. Create a Virtual Environment

```bash
python -m venv edge-env
source edge-env/bin/activate
```

## 8. Install Python Dependencies

```bash
pip install -r requirements.txt
```

> ⏳ This step may take a while. Be patient.

## 9. Run script

```bash
python main.py
```

> ⏳ The starting process may take 30 secs. Be patient.

```
