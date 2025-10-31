#!/usr/bin/env python3

import subprocess
import re
import logging
import logging.handlers
import time

# Setup logging to use syslog
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
# Get the root logger
logger = logging.getLogger()
# Add the syslog handler
# On most systems, this will be /dev/log or /var/run/log
handler = logging.handlers.SysLogHandler(address="/dev/log")
logger.addHandler(handler)

# GPU Temperature thresholds (more aggressive than CPU)
GPU_CRITICAL_TEMP = 85  # At this point, fans should be maxed
GPU_HIGH_TEMP = 75  # Aggressive cooling needed
GPU_WARM_TEMP = 65  # Moderate cooling
GPU_NORMAL_TEMP = 55  # Normal cooling

# CPU/Board Temperature thresholds
CPU_CRITICAL_TEMP = 75  # Critical for CPU
CPU_HIGH_TEMP = 65  # High for CPU
CPU_WARM_TEMP = 55  # Warm for CPU
CPU_NORMAL_TEMP = 45  # Normal for CPU


# Function to get temperature information
def get_temp_info():
    try:
        result = subprocess.run(
            ["ipmitool", "sdr", "type", "temperature"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ) as e:
        logging.error(f"Error getting temp info: {e}")
        return ""


# Function to get GPU temperature information
def get_gpu_temp_info():
    try:
        result = subprocess.run(
            ["nvidia-smi", "-q", "-d", "TEMPERATURE"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        temp_info = result.stdout
        # Find all GPU temperatures
        matches = re.findall(r"GPU Current Temp\s+:\s+(\d+) C", temp_info)
        return [int(temp) for temp in matches] if matches else []
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return []


# Function to extract numeric values from strings based on a regex pattern
def extract_numeric_value(s, pattern):
    match = re.search(pattern, s)
    return float(match.group(1)) if match else None


# Function to set fan speed
def set_fan_speed(speed):
    try:
        subprocess.run(
            ["ipmitool", "raw", "0x30", "0x30", "0x02", "0xff", f"0x{speed:02x}"],
            check=True,
            timeout=10,
        )
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ) as e:
        logging.error(f"Failed to set fan speed: {e}")


# Function to determine fan speed based on GPU temperature (priority)
def get_fan_speed_for_gpu(gpu_temp):
    """Determine fan speed based on GPU temperature with aggressive thresholds."""
    if gpu_temp >= GPU_CRITICAL_TEMP:
        return 0x64  # 100% - Critical, max cooling
    elif gpu_temp >= GPU_HIGH_TEMP:
        return 0x5A  # 90% - High, aggressive cooling
    elif gpu_temp >= GPU_WARM_TEMP:
        return 0x50  # 80% - Warm, increased cooling
    elif gpu_temp >= GPU_NORMAL_TEMP:
        return 0x3C  # 60% - Normal, moderate cooling
    else:
        return 0x28  # 40% - Low, minimal cooling


# Function to determine fan speed based on CPU/board temperature
def get_fan_speed_for_cpu(cpu_temp):
    """Determine fan speed based on CPU/board temperature."""
    if cpu_temp >= CPU_CRITICAL_TEMP:
        return 0x64  # 100%
    elif cpu_temp >= CPU_HIGH_TEMP:
        return 0x50  # 80%
    elif cpu_temp >= CPU_WARM_TEMP:
        return 0x3C  # 60%
    elif cpu_temp >= CPU_NORMAL_TEMP:
        return 0x28  # 40%
    else:
        return 0x14  # 20%


# Main daemon loop
def main():
    logging.info("Fan control daemon started (GPU-optimized version).")
    # On start, set fans to a safe speed while we get first readings
    set_fan_speed(0x50)  # 80% - safer default for GPU workloads

    while True:
        try:
            # Get temperature info
            temp_info_str = get_temp_info()
            temp_info = temp_info_str.split("\n") if temp_info_str else []

            board_cpu_temps = [
                temp
                for temp in [
                    extract_numeric_value(s, r"\b(\d+\.?\d*) degrees C\b")
                    for s in temp_info
                ]
                if temp is not None
            ]
            gpu_temps = get_gpu_temp_info()

            highest_cpu_temp = max(board_cpu_temps) if board_cpu_temps else None
            highest_gpu_temp = max(gpu_temps) if gpu_temps else None

            # GPU temperatures take priority - they run hotter and need aggressive cooling
            if highest_gpu_temp is not None:
                # Use GPU-based fan speed, but ensure CPU doesn't override if it's critical
                fan_speed = get_fan_speed_for_gpu(highest_gpu_temp)

                # If CPU is also critical, ensure we're at max speed
                if (
                    highest_cpu_temp is not None
                    and highest_cpu_temp >= CPU_CRITICAL_TEMP
                ):
                    fan_speed = max(fan_speed, get_fan_speed_for_cpu(highest_cpu_temp))

                logging.info(
                    f"GPU Temp: {highest_gpu_temp}°C, CPU Temp: {highest_cpu_temp if highest_cpu_temp else 'N/A'}°C, "
                    f"Setting fan speed: {fan_speed}% (hex: 0x{fan_speed:02x})"
                )
            elif highest_cpu_temp is not None:
                # Fallback to CPU-based control if no GPU temps available
                fan_speed = get_fan_speed_for_cpu(highest_cpu_temp)
                logging.info(
                    f"CPU Temp: {highest_cpu_temp}°C, Setting fan speed: {fan_speed}% (hex: 0x{fan_speed:02x})"
                )
            else:
                # Failsafe if no temps available
                logging.warning(
                    "Could not read any temperature sensors. Maintaining safe 80% fan speed."
                )
                fan_speed = 0x50  # 80% failsafe speed
                time.sleep(10)
                continue

            set_fan_speed(fan_speed)

            # Faster check interval for GPU workloads (3 seconds instead of 5)
            # This allows faster response to temperature changes
            time.sleep(3)

        except Exception as e:
            logging.error(f"An error occurred in the main loop: {e}")
            # In case of an unexpected error, set a safe fan speed and wait before retrying
            set_fan_speed(0x50)  # 80% - safer for GPU workloads
            time.sleep(10)


if __name__ == "__main__":
    main()
