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

# Hysteresis thresholds (prevent fan oscillation)
# Use lower thresholds when decreasing fan speed to prevent thrashing
GPU_HIGH_TEMP_DOWN = 72  # Drop from 90% to 80% when temp goes below this
GPU_WARM_TEMP_DOWN = 62  # Drop from 80% to 60% when temp goes below this
GPU_NORMAL_TEMP_DOWN = 52  # Drop from 60% to 40% when temp goes below this

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
def get_fan_speed_for_gpu(gpu_temp, current_fan_speed=None):
    """Determine fan speed based on GPU temperature with aggressive thresholds and hysteresis.

    Args:
        gpu_temp: Current GPU temperature in Celsius
        current_fan_speed: Current fan speed (hex value) for hysteresis calculation
    """
    # Use hysteresis to prevent fan oscillation
    # When increasing: use normal thresholds
    # When decreasing: use lower thresholds (hysteresis)

    if gpu_temp >= GPU_CRITICAL_TEMP:
        return 0x64  # 100% - Critical, max cooling
    elif gpu_temp >= GPU_HIGH_TEMP:
        return 0x5A  # 90% - High, aggressive cooling
    elif gpu_temp >= GPU_WARM_TEMP:
        # Hysteresis: if currently at 90%, stay at 90% until temp drops below 72°C
        if current_fan_speed == 0x5A and gpu_temp >= GPU_HIGH_TEMP_DOWN:
            return 0x5A  # Stay at 90% until temp drops below 72°C
        return 0x50  # 80% - Warm, increased cooling
    elif gpu_temp >= GPU_NORMAL_TEMP:
        # Hysteresis: if currently at 80%, stay at 80% until temp drops below 62°C
        if current_fan_speed == 0x50 and gpu_temp >= GPU_WARM_TEMP_DOWN:
            return 0x50  # Stay at 80% until temp drops below 62°C
        return 0x3C  # 60% - Normal, moderate cooling
    else:
        # Hysteresis: if currently at 60%, stay at 60% until temp drops below 52°C
        if current_fan_speed == 0x3C and gpu_temp >= GPU_NORMAL_TEMP_DOWN:
            return 0x3C  # Stay at 60% until temp drops below 52°C
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
    logging.info("Fan control daemon started (GPU-optimized version with hysteresis).")
    # On start, set fans to a safe speed while we get first readings
    current_fan_speed = 0x50  # 80% - safer default for GPU workloads
    set_fan_speed(current_fan_speed)

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
                # Use GPU-based fan speed with hysteresis, but ensure CPU doesn't override if it's critical
                fan_speed = get_fan_speed_for_gpu(highest_gpu_temp, current_fan_speed)

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

            # Only update fan speed if it changed (reduce unnecessary IPMI calls)
            if fan_speed != current_fan_speed:
                set_fan_speed(fan_speed)
                current_fan_speed = fan_speed

            # Faster check interval for GPU workloads (3 seconds instead of 5)
            # This allows faster response to temperature changes
            time.sleep(3)

        except Exception as e:
            logging.error(f"An error occurred in the main loop: {e}")
            # In case of an unexpected error, set a safe fan speed and wait before retrying
            set_fan_speed(0x50)  # 80% - safer for GPU workloads
            current_fan_speed = 0x50
            time.sleep(10)


if __name__ == "__main__":
    main()
