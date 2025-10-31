#!/usr/bin/env python3

import subprocess
import re
import logging
import logging.handlers
import time
import fcntl
import os
import sys

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


# Function to enable manual fan control
def enable_manual_fan_control():
    """Enable manual fan control mode on Dell server."""
    try:
        # Enable manual fan control: 0x30 0x30 0x01 0x00
        result = subprocess.run(
            ["ipmitool", "raw", "0x30", "0x30", "0x01", "0x00"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        logging.info("Manual fan control enabled")
        return True
    except Exception as e:
        logging.warning(f"Failed to enable manual fan control (may already be enabled): {e}")
        return False


# Function to set fan speed
def set_fan_speed(speed):
    """Set fan speed via IPMI and verify it was set.
    
    For Dell servers, we need to:
    1. Enable manual fan control first
    2. Set fan speed (0x30 0x30 0x02 0xff <hex_speed>)
    """
    # Ensure manual fan control is enabled
    enable_manual_fan_control()
    
    try:
        result = subprocess.run(
            ["ipmitool", "raw", "0x30", "0x30", "0x02", "0xff", f"0x{speed:02x}"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        # Log if there's any output (IPMI sometimes returns status)
        if result.stdout.strip():
            logging.info(f"IPMI response: {result.stdout.strip()}")
        if result.stderr.strip():
            logging.warning(f"IPMI stderr: {result.stderr.strip()}")
        
        # Verify the command worked by checking if we got any error response
        # Some Dell servers return "01" on success, "00" on failure
        if result.stdout.strip() == "01":
            logging.warning("IPMI returned error code 01 - command may have failed")
            return False
            
        return True
    except subprocess.CalledProcessError as e:
        logging.error(
            f"Failed to set fan speed {speed}% (0x{speed:02x}): IPMI command failed with code {e.returncode}"
        )
        if e.stderr:
            logging.error(f"IPMI error output: {e.stderr}")
        if e.stdout:
            logging.error(f"IPMI output: {e.stdout}")
        return False
    except FileNotFoundError:
        logging.error("ipmitool not found! Install ipmitool package.")
        return False
    except subprocess.TimeoutExpired:
        logging.error(f"IPMI command timed out setting fan speed {speed}%")
        return False
    except Exception as e:
        logging.error(f"Unexpected error setting fan speed: {e}")
        return False


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
    if not set_fan_speed(current_fan_speed):
        logging.error(
            "CRITICAL: Failed to set initial fan speed! Check IPMI connection."
        )

    # Temperature smoothing to prevent rapid fluctuations
    temp_history = []
    TEMP_HISTORY_SIZE = 5  # Average last 5 readings (25 seconds with 5s intervals)
    last_logged_change = time.time()
    critical_fan_speed_time = 0  # Track when we set critical fan speed
    MIN_CRITICAL_FAN_DURATION = 120  # Keep fans at 90%+ for at least 120 seconds (2 minutes) after critical temp
    last_fan_change_time = 0  # Track when we last changed fan speed
    MIN_TIME_BETWEEN_CHANGES = (
        30  # Minimum 30 seconds between ANY fan speed changes (increased from 15)
    )

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

            # Smooth GPU temperature to prevent rapid fluctuations
            # BUT: Use raw temperature for critical temps or if we don't have enough history
            if highest_gpu_temp is not None:
                # Critical temps bypass smoothing - respond immediately
                if (
                    highest_gpu_temp >= GPU_CRITICAL_TEMP
                    or highest_gpu_temp >= GPU_HIGH_TEMP
                ):
                    # Use raw temperature for high/critical temps
                    effective_gpu_temp = highest_gpu_temp
                else:
                    # Only use smoothing for non-critical temps
                    temp_history.append(highest_gpu_temp)
                    if len(temp_history) > TEMP_HISTORY_SIZE:
                        temp_history.pop(0)
                    # Use averaged temperature if we have enough history, otherwise use raw
                    if len(temp_history) >= TEMP_HISTORY_SIZE:
                        effective_gpu_temp = sum(temp_history) / len(temp_history)
                    else:
                        effective_gpu_temp = (
                            highest_gpu_temp  # Use raw until we have history
                        )
            else:
                effective_gpu_temp = None

            # GPU temperatures take priority - they run hotter and need aggressive cooling
            if effective_gpu_temp is not None:
                # Use effective GPU temperature with hysteresis
                # But bypass hysteresis for critical temps
                if effective_gpu_temp >= GPU_CRITICAL_TEMP:
                    fan_speed = 0x64  # Force 100% for critical temps
                else:
                    fan_speed = get_fan_speed_for_gpu(
                        effective_gpu_temp, current_fan_speed
                    )

                # If CPU is also critical, ensure we're at max speed
                if (
                    highest_cpu_temp is not None
                    and highest_cpu_temp >= CPU_CRITICAL_TEMP
                ):
                    fan_speed = max(fan_speed, get_fan_speed_for_cpu(highest_cpu_temp))

                # Prevent rapid cycling: Maintain high fan speeds for minimum duration
                current_time = time.time()
                if fan_speed >= 0x5A:  # If we're setting to 90% or 100%
                    if critical_fan_speed_time == 0:
                        critical_fan_speed_time = current_time
                        logging.info(f"Critical fan speed activated at {fan_speed}%")
                elif (
                    current_fan_speed >= 0x5A
                ):  # Currently at 90%+ but trying to reduce
                    # Only allow reduction if we've been at high speed for minimum duration
                    # AND temperature has dropped significantly
                    if (
                        current_time - critical_fan_speed_time
                        < MIN_CRITICAL_FAN_DURATION
                    ):
                        # Force maintain high fan speed
                        fan_speed = current_fan_speed
                        logging.warning(
                            f"Preventing fan speed reduction: maintaining {fan_speed}% for "
                            f"{MIN_CRITICAL_FAN_DURATION - (current_time - critical_fan_speed_time):.0f}s more"
                        )
                    elif (
                        effective_gpu_temp < GPU_HIGH_TEMP_DOWN - 5
                    ):  # Temp dropped significantly below threshold
                        # Allow reduction only if temp dropped well below threshold
                        critical_fan_speed_time = 0
                        logging.info(
                            f"Allowing fan speed reduction: temp dropped to {effective_gpu_temp:.1f}°C"
                        )
                    else:
                        # Keep high speed even if temp slightly dropped
                        fan_speed = current_fan_speed
                else:
                    # Not at critical speed, reset timer
                    critical_fan_speed_time = 0

                # Only log when fan speed actually changes or every 30 seconds
                if fan_speed != current_fan_speed:
                    logging.info(
                        f"GPU Temp: {effective_gpu_temp:.1f}°C (raw: {highest_gpu_temp}°C), "
                        f"CPU Temp: {highest_cpu_temp if highest_cpu_temp else 'N/A'}°C, "
                        f"CHANGING fan speed: {current_fan_speed}% → {fan_speed}% (hex: 0x{fan_speed:02x})"
                    )
                    last_logged_change = time.time()
                elif time.time() - last_logged_change >= 30:
                    # Log status every 30 seconds even if no change
                    logging.info(
                        f"GPU Temp: {effective_gpu_temp:.1f}°C (raw: {highest_gpu_temp}°C), "
                        f"CPU Temp: {highest_cpu_temp if highest_cpu_temp else 'N/A'}°C, "
                        f"Maintaining fan speed: {fan_speed}% (hex: 0x{fan_speed:02x})"
                    )
                    last_logged_change = time.time()
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
            # BUT: Always update for critical temps to ensure fans are actually responding
            # AND: Enforce minimum time between changes to prevent thrashing
            time_since_last_change = time.time() - last_fan_change_time

            # Allow immediate changes only for critical temps or if enough time has passed
            is_critical = (
                highest_gpu_temp is not None and highest_gpu_temp >= GPU_CRITICAL_TEMP
            ) or fan_speed >= 0x5A

            if fan_speed != current_fan_speed:
                if is_critical or time_since_last_change >= MIN_TIME_BETWEEN_CHANGES:
                    success = set_fan_speed(fan_speed)
                    if success:
                        current_fan_speed = fan_speed
                        last_fan_change_time = time.time()
                        logging.info(
                            f"Fan speed successfully set to: {fan_speed}% (0x{fan_speed:02x})"
                        )
                    else:
                        logging.error(
                            f"FAILED to set fan speed to {fan_speed}% (0x{fan_speed:02x}) - keeping current {current_fan_speed}%"
                        )
                else:
                    # Prevent change - too soon since last change
                    logging.warning(
                        f"BLOCKED fan speed change: {current_fan_speed}% → {fan_speed}% "
                        f"(wait {MIN_TIME_BETWEEN_CHANGES - time_since_last_change:.0f}s more)"
                    )
                    fan_speed = current_fan_speed  # Keep current speed
            # Removed aggressive re-apply logic - it was causing thrashing by sending IPMI commands every 5 seconds
            # The main logic above handles setting fan speeds correctly when they change

            # Check interval - slower for stability (reduces IPMI calls)
            # Even critical temps checked every 5 seconds to prevent thrashing
            check_interval = 5  # Default - same for all temps
            time.sleep(check_interval)

        except Exception as e:
            logging.error(f"An error occurred in the main loop: {e}")
            # In case of an unexpected error, try to set a safe fan speed and wait before retrying
            if not set_fan_speed(0x50):  # 80% - safer for GPU workloads
                logging.error("Failed to set failsafe fan speed!")
            current_fan_speed = 0x50
            time.sleep(10)


if __name__ == "__main__":
    main()
