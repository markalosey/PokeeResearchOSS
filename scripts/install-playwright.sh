#!/bin/bash
# Copyright 2025 Pokee AI Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Install Playwright browser automation dependencies
# This script installs Playwright Python package and Chromium browser binaries
# Required system dependencies for Debian/Ubuntu are also installed

set -e

echo "Installing Playwright Python package..."
pip install playwright

echo "Installing Playwright Chromium browser..."
playwright install chromium

echo "Installing system dependencies for Debian/Ubuntu..."
sudo apt-get update
sudo apt-get install -y \
    chromium-browser \
    chromium-driver \
    libnss3 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libcairo2 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libpango-1.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    libxss1 \
    libasound2 \
    || echo "Warning: Some system dependencies may have failed to install"

echo "Playwright installation complete!"
echo "Verify installation: python3 -c 'from playwright.async_api import async_playwright; print(\"OK\")'"

