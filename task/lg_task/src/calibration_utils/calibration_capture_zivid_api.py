#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import glob
import datetime
import argparse
import zivid


def capture(file):
    app = zivid.Application()

    print("Connecting to camera")
    camera = app.connect_camera()

    suggest_settings_parameters = zivid.capture_assistant.SuggestSettingsParameters(
        max_capture_time=datetime.timedelta(milliseconds=1200),
        ambient_light_frequency=zivid.capture_assistant.SuggestSettingsParameters.AmbientLightFrequency.hz60,
    )

    print(f"Running Capture Assistant with parameters: {suggest_settings_parameters}")
    settings = zivid.capture_assistant.suggest_settings(camera, suggest_settings_parameters)

    print("Settings suggested by Capture Assistant:")
    for acquisition in settings.acquisitions:
        print(acquisition)

    print("Manually configuring processing settings (Capture Assistant only suggests acquisition settings)")
    settings.processing.filters.reflection.removal.enabled = True
    settings.processing.filters.smoothing.gaussian.enabled = True
    settings.processing.filters.smoothing.gaussian.sigma = 1.5

    print("Capturing frame")
    with camera.capture(settings) as frame:
        print(f"Saving frame to file: {file}")
        frame.save(file)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Capture a Zivid image")
        parser.add_argument("--file", help="File to save image to", required=True)
        args = parser.parse_args()
        file = args.file
        capture(file)
        sys.exit(0)
    except Exception as e:
        print(e)
        sys.exit(1)
