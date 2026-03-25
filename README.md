# RockLook — BUILDCORED ORCAS Day 1

## What it does

Detects when you tilt your head down via webcam and plays rock music — look back up and it pauses.

## Hardware concept

This mirrors a tilt sensor triggering a relay: the webcam + FaceMesh acts as the sensor, a pitch threshold acts as the comparator, and pygame audio is the actuator.

## Screenshot / GIF


## What I would do differently

I would use eye-gaze direction instead of head pitch so it triggers when you look down with your eyes, not just your whole head.

## Run it

```bash
pip install -r requirements.txt
python rocklook.py
```