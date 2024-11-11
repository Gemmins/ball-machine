import pygame
import math
import serial
import time

"""
This is the controller side of the robot control
The computer on the robot runs this script then communicates information to the pico microcontroller
"""

DEADZONE = 0.2

def apply_deadzone(value, deadzone):
    if abs(value) < deadzone:
        return 0
    return value

def main():
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("No controller found. Please connect a DualShock 4 controller.")
        return

    controller = pygame.joystick.Joystick(0)
    controller.init()

    if "" not in controller.get_name():
        print("This script is designed for DualShock 4 controllers.")
        print(f"Connected controller: {controller.get_name()}")
        return

    ser = serial.Serial('COM6', 115200, timeout=1)
    time.sleep(2)  # Allow time for the serial connection to establish

    print("DualShock 4 controller connected. Press Ctrl+C to exit.")

    last_motor_direction = 0  # 0 for forward, 1 for backward

    try:
        while True:
            pygame.event.pump()

            # Right stick for stepper motor
            x_axis_right = apply_deadzone(controller.get_axis(2), DEADZONE)
            y_axis_right = apply_deadzone(controller.get_axis(3), DEADZONE)

            # Left stick for PWM motors
            y_axis_left = apply_deadzone(controller.get_axis(1), DEADZONE)  # Forward/Backward axis

            # Calculate stepper motor rate
            magnitude = math.sqrt(x_axis_right**2 + y_axis_right**2)
            angle = math.degrees(math.atan2(y_axis_right, x_axis_right))
            speed = int(magnitude * 100)
            direction = 1 if -90 <= angle <= 90 else -1
            stepper_rate = direction * speed

            # Calculate PWM values for left and right motors
            pwm_value = int(abs(y_axis_left) * 100)

            # Determine motor direction, maintaining previous direction in deadzone
            if y_axis_left < -DEADZONE:
                motor_direction = 0  # Forward
                last_motor_direction = 0
            elif y_axis_left > DEADZONE:
                motor_direction = 1  # Backward
                last_motor_direction = 1
            else:
                motor_direction = last_motor_direction

            # Prepare data string to send
            data = f"{stepper_rate},{pwm_value},{motor_direction}\n"
            ser.write(data.encode())

            print(f"Stepper rate: {stepper_rate}, PWM: {pwm_value}, Direction: {'Backward' if motor_direction else 'Forward'}")

            time.sleep(0.1)  # Small delay to reduce output frequency

    except KeyboardInterrupt:
        print("\nExiting...")

    finally:
        ser.close()
        pygame.quit()

if __name__ == "__main__":
    main()