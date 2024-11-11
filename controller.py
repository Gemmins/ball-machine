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

    ser = serial.Serial('COM6', 115200, timeout=1)
    time.sleep(2)  # Allow time for the serial connection to establish

    last_motor_direction = 0

    try:
        while True:
            pygame.event.pump()

            # Right stick for stepper motor
            x_axis_right = apply_deadzone(controller.get_axis(2), DEADZONE)

            # Left stick for PWM motors
            y_axis_left = apply_deadzone(controller.get_axis(1), DEADZONE)

            stepper_rate = int(x_axis_right * 100)

            # Calculate PWM values for left and right motors
            pwm_value = int(abs(y_axis_left) * 50)

            # Determine motor direction
            if y_axis_left < -DEADZONE:
                motor_direction = 0
                last_motor_direction = 0
            elif y_axis_left > DEADZONE:
                motor_direction = 1
                last_motor_direction = 1
            else:
                motor_direction = last_motor_direction

            # Send data to Pico
            data = f"{stepper_rate},{pwm_value},{motor_direction}\n"
            ser.write(data.encode())
            print(f"Sent: {data.strip()}")

            # Read acknowledgment from Pico
            response = ser.readline().decode().strip()
            if response:
                if response == "ACK":
                    print("Pico ACK")
                elif response == "ERR":
                    print("Pico Error")

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nExiting...")

    finally:
        ser.close()
        pygame.quit()


if __name__ == "__main__":
    main()