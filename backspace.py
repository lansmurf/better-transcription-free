import pyautogui
import time

def type_and_erase(message, delay=2):
    # Type the message
    pyautogui.write(message, interval=0.01)
    time.sleep(delay)  # Wait for a specified delay

    # Erase the message
    for _ in message:
        pyautogui.press('backspace')

# Example usage
time.sleep(3)
message = "Recording!"
type_and_erase(message)
