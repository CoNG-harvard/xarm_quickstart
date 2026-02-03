import pyspacemouse
import time

def test_mouse():
    # Using dof_callback for rotation/translation and button_callback for buttons
    # pyspacemouse.print_state is a built-in helper that prints everything
    dev = pyspacemouse.open(
        dof_callback=pyspacemouse.print_state, 
        button_callback=pyspacemouse.print_buttons
    )
    
    if dev:
        print("SpaceMouse connected! Move the puck to see data...")
        try:
            while True:
                # IMPORTANT: You must call read() to process the queue
                # even if you are using callbacks.
                dev.read() 
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            dev.close()
    else:
        print("No SpaceMouse found.")

if __name__ == "__main__":
    test_mouse()