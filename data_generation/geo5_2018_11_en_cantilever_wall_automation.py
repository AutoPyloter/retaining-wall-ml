from pywinauto import Application, keyboard, mouse
import re
import time
import pyperclip
import pyautogui
import os
PAUSE_ENTER = 0.02
PAUSE_TAB=0.02

class CantileverWall:
    _prev_geometry_params = []
    _prev_water_level = None
    _prev_surcharge_load = 0
    _scenario_index=1

    def __init__(self):
        self.app = None
        self.window = None


    def connect_to_application(self):
        """Connects to the GEO5 - Cantilever Wall - guz window."""
        regex = r"(?=.*GEO5)(?=.*Cantilever Wall)(?=.*guz)"
        try:
            self.app = Application(backend="win32").connect(title_re=regex)
            self.window = self.app.window(title_re=regex)
            if not self.window.exists():
                print("Application window not found.")
        except Exception as e:
            print(f"Connection error: {e}")


    def focus_on_window(self):
        """Sets focus to the application window."""
        if self.window and self.window.exists():
            try:
                self.window.set_focus()
            except Exception as e:
                print(f"Error focusing on window: {e}")
        else:
            print("No window available to focus on.")

    @staticmethod
    def clear_clipboard():
        """
        Clears the clipboard and confirms it has been cleared.
        """
        pyperclip.copy("")  # Clear the clipboard
        for _ in range(20):
            for _ in range(100):
                if pyperclip.paste() == "":
                    return True  # Clipboard cleared
                time.sleep(0.01)
            if pyperclip.paste() == "":
                return True
            else:
                pyperclip.copy("")
        return False  # Clipboard could not be cleared
    
    @staticmethod
    def copy_to_clipboard():
        """
        Sends Ctrl+A and Ctrl+C keystrokes and waits until the clipboard is populated.
        """
        while True:  # Loop continues as long as user responds "yes"
            keyboard.send_keys("^a^c", pause=PAUSE_TAB)  # Send Ctrl+A and Ctrl+C
            for _ in range(20):
                for _ in range(100):
                    clipboard_content = pyperclip.paste()
                    if clipboard_content != "":
                        return clipboard_content  # Clipboard populated, return content
                    time.sleep(0.01)
                if pyperclip.paste() == "":
                    pyperclip.copy("")
                else:
                    return pyperclip.paste()  # Clipboard populated, return content
                print("Problem occurred during copy. Retrying.")
                keyboard.send_keys("^a^c", pause=PAUSE_TAB)

            # If clipboard is still empty, prompt the user
            print("Clipboard copy failed. Do you want to continue? (Yes/No)")
            user_input = input("> ").strip().lower()

            if user_input in ["yes", "y"]:
                print("Waiting 3 seconds and retrying...")
                time.sleep(3)
            elif user_input in ["no", "n"]:
                print("Operation terminated.")
                exit()
            else:
                print("Invalid input. Please type 'Yes' or 'No'.")


    def read_field_value(self):
        """
        Clears the clipboard and then copies data into it.
        Returns the clipboard content as a string.
        """
        # Clear clipboard
        if not self.clear_clipboard():
            raise RuntimeError("Clipboard could not be cleared.")

        # Populate clipboard
        clipboard_content = self.copy_to_clipboard()
        if clipboard_content is None:
            raise RuntimeError("Clipboard was not populated.")

        return clipboard_content  # Return data from clipboard


    def check_window_open(self, title_regex):
        """
        Checks whether a window with the specified title regex is open.
        Returns True if the window exists, False otherwise.
        """
        try:
            window = self.app.window(title_re=title_regex)
            return window.exists()
        except Exception as e:
            print(f"Window check error: {e}")
            return False
    
    def map_to_geo5_params(self, wall_params):
        """
        Computes GEO5 input parameters from raw wall geometry parameters.
        Returns a list of values mapped to GEO5 field order.
        """
        return [
            wall_params[4],                          # stem_top_width
            wall_params[0],                          # wall_height (H)
            wall_params[5],                          # base_front_width
            wall_params[2],                          # base_thickness
            round(wall_params[1] - wall_params[2] - wall_params[3], 2),  # heel_width
            round(wall_params[5] + wall_params[6], 2),                   # base_total_width
            wall_params[7],                          # stem_bottom_width
            round(0 if wall_params[3] == wall_params[4] else wall_params[0] / (wall_params[3] - wall_params[4]), 2),  # batter_slope
            wall_params[8]                           # key_depth
        ]

    def geometry(self, geo5_params):
        """
        Enters values into the corresponding GEO5 geometry fields and saves the
        last entered values for comparison in subsequent calls.
        """
        try:
            # Check window availability
            if not self.window or not self.window.exists():
                raise RuntimeError("Window for value entry is not available.")

            # Focus on the window
            self.window.set_focus()

            # Navigate to the starting position using key combinations
            if round(geo5_params[5] - geo5_params[2], 2) != 0:
                keyboard.send_keys("{F10}IP{F10}IG{TAB}", pause=PAUSE_TAB)  # F10IP F10IG then one TAB
                keyboard.send_keys("{SPACE}", pause=PAUSE_TAB)  # Press SPACE
                keyboard.send_keys("{TAB 15}", pause=PAUSE_TAB)  # Press TAB 15 times
            else:
                keyboard.send_keys("{F10}IP{F10}IG{TAB 2}", pause=PAUSE_TAB)  # F10IP F10IG then two TABs
                keyboard.send_keys("{SPACE}", pause=PAUSE_TAB)  # Press SPACE
                keyboard.send_keys("{TAB 14}", pause=PAUSE_TAB)  # Press TAB 14 times

            # Enter values into fields sequentially
            if round(geo5_params[5] - geo5_params[2], 2) != 0:
                for index, value in enumerate(geo5_params):
                    self.process_field(index, value)
            else:
                for index, value in enumerate(geo5_params):
                    if index in {5, 6, 8}:  # Skip deactivated tabs
                        continue
                    self.process_field_skip_deactivated(index, value)

            # Save the current values
            CantileverWall._prev_geometry_params = geo5_params[:]

            return geo5_params

        except IndexError as e:
            raise RuntimeError(f"Error: Input is missing or invalid - {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Error: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error: {e}")
        

    def process_field_skip_deactivated(self, index, value):
        """
        Enters a value into the specified field, skipping deactivated tabs.
        Compares against the previously entered values to avoid redundant input.
        """
        try:
            # Enter value if it differs from the last recorded value or index is new
            if (
                index >= len(CantileverWall._prev_geometry_params)
                or value != CantileverWall._prev_geometry_params[index]
            ):
                keyboard.send_keys(f"{value}", pause=PAUSE_ENTER)
            time.sleep(0.15)
            # Skip deactivated tabs
            if index not in {5, 6, 8}:
                keyboard.send_keys("{TAB}", pause=PAUSE_TAB)

        except Exception as e:
            print(f"Error entering value for field ({index}): {e}")
    

    def process_field(self, index, value):
        """
        Enters a value into the specified field.
        Handles special cases and conditions per field index.
        Optionally validates the entered value via the clipboard before proceeding.
        """
        try:
            # Special-case pre-processing per field index:
            if index == 2:
                keyboard.send_keys("0,05{TAB 3}4+{TAB 3}", pause=PAUSE_TAB)
            if index == 6:
                keyboard.send_keys("0.05{TAB 2}0+{TAB 2}", pause=PAUSE_TAB)

            # Enter value if it differs from the previous or belongs to special indices
            if (
                index >= len(CantileverWall._prev_geometry_params)
                or value != CantileverWall._prev_geometry_params[index]
                or index in {2,5,6, 8}
            ):
                keyboard.send_keys(f"{value}", pause=PAUSE_ENTER)
            else:
                pass
            time.sleep(0.15)
            # Move to the next field
            keyboard.send_keys("{TAB}", pause=PAUSE_TAB)

        except Exception as e:
            print(f"Error entering value for field ({index}): {e}")

    def profile(self, soil_params):
        """
        Navigates to the profile section and sets the soil depth value.
        """
        soil_depth = soil_params[0]
        self.window.set_focus()
        keyboard.send_keys("{F10}IR", pause=PAUSE_TAB)
        keyboard.send_keys("{TAB}{DOWN}", pause=PAUSE_TAB)
        keyboard.send_keys("+{F10}")
        keyboard.send_keys("E", pause=PAUSE_TAB)
        keyboard.send_keys(f"{soil_depth}", pause=PAUSE_ENTER)
        keyboard.send_keys("{ENTER 2}", pause=PAUSE_ENTER)

            
    def soil(self, soil_params):
        """
        Enters soil properties (unit_weight, friction_angle, cohesion, wall_friction_angle)
        into the relevant GEO5 fields.
        """
        try:
            if len(soil_params) < 2:
                raise ValueError("Input incomplete. At least two values (friction_angle and cohesion) are required.")
            unit_weight = soil_params[0]
            friction_angle = soil_params[1]
            cohesion = soil_params[2]
            wall_friction_angle = round(friction_angle * 2 / 3, 2)

            if not self.window or not self.window.exists():
                raise RuntimeError("Window for value entry is not available.")

            self.window.set_focus()

            # Enter soil parameters
            keyboard.send_keys("{F10}IO{TAB}{SPACE}", pause=PAUSE_TAB)

            keyboard.send_keys("+{F10}")


            keyboard.send_keys("E{TAB}", pause=PAUSE_TAB)
            keyboard.send_keys(f"{unit_weight}", pause=PAUSE_ENTER)
            keyboard.send_keys("{TAB 2}", pause=PAUSE_TAB)
            keyboard.send_keys(f"{friction_angle}", pause=PAUSE_ENTER)  # Enter friction angle
            keyboard.send_keys("{TAB}", pause=PAUSE_TAB)
            keyboard.send_keys(f"{cohesion}", pause=PAUSE_ENTER)  # Enter cohesion
            keyboard.send_keys("{TAB}", pause=PAUSE_TAB)
            keyboard.send_keys(f"{wall_friction_angle}", pause=PAUSE_ENTER)  # Enter wall friction angle
            keyboard.send_keys("{TAB}{LEFT 3}", pause=PAUSE_TAB)
            if cohesion != 0:
                keyboard.send_keys("{RIGHT}", pause=PAUSE_TAB)

            keyboard.send_keys("+{TAB 8}{SPACE}", pause=PAUSE_TAB)

        except IndexError:
            print("Error: Invalid input format. Required index not present.")
        except RuntimeError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    def water(self, wall_params):
        """
        Sets the water level in GEO5 to the value provided in wall_params[0].
        Skips the operation if the value is unchanged from the previous call.
        Validates the entered value via clipboard comparison.
        """
        try:
            if not self.clear_clipboard():
                raise RuntimeError("Clipboard could not be cleared.")

            # New water level value
            value = wall_params[0]

            # Skip if the same value was already entered
            if CantileverWall._prev_water_level == value:
                return

            # Check window availability
            if not self.window or not self.window.exists():
                raise RuntimeError("Window for setting water level is not available.")

            # Focus on the window
            self.window.set_focus()

            # Navigate to water level input and enter value
            keyboard.send_keys("{F10}IW{TAB 2}{SPACE}", pause=PAUSE_TAB)
            keyboard.send_keys("{TAB 4}", pause=PAUSE_TAB)
            keyboard.send_keys(f"{value}", pause=PAUSE_ENTER)

            clipboard_content = self.copy_to_clipboard()
            if clipboard_content is None:
                raise ValueError("Clipboard was not populated; value could not be validated.")
            clipboard_content = clipboard_content.replace(",", ".")
            if clipboard_content != str(value):
                raise ValueError(
                    f"Entered value ({value}) does not match clipboard value ({clipboard_content})."
                )

            # Save the new value
            CantileverWall._prev_water_level = value

        except IndexError:
            print("Error: Invalid input format. Correct index for water level not found.")
        except RuntimeError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    def surcharge(self, surcharge_load):
        """
        Applies a surcharge load in GEO5.
        Skips if the value is unchanged. Creates a new surcharge entry if none exists,
        or updates/removes the existing one depending on the value.
        """
        if CantileverWall._prev_surcharge_load == surcharge_load:
            # If the value is unchanged, skip the operation.
            print(f"Surcharge load unchanged ({surcharge_load} kN/m²), skipping.")
            return

        connected = False
        regex = r"(?=.*GEO5)(?=.*Cantilever Wall)(?=.*guz)"

        # Attempt to connect to the Cantilever Wall interface
        for i in range(5000):
            try:
                app = Application(backend="win32").connect(title_re=regex)
                main_window = app.window(title_re=regex)
                connected = True
                break
            except Exception:
                time.sleep(0.1)

        if not connected:
            print("Failed to connect to GEO5 window.")
            return

        main_window = app.window(title_re=regex)
        dropdown1 = main_window.child_window(class_name="TEnvToolScroller", found_index=1)

        if CantileverWall._prev_surcharge_load == 0:
            # Actions to take when _prev_surcharge_load is zero (no existing surcharge)
            keyboard.send_keys("{F10}IC", pause=0.1)
            rect = dropdown1.rectangle()
            mouse.click(button='left', coords=(rect.left + 30, rect.top + 10))
            keyboard.send_keys("sursaj", pause=0.1)  # Type surcharge name in the first text box
            keyboard.send_keys("{TAB 3}", pause=0.1)  # Press TAB 3 times
            keyboard.send_keys(f"{surcharge_load}", pause=0.1)  # Enter the input value
            keyboard.send_keys("{TAB}{ENTER}", pause=0.1)  # TAB and ENTER
            keyboard.send_keys("{ESC}", pause=0.1)  # ESC

        else:
            if surcharge_load == 0:
                # Actions to take when surcharge_load is zero (remove surcharge)
                keyboard.send_keys("{F10}IC", pause=0.1)
                keyboard.send_keys("{TAB}{SPACE}", pause=0.1)
                rect = dropdown1.rectangle()
                mouse.click(button='left', coords=(rect.left + 230, rect.top + 10))
                keyboard.send_keys("{ENTER}", pause=0.1)

            else:
                # Actions to take when surcharge_load is non-zero (update surcharge)
                keyboard.send_keys("{F10}IC", pause=0.1)
                keyboard.send_keys("{TAB}{SPACE}", pause=0.1)
                rect = dropdown1.rectangle()
                mouse.click(button='left', coords=(rect.left + 125, rect.top + 10))
                keyboard.send_keys("{TAB 3}", pause=0.1)
                keyboard.send_keys(f"{surcharge_load}", pause=0.1)
                keyboard.send_keys("{TAB}{ENTER}", pause=0.1)


        # Update _prev_surcharge_load after operation completes
        CantileverWall._prev_surcharge_load = surcharge_load

    def ff_resistance(self, resistance_params):
        """
        Enters foundation friction resistance parameters (wall_friction_angle, h) into
        the relevant GEO5 fields under the resistance settings tab.
        """
        wall_friction_angle = resistance_params[0]
        h = resistance_params[1]
        keyboard.send_keys("{F10}IE{TAB 9}", pause=PAUSE_TAB)
        keyboard.send_keys(f"{wall_friction_angle}", pause=PAUSE_ENTER)
        keyboard.send_keys("{TAB}", pause=PAUSE_TAB)
        keyboard.send_keys(f"{h}", pause=PAUSE_ENTER)

    def earthquake(self, seismic_params):
        """
        Enters seismic coefficients (kh, kv) into the GEO5 earthquake settings tab.
        """
        kh = seismic_params[0]
        kv = seismic_params[1]
        keyboard.send_keys("{F10}IH{TAB 4}", pause=PAUSE_TAB)
        keyboard.send_keys(f"{kh}", pause=PAUSE_ENTER)
        keyboard.send_keys("{TAB}", pause=PAUSE_TAB)
        keyboard.send_keys(f"{kv}", pause=PAUSE_ENTER)

    def stability(self):

        """Navigates to the stability check window, reads the results, and returns them."""
        self.window.set_focus()
        keyboard.send_keys("{F10}AS{ENTER}{F10}IY", pause=PAUSE_TAB)

        # Attempt to connect to the stability window via regex
        regex = r"(?=.*Stability)(?=.*Cantilever Wall).+"
        connected = False

        for i in range(5000):
            try:
                app = Application(backend="win32").connect(title_re=regex)
                main_window = app.window(title_re=regex)
                connected = True
                break
            except Exception:
                time.sleep(0.1)

        if connected:
            time.sleep(0.01)
            dropdown1 = main_window.child_window(class_name="TStabFrameAnalysis", found_index=0)
            rect = dropdown1.rectangle()
            mouse.click(button='left', coords=(rect.left + 50, rect.top + 90))
            time.sleep(10)

            dropdown2 = None
            rect2 = None
            while rect2 is None:
                try:
                    dropdown2 = main_window.child_window(class_name="TTextInputEx", found_index=1)
                    rect2 = dropdown2.rectangle()
                except Exception:
                    time.sleep(0.01)

            dropdown1 = main_window.child_window(class_name="TEnvToolScroller", found_index=1)
            rect = dropdown1.rectangle()
            mouse.click(button='left', coords=(rect.left + 360, rect.top + 10))
            slip_center_x = self.read_field_value()
            keyboard.send_keys("{TAB}", pause=PAUSE_TAB)
            slip_center_z = self.read_field_value()
            keyboard.send_keys("{TAB}", pause=PAUSE_TAB)
            slip_radius = self.read_field_value()
            keyboard.send_keys("{ESC}", pause=PAUSE_TAB)
            mouse.click(button='left', coords=(rect.right - 50, rect.top + 10))
            time.sleep(0.01)

            keyboard.send_keys("{TAB}", pause=PAUSE_TAB)
            text = self.read_field_value()
            # Parse values according to the expected data format
            fa_match = re.search(r"Sum of active forces\s*:\s*Fa\s*=\s*(\d+,\d+)", text)
            fp_match = re.search(r"Sum of passive forces\s*:\s*Fp\s*=\s*(\d+,\d+)", text)
            ma_match = re.search(r"Sliding moment\s*:\s*Ma\s*=\s*(\d+,\d+)", text)
            mp_match = re.search(r"Resisting moment\s*:\s*Mp\s*=\s*(\d+,\d+)", text)

            values = []
            if fa_match:
                values.append(float(fa_match.group(1).replace(',', '.')))
            if fp_match:
                values.append(float(fp_match.group(1).replace(',', '.')))
            if ma_match:
                values.append(float(ma_match.group(1).replace(',', '.')))
            if mp_match:
                values.append(float(mp_match.group(1).replace(',', '.')))

            
            values.append(float(slip_center_x.replace(',', '.')))
            values.append(float(slip_center_z.replace(',', '.')))
            values.append(float(slip_radius.replace(',', '.')))
            keyboard.send_keys("{TAB}{ENTER}", pause=PAUSE_TAB)
            
            time.sleep(0.5)


            # Build filename and save screenshot
            values_str = "_".join([f"{value:.2f}" for value in values])  # Convert values to string
            screenshot_path = f"screenshots/{CantileverWall._scenario_index}_stability_{values_str}.png"
            CantileverWall._scenario_index = CantileverWall._scenario_index + 1
            # Take and save screenshot
            os.makedirs("screenshots", exist_ok=True)  # Create 'screenshots' directory
            pyautogui.screenshot(screenshot_path)
            print(f"Screenshot saved: {screenshot_path}")            


            keyboard.send_keys("%{F4}{ENTER}", pause=PAUSE_TAB)
            timeout = 30  # Max wait time for operation (seconds)
            for _ in range(timeout):
                stabilite_window_open = self.check_window_open(r"(?=.*Stability)(?=.*Cantilever Wall)")
                if stabilite_window_open:
                    time.sleep(0.01)  # Wait for the window to close
                else:
                    break
            else:
                raise TimeoutError("Window did not close in the specified time. Check operation.")

            return values
        else:
            print("Stability window not found.")
            return []
