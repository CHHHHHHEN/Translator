from __future__ import annotations

import time
import pyperclip
from pynput.keyboard import Key, Controller


class ClipboardController:
    def __init__(self) -> None:
        self.keyboard = Controller()

    def get_selected_text(self) -> str:
        # Clear clipboard first to ensure we get new content
        old_clipboard = pyperclip.paste()
        pyperclip.copy("") 
        
        # Press Ctrl+C
        with self.keyboard.pressed(Key.ctrl):
            self.keyboard.press('c')
            self.keyboard.release('c')
            
        # Wait for clipboard update
        # Retry a few times
        text = ""
        for _ in range(10): # Increased retries
            time.sleep(0.05)
            text = pyperclip.paste()
            if text:
                break
        
        if not text:
            # Restore old if nothing copied (maybe no selection)
            pyperclip.copy(old_clipboard)
            return ""
            
        return text

    def select_all(self) -> None:
        """Sends Ctrl+A to select all text."""
        with self.keyboard.pressed(Key.ctrl):
            self.keyboard.press('a')
            self.keyboard.release('a')
        time.sleep(0.05)
    
    def paste_text(self, text: str) -> None:
        pyperclip.copy(text)
        with self.keyboard.pressed(Key.ctrl):
            self.keyboard.press('v')
            self.keyboard.release('v')


def replace_text(original: str, replacements: dict[str, str]) -> str:
    """Apply a chain of textual replacements."""
    result = original
    for needle, replacement in sorted(replacements.items(), key=lambda item: -len(item[0])):
        result = result.replace(needle, replacement)
    return result
